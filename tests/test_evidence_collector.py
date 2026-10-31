import json
import unittest
from unittest.mock import patch

import requests

from utils.evidence_collector import (
    EvidenceCollector,
    SourceType,
    SourceTier,
)


class FakeResponse:
    def __init__(self, content=b"[]", headers=None, status_code=200):
        self.content = content
        self.headers = headers or {}
        self.status_code = status_code
        self.is_redirect = False
        self.is_permanent_redirect = False
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}")

    def iter_content(self, chunk_size):
        yield self.content

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class FakeSearch:
    def __init__(self, results):
        self.results = results

    def text(self, query, max_results):
        return self.results[:max_results]


class EvidenceCollectorTests(unittest.TestCase):
    def setUp(self):
        self.public_resolver = lambda host, port, type: [
            (None, None, None, None, ("93.184.216.34", port))
        ]

    def test_blocks_local_and_private_targets_and_allows_public_host(self):
        collector = EvidenceCollector(resolver=self.public_resolver)
        self.assertFalse(collector._safe_public_url("http://127.0.0.1/a"))
        self.assertFalse(collector._safe_public_url("http://192.168.1.4/a"))
        self.assertFalse(collector._safe_public_url("file:///C:/secret"))
        self.assertFalse(collector._safe_public_url("http://localhost/a"))
        self.assertTrue(collector._safe_public_url("https://example.org/a"))

        private_dns = EvidenceCollector(
            resolver=lambda *args, **kwargs: [
                (None, None, None, None, ("10.0.0.4", 443))
            ]
        )
        self.assertFalse(private_dns._safe_public_url("https://attacker.example"))

    def test_source_tiers_are_conservative_and_allow_configured_domains(self):
        with patch.dict(
            "os.environ",
            {
                "VERIFICATION_OFFICIAL_DOMAINS": "museum.example",
                "VERIFICATION_REPUTABLE_DOMAINS": "trusted.example",
            },
        ):
            self.assertEqual(
                EvidenceCollector._source_classification("https://museum.gov.in"),
                (SourceType.OFFICIAL, SourceTier.TIER_1),
            )
            self.assertEqual(
                EvidenceCollector._source_classification("https://museum.example/about")[1],
                SourceTier.TIER_1,
            )
            self.assertEqual(
                EvidenceCollector._source_classification("https://trusted.example/article"),
                (SourceType.PUBLICATION, SourceTier.TIER_3),
            )
        self.assertEqual(
            EvidenceCollector._source_classification("https://bbc.com/travel")[1],
            SourceTier.TIER_3,
        )
        self.assertEqual(
            EvidenceCollector._source_classification("https://x.com/post")[1],
            SourceTier.TIER_4,
        )
        self.assertEqual(
            EvidenceCollector._source_classification("https://fake.gov.com/news")[1],
            SourceTier.TIER_4,
        )

    def test_retries_are_bounded_and_redirects_are_refused(self):
        session = FakeSession(
            [requests.Timeout("timeout"), requests.Timeout("timeout"), requests.Timeout("timeout")]
        )
        sleeps = []
        collector = EvidenceCollector(
            retries=3,
            search_client=FakeSearch([]),
            resolver=self.public_resolver,
            sleeper=sleeps.append,
        )
        with patch.object(EvidenceCollector, "_http_session", session):
            with self.assertRaises(requests.Timeout):
                collector._request_bytes("https://example.org/data")
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(collector.request_attempts, 3)
        self.assertEqual(len(sleeps), 2)

        redirect = FakeResponse(headers={"Location": "http://127.0.0.1/"})
        redirect.is_redirect = True
        collector = EvidenceCollector(
            retries=1, resolver=self.public_resolver, sleeper=lambda _: None
        )
        with patch.object(EvidenceCollector, "_http_session", FakeSession([redirect])):
            with self.assertRaisesRegex(ValueError, "redirect refused"):
                collector._request_bytes("https://example.org/redirect")

    def test_osm_produces_only_retrieved_address_content(self):
        payload = json.dumps(
            [
                {
                    "display_name": "Taj Mahal, Agra, Uttar Pradesh, India",
                    "address": {"city": "Agra", "state": "Uttar Pradesh"},
                    "type": "attraction",
                }
            ]
        ).encode()
        collector = EvidenceCollector(
            max_sources=3,
            retries=1,
            resolver=self.public_resolver,
            sleeper=lambda _: None,
        )
        collector._last_nominatim_request = 0
        collector._nominatim_cache.clear()
        with patch.object(
            EvidenceCollector,
            "_http_session",
            FakeSession([FakeResponse(payload)]),
        ):
            collector._collect_osm("Taj Mahal", "Agra", "LOCATION")

        self.assertEqual(len(collector.evidence_list), 1)
        evidence = collector.evidence_list[0]
        self.assertEqual(evidence.source_tier, SourceTier.TIER_2)
        self.assertIn("Taj Mahal, Agra", evidence.relevance)
        self.assertIn("nominatim.openstreetmap.org", evidence.source_url)
        self.assertIsNone(evidence.supports_claim)
        self.assertNotIn("opening_hours", evidence.relevance)
        self.assertIn("OpenStreetMap contributors", evidence.relevance)
        self.assertIn("one request per second", collector.usage_notices[0])

    def test_wikidata_only_emits_retrieved_structured_statements(self):
        payloads = iter(
            [
                {"search": [{"id": "Q123"}]},
                {
                    "entities": {
                        "Q123": {
                            "labels": {"en": {"value": "Museum X"}},
                            "descriptions": {"en": {"value": "museum in Mumbai"}},
                            "claims": {
                                "P276": [
                                    {
                                        "mainsnak": {
                                            "datavalue": {
                                                "value": {"entity-type": "item", "id": "Q456"}
                                            }
                                        }
                                    }
                                ]
                            },
                            "lastrevid": 99,
                        }
                    }
                },
                {"entities": {"Q456": {"labels": {"en": {"value": "Mumbai"}}}}},
                {
                    "query": {
                        "pages": {
                            "1": {
                                "revisions": [
                                    {"timestamp": "2026-09-30T12:00:00Z"}
                                ]
                            }
                        }
                    }
                },
            ]
        )
        calls = []
        collector = EvidenceCollector(resolver=self.public_resolver)

        def fake_json(url, *, params, headers=None):
            calls.append((url, params, headers))
            return next(payloads)

        collector._request_json = fake_json
        collector._collect_wikidata("Museum X", "Mumbai", "LOCATION")

        self.assertEqual(len(collector.evidence_list), 1)
        evidence = collector.evidence_list[0]
        self.assertEqual(evidence.source_tier, SourceTier.TIER_2)
        self.assertIn("Museum X", evidence.relevance)
        self.assertIn("location: Mumbai", evidence.relevance)
        self.assertEqual(
            evidence.published_at.isoformat(), "2026-09-30T12:00:00+00:00"
        )
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(call[2]["User-Agent"].startswith("AI-Trip-Planner") for call in calls))

    def test_web_search_fetches_only_valid_distinct_hosts_and_caps_sources(self):
        collector = EvidenceCollector(
            max_sources=2,
            search_client=FakeSearch(
                [
                    {"title": "Official fact", "href": "https://visit.example/first"},
                    {"title": "Duplicate host", "href": "https://visit.example/second"},
                    {"title": "Another source", "href": "https://museum.example/info"},
                    {"title": "Local target", "href": "http://127.0.0.1/secret"},
                ]
            ),
            resolver=self.public_resolver,
        )
        collector._fetch_page = lambda url: ("Retrieved source text.", None)
        collector._collect_web_search(
            {
                "entity_name": "Museum X",
                "location": "Mumbai",
                "attribute": "opening hours",
                "claimed_value": "closed Mondays",
            }
        )

        self.assertEqual(len(collector.evidence_list), 2)
        self.assertEqual(
            {item.source_host for item in collector.evidence_list},
            {"visit.example", "museum.example"},
        )
        self.assertTrue(all(item.relevance == "Retrieved source text." for item in collector.evidence_list))
        self.assertGreaterEqual(collector.clock(), 0)

    def test_unavailable_search_is_reported_after_retries(self):
        class BrokenSearch:
            def __init__(self):
                self.calls = 0

            def text(self, query, max_results):
                self.calls += 1
                raise RuntimeError("offline")

        search = BrokenSearch()
        collector = EvidenceCollector(
            retries=3, search_client=search, sleeper=lambda _: None
        )
        collector._collect_web_search(
            {
                "entity_name": "Museum X",
                "location": "Mumbai",
                "attribute": "hours",
                "claimed_value": "closed",
            }
        )
        self.assertEqual(search.calls, 3)
        self.assertEqual(collector.evidence_list, [])
        self.assertEqual(collector.failures, ["web_search:RuntimeError"])


if __name__ == "__main__":
    unittest.main()
