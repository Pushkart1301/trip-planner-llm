import json
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from utils.evidence_collector import Evidence, SourceTier, SourceType
from utils.verifier import Verifier


NOW = datetime(2026, 10, 4, tzinfo=timezone.utc)


def make_evidence(
    host: str,
    tier: SourceTier,
    supports: bool | None,
    *,
    published_at: datetime | None = NOW - timedelta(days=10),
) -> Evidence:
    source_type = (
        SourceType.OFFICIAL if tier == SourceTier.TIER_1 else SourceType.PUBLICATION
    )
    return Evidence(
        source_title=host,
        source_url=f"https://{host}/fact",
        source_type=source_type,
        source_tier=tier,
        supports_claim=supports,
        relevance="The retrieved page text.",
        retrieved_at=NOW,
        published_at=published_at,
    )


def claim(
    claim_type="OPENING_HOURS",
    temporal_type="SEMI_STABLE",
):
    return {
        "entity_name": "Museum X",
        "location": "Mumbai",
        "claim_type": claim_type,
        "attribute": "opening_hours",
        "claimed_value": "closed on Mondays",
        "temporal_type": temporal_type,
        "claim_text": "Museum X is closed on Mondays.",
    }


class FakeLlm:
    def __init__(self, response):
        self.response = response
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content=self.response)


class VerifierTests(unittest.TestCase):
    def test_one_fresh_tier_one_source_can_approve(self):
        evidence = [make_evidence("museum.gov", SourceTier.TIER_1, True)]
        result = Verifier(None).decide(claim(), evidence, now=NOW)
        self.assertEqual(result["decision"], "APPROVED")
        self.assertIn("Tier 1", result["reasoning"])

    def test_two_independent_fresh_tier_two_or_three_sources_can_approve(self):
        evidence = [
            make_evidence("osm.example", SourceTier.TIER_2, True),
            make_evidence("guide.example", SourceTier.TIER_3, True),
        ]
        result = Verifier(None).decide(claim(), evidence, now=NOW)
        self.assertEqual(result["decision"], "APPROVED")

    def test_one_tier_three_source_is_insufficient(self):
        result = Verifier(None).decide(
            claim(),
            [make_evidence("guide.example", SourceTier.TIER_3, True)],
            now=NOW,
        )
        self.assertEqual(result["decision"], "UNCERTAIN")

    def test_duplicate_hosts_do_not_count_as_independent_sources(self):
        evidence = [
            make_evidence("guide.example", SourceTier.TIER_3, True),
            make_evidence("sub.guide.example", SourceTier.TIER_3, True),
        ]
        self.assertEqual(
            Verifier(None).decide(claim(), evidence, now=NOW)["decision"],
            "UNCERTAIN",
        )

    def test_strong_contradiction_rejects_without_support(self):
        result = Verifier(None).decide(
            claim(),
            [make_evidence("museum.osm.example", SourceTier.TIER_2, False)],
            now=NOW,
        )
        self.assertEqual(result["decision"], "REJECTED")

    def test_conflicting_sources_are_uncertain(self):
        evidence = [
            make_evidence("museum.gov", SourceTier.TIER_1, True),
            make_evidence("osm.example", SourceTier.TIER_2, False),
        ]
        self.assertEqual(
            Verifier(None).decide(claim(), evidence, now=NOW)["decision"],
            "UNCERTAIN",
        )

    def test_high_risk_requires_tier_one_support(self):
        evidence = [
            make_evidence("guide-one.example", SourceTier.TIER_3, True),
            make_evidence("guide-two.example", SourceTier.TIER_3, True),
        ]
        result = Verifier(None).decide(
            claim("VISA_REQUIREMENT"), evidence, now=NOW
        )
        self.assertEqual(result["decision"], "UNCERTAIN")
        self.assertIn("Tier 1", result["reasoning"])

    def test_stale_and_undated_time_sensitive_sources_are_uncertain(self):
        stale = make_evidence(
            "news.example",
            SourceTier.TIER_1,
            True,
            published_at=NOW - timedelta(days=31),
        )
        undated = make_evidence(
            "official.example",
            SourceTier.TIER_1,
            True,
            published_at=None,
        )
        for evidence in (stale, undated):
            with self.subTest(evidence=evidence.source_host):
                result = Verifier(None).decide(
                    claim(temporal_type="TIME_SENSITIVE"), [evidence], now=NOW
                )
                self.assertEqual(result["decision"], "UNCERTAIN")

    def test_network_failures_and_missing_evidence_fail_closed(self):
        evidence = [make_evidence("museum.gov", SourceTier.TIER_1, True)]
        result = Verifier(None).decide(
            claim(), evidence, now=NOW, collection_failures=["search:Timeout"]
        )
        self.assertEqual(result["decision"], "UNCERTAIN")
        self.assertEqual(
            Verifier(None).decide(claim(), [], now=NOW)["decision"], "UNCERTAIN"
        )

    def test_llm_only_assesses_sources_and_must_return_complete_json(self):
        evidence = [
            make_evidence("museum.gov", SourceTier.TIER_1, None),
            make_evidence("guide.example", SourceTier.TIER_3, None),
        ]
        response = json.dumps(
            {
                "assessments": [
                    {"index": 0, "supports_claim": True, "relevance": "Direct statement."},
                    {"index": 1, "supports_claim": None, "relevance": "Unclear."},
                ],
                "reasoning_summary": "One direct source and one unclear source.",
            }
        )
        verifier = Verifier(FakeLlm(response))
        assessed = verifier.summarize_evidence(evidence, claim())

        self.assertIs(assessed[0].supports_claim, True)
        self.assertIsNone(assessed[1].supports_claim)
        self.assertEqual(verifier.decide(claim(), assessed, now=NOW)["decision"], "APPROVED")
        self.assertIn("untrusted data", verifier.llm.messages[0].content)
        self.assertIn("Do not make an overall verification decision", verifier.llm.messages[0].content)

        invalid = Verifier(FakeLlm('{"assessments": [], "reasoning_summary": "ok"}'))
        unchanged = invalid.summarize_evidence(evidence, claim())
        self.assertIsNone(unchanged[0].supports_claim)
        self.assertEqual(invalid.decide(claim(), unchanged, now=NOW)["decision"], "UNCERTAIN")


if __name__ == "__main__":
    unittest.main()
