"""Bounded evidence collection from public sources for feedback verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum, IntEnum
from html.parser import HTMLParser
import ipaddress
import json
import logging
import os
import re
import socket
import threading
import time
from typing import Any
from urllib.parse import urlencode, urlparse

import requests


logger = logging.getLogger(__name__)


class SourceType(str, Enum):
    OFFICIAL = "OFFICIAL"
    STRUCTURED_DATA = "STRUCTURED_DATA"
    PUBLICATION = "PUBLICATION"
    BLOG = "BLOG"
    UNKNOWN = "UNKNOWN"


class SourceTier(IntEnum):
    TIER_1 = 1
    TIER_2 = 2
    TIER_3 = 3
    TIER_4 = 4


@dataclass
class Evidence:
    source_title: str
    source_url: str
    source_type: SourceType
    source_tier: SourceTier
    supports_claim: bool | None
    relevance: str
    retrieved_at: datetime
    published_at: datetime | None = None
    summary: str | None = None

    @property
    def source_host(self) -> str:
        return (urlparse(self.source_url).hostname or "").lower()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_type"] = self.source_type.value
        result["source_tier"] = self.source_tier.value
        result["retrieved_at"] = self.retrieved_at.isoformat()
        result["published_at"] = (
            self.published_at.isoformat() if self.published_at else None
        )
        return result


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._hidden_depth = 0
        self.title = ""
        self._inside_title = False
        self.meta: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key.lower(): value or "" for key, value in attrs}
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self._hidden_depth += 1
        if tag.lower() == "title":
            self._inside_title = True
        if tag.lower() == "meta":
            name = (
                attrs_dict.get("property")
                or attrs_dict.get("name")
                or attrs_dict.get("itemprop")
            ).lower()
            value = attrs_dict.get("content", "")
            if name and value:
                self.meta[name] = value

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript", "svg"} and self._hidden_depth:
            self._hidden_depth -= 1
        if lowered == "title":
            self._inside_title = False
        if lowered in {"p", "br", "div", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._inside_title:
            self.title += data
        if not self._hidden_depth:
            normalized = " ".join(data.split())
            if normalized:
                self.parts.append(normalized)


class EvidenceCollector:
    """Collect a small set of retrieved sources; never invent evidence records."""

    MAX_SOURCES = 3
    MAX_RESPONSE_BYTES = 1_000_000
    MAX_EVIDENCE_TEXT = 6_000
    _nominatim_lock = threading.Lock()
    _last_nominatim_request = 0.0
    _nominatim_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
    _nominatim_cache_lock = threading.Lock()
    _http_session = requests.Session()
    TIER_3_DOMAINS = {
        "apnews.com",
        "bbc.com",
        "bbc.co.uk",
        "britannica.com",
        "history.com",
        "lonelyplanet.com",
        "nationalgeographic.com",
        "reuters.com",
        "smithsonianmag.com",
        "theguardian.com",
        "wikipedia.org",
    }
    TIER_4_DOMAINS = {
        "blogspot.com",
        "facebook.com",
        "instagram.com",
        "medium.com",
        "reddit.com",
        "tiktok.com",
        "twitter.com",
        "wordpress.com",
        "x.com",
    }

    def __init__(
        self,
        max_sources: int = 3,
        timeout: float = 5,
        retries: int = 3,
        *,
        search_client: Any | None = None,
        clock: Any = time.monotonic,
        sleeper: Any = time.sleep,
        resolver: Any = socket.getaddrinfo,
    ) -> None:
        self.max_sources = min(max(1, max_sources), self.MAX_SOURCES)
        self.timeout = min(max(float(timeout), 0.1), 15.0)
        self.retries = min(max(int(retries), 1), 3)
        self.search_client = search_client
        self.clock = clock
        self.sleeper = sleeper
        self.resolver = resolver
        self.evidence_list: list[Evidence] = []
        self.failures: list[str] = []
        self.request_attempts = 0

    @property
    def usage_notices(self) -> list[str]:
        if any(
            evidence.source_host == "nominatim.openstreetmap.org"
            for evidence in self.evidence_list
        ):
            return [
                "© OpenStreetMap contributors (ODbL). Nominatim is limited to "
                "one request per second and requires attribution; see "
                "https://operations.osmfoundation.org/policies/nominatim/ and "
                "https://www.openstreetmap.org/copyright."
            ]
        return []

    def collect_for_claim(self, claim_dict: dict[str, Any]) -> list[Evidence]:
        self.evidence_list = []
        self.failures = []
        self.request_attempts = 0

        entity = self._clean_text(claim_dict.get("entity_name"), 200)
        location = self._clean_text(claim_dict.get("location"), 200)
        if not entity:
            return []
        claim_type = claim_dict.get("claim_type")
        query = " ".join(part for part in (entity, location) if part)

        if claim_type in {
            "OPENING_HOURS",
            "ADDRESS",
            "LOCATION",
            "CLOSURE",
        }:
            self._collect_osm(entity, location, claim_type)

        if claim_type in {
            "GENERAL_FACT",
            "LOCATION",
            "ADDRESS",
            "TRAVEL_TIME",
        }:
            self._collect_wikidata(entity, location, claim_type)

        self._collect_web_search(claim_dict)
        return self.evidence_list[: self.max_sources]

    @staticmethod
    def _clean_text(value: Any, max_length: int) -> str:
        if not isinstance(value, str):
            return ""
        return re.sub(r"[\x00-\x1f\x7f]+", " ", value).strip()[:max_length]

    @staticmethod
    def _user_agent() -> str:
        return os.getenv(
            "VERIFICATION_USER_AGENT",
            "AI-Trip-Planner/1.0 (https://github.com/sunnysavita10/AI_Trip_Planner)",
        )

    def _safe_public_url(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or len(url) > 2048
            ):
                return False
            host = parsed.hostname.rstrip(".").lower()
            if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
                return False

            try:
                addresses = [ipaddress.ip_address(host)]
            except ValueError:
                addresses = [
                    ipaddress.ip_address(answer[4][0])
                    for answer in self.resolver(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
                ]
            return bool(addresses) and all(address.is_global for address in addresses)
        except (OSError, ValueError, TypeError):
            return False

    @staticmethod
    def _source_classification(url: str) -> tuple[SourceType, SourceTier]:
        host = (urlparse(url).hostname or "").lower().rstrip(".")
        configured_official = {
            domain.strip().lower().lstrip(".").rstrip(".")
            for domain in os.getenv("VERIFICATION_OFFICIAL_DOMAINS", "").split(",")
            if domain.strip()
        }
        configured_reputable = {
            domain.strip().lower().lstrip(".").rstrip(".")
            for domain in os.getenv("VERIFICATION_REPUTABLE_DOMAINS", "").split(",")
            if domain.strip()
        }
        is_configured_official = any(
            host == domain or host.endswith("." + domain)
            for domain in configured_official
        )
        government_suffixes = (
            ".gov",
            ".gov.uk",
            ".gov.in",
            ".gov.au",
            ".govt.nz",
            ".gov.ca",
            ".gov.sg",
            ".gov.ph",
            ".gov.my",
            ".gov.za",
            ".gov.br",
            ".gov.jp",
            ".gov.kr",
            ".gov.cn",
            ".gov.hk",
            ".gov.tw",
            ".gov.pk",
            ".gov.bd",
            ".gov.np",
            ".gov.lk",
            ".gov.ie",
            ".gov.de",
            ".nic.in",
        )
        is_government = any(host.endswith(suffix) for suffix in government_suffixes)
        if is_configured_official or is_government:
            return SourceType.OFFICIAL, SourceTier.TIER_1
        if any(
            host == domain or host.endswith("." + domain)
            for domain in configured_reputable | EvidenceCollector.TIER_3_DOMAINS
        ):
            return SourceType.PUBLICATION, SourceTier.TIER_3
        if any(
            host == domain or host.endswith("." + domain)
            for domain in EvidenceCollector.TIER_4_DOMAINS
        ):
            return SourceType.BLOG, SourceTier.TIER_4
        return SourceType.UNKNOWN, SourceTier.TIER_4

    def _request_bytes(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        max_bytes: int | None = None,
        expected_content_types: tuple[str, ...] | None = None,
    ) -> bytes:
        if not self._safe_public_url(url):
            raise ValueError("URL is not a validated public HTTP(S) address")
        limit = max_bytes or self.MAX_RESPONSE_BYTES
        last_error: Exception | None = None

        for attempt in range(self.retries):
            self.request_attempts += 1
            response = None
            try:
                if urlparse(url).hostname == "nominatim.openstreetmap.org":
                    with self._nominatim_lock:
                        delay = 1.0 - (
                            self.clock() - self._last_nominatim_request
                        )
                        if delay > 0:
                            self.sleeper(delay)
                        type(self)._last_nominatim_request = self.clock()
                response = self._http_session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=(min(self.timeout, 5), self.timeout),
                    allow_redirects=False,
                    stream=True,
                )
                if response.is_redirect or response.is_permanent_redirect:
                    raise ValueError("Evidence source redirected; redirect refused")
                response.raise_for_status()
                if expected_content_types:
                    content_type = response.headers.get("Content-Type", "").lower()
                    if not any(
                        content_type.startswith(expected_type)
                        for expected_type in expected_content_types
                    ):
                        raise ValueError("Evidence page is not an HTML document")
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > limit:
                    raise ValueError("Evidence response exceeded size limit")
                content = bytearray()
                for chunk in response.iter_content(chunk_size=16_384):
                    content.extend(chunk)
                    if len(content) > limit:
                        raise ValueError("Evidence response exceeded size limit")
                return bytes(content)
            except (requests.RequestException, ValueError, OSError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    self.sleeper(min(0.5 * (2**attempt), 2.0))
            finally:
                if response is not None:
                    response.close()

        assert last_error is not None
        raise last_error

    def _request_json(
        self,
        url: str,
        *,
        params: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> Any:
        raw = self._request_bytes(url, params=params, headers=headers)
        return json.loads(raw.decode("utf-8"))

    def _collect_osm(self, entity: str, location: str, claim_type: str) -> None:
        if len(self.evidence_list) >= self.max_sources:
            return
        query = ", ".join(part for part in (entity, location) if part)
        now = self.clock()
        with self._nominatim_cache_lock:
            cached = self._nominatim_cache.get(query.lower())
            if cached and cached[0] > now:
                results = cached[1]
            else:
                results = None
        if results is None:
            try:
                params = {
                    "q": query,
                    "format": "jsonv2",
                    "addressdetails": 1,
                    "limit": 1,
                }
                payload = self._request_json(
                    "https://nominatim.openstreetmap.org/search",
                    params=params,
                    headers={"User-Agent": self._user_agent()},
                )
                if not isinstance(payload, list):
                    raise ValueError("Unexpected Nominatim response")
                results = payload
                with self._nominatim_cache_lock:
                    self._nominatim_cache[query.lower()] = (
                        self.clock() + 86_400,
                        results,
                    )
            except (requests.RequestException, ValueError, OSError, json.JSONDecodeError) as exc:
                self.failures.append(f"nominatim:{type(exc).__name__}")
                return

        if not results:
            return
        item = results[0]
        display_name = self._clean_text(item.get("display_name"), 1000)
        address = item.get("address")
        address_text = (
            self._clean_text(", ".join(str(value) for value in address.values()), 1000)
            if isinstance(address, dict)
            else ""
        )
        if claim_type in {"OPENING_HOURS", "CLOSURE"}:
            evidence_text = "OpenStreetMap search result does not expose opening-hours tags."
        else:
            evidence_text = " | ".join(part for part in (display_name, address_text) if part)
        if not evidence_text:
            return
        evidence_text = "© OpenStreetMap contributors. " + evidence_text
        self.evidence_list.append(
            Evidence(
                source_title="OpenStreetMap Nominatim search result",
                source_url="https://nominatim.openstreetmap.org/search?"
                + urlencode(
                    {"q": query, "format": "jsonv2", "addressdetails": 1, "limit": 1}
                ),
                source_type=SourceType.STRUCTURED_DATA,
                source_tier=SourceTier.TIER_2,
                supports_claim=None,
                relevance=evidence_text[: self.MAX_EVIDENCE_TEXT],
                retrieved_at=datetime.now(timezone.utc),
            )
        )

    def _collect_wikidata(
        self, entity_name: str, location: str, claim_type: str
    ) -> None:
        if len(self.evidence_list) >= self.max_sources:
            return
        headers = {
            "User-Agent": self._user_agent(),
            "Accept": "application/json",
        }
        try:
            query = " ".join(part for part in (entity_name, location) if part)
            search_payload = self._request_json(
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbsearchentities",
                    "search": query,
                    "language": "en",
                    "limit": 1,
                    "format": "json",
                },
                headers=headers,
            )
            results = search_payload.get("search", [])
            if not results:
                return
            entity_id = self._clean_text(results[0].get("id"), 20)
            if not re.fullmatch(r"Q[1-9]\d*", entity_id):
                return
            entity_payload = self._request_json(
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbgetentities",
                    "ids": entity_id,
                    "props": "labels|descriptions|claims|info",
                    "languages": "en",
                    "format": "json",
                },
                headers=headers,
            )
            entity = entity_payload.get("entities", {}).get(entity_id, {})
            label = self._clean_text(
                entity.get("labels", {}).get("en", {}).get("value"), 200
            )
            description = self._clean_text(
                entity.get("descriptions", {}).get("en", {}).get("value"), 400
            )
            claim_properties = {
                "LOCATION": {
                    "P17": "country",
                    "P131": "administrative area",
                    "P276": "location",
                    "P625": "coordinates",
                },
                "ADDRESS": {
                    "P6375": "street address",
                    "P131": "administrative area",
                    "P276": "location",
                    "P17": "country",
                },
                "TRAVEL_TIME": {"P2047": "duration"},
                "GENERAL_FACT": {
                    "P17": "country",
                    "P31": "instance of",
                    "P84": "architect",
                    "P131": "administrative area",
                    "P154": "logo",
                    "P276": "location",
                    "P571": "inception",
                },
            }.get(claim_type, {})
            claims = entity.get("claims", {})
            statements: list[str] = []
            for property_id, property_label in claim_properties.items():
                for statement in claims.get(property_id, [])[:3]:
                    mainsnak = statement.get("mainsnak", {})
                    datavalue = mainsnak.get("datavalue", {})
                    value = datavalue.get("value")
                    if value is None:
                        continue
                    if isinstance(value, dict) and value.get("id"):
                        rendered_value = value["id"]
                    elif isinstance(value, dict) and {
                        "latitude",
                        "longitude",
                    }.issubset(value):
                        rendered_value = (
                            f'{value["latitude"]}, {value["longitude"]}'
                        )
                    elif isinstance(value, dict) and "amount" in value:
                        rendered_value = str(value["amount"])
                    elif isinstance(value, dict) and "time" in value:
                        rendered_value = str(value["time"])
                    else:
                        rendered_value = str(value)
                    statements.append(
                        f"{property_label}: {self._clean_text(rendered_value, 300)}"
                    )
            value_ids = {
                item.split(": ", 1)[1]
                for item in statements
                if re.fullmatch(r"Q[1-9]\d*", item.split(": ", 1)[-1])
            }
            if value_ids:
                labels_payload = self._request_json(
                    "https://www.wikidata.org/w/api.php",
                    params={
                        "action": "wbgetentities",
                        "ids": "|".join(sorted(value_ids)[:20]),
                        "props": "labels",
                        "languages": "en",
                        "format": "json",
                    },
                    headers=headers,
                )
                value_labels = {
                    identifier: self._clean_text(
                        value.get("labels", {}).get("en", {}).get("value"), 200
                    )
                    for identifier, value in labels_payload.get(
                        "entities", {}
                    ).items()
                }
                statements = [
                    item.split(": ", 1)[0]
                    + ": "
                    + value_labels.get(item.split(": ", 1)[1], item.split(": ", 1)[1])
                    if item.split(": ", 1)[-1] in value_labels
                    else item
                    for item in statements
                ]
            revision_id = entity.get("lastrevid")
            published_at = None
            if revision_id:
                revision_payload = self._request_json(
                    "https://www.wikidata.org/w/api.php",
                    params={
                        "action": "query",
                        "prop": "revisions",
                        "revids": revision_id,
                        "rvprop": "timestamp",
                        "format": "json",
                    },
                    headers=headers,
                )
                pages = revision_payload.get("query", {}).get("pages", {})
                page = next(iter(pages.values()), {})
                revisions = page.get("revisions", [])
                if revisions:
                    published_at = self._parse_date(
                        revisions[0].get("timestamp")
                    )
            if not statements:
                return
            retrieved_text = "; ".join(statements)
            if description:
                retrieved_text = f"{description}. {retrieved_text}"
            if label:
                retrieved_text = f"{label}: {retrieved_text}"
            self.evidence_list.append(
                Evidence(
                    source_title=f"Wikidata: {label or entity_id}",
                    source_url=f"https://www.wikidata.org/wiki/{entity_id}",
                    source_type=SourceType.STRUCTURED_DATA,
                    source_tier=SourceTier.TIER_2,
                    supports_claim=None,
                    relevance=retrieved_text[: self.MAX_EVIDENCE_TEXT],
                    retrieved_at=datetime.now(timezone.utc),
                    published_at=published_at,
                )
            )
        except (requests.RequestException, ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
            self.failures.append(f"wikidata:{type(exc).__name__}")

    def _get_search_client(self) -> Any:
        if self.search_client is not None:
            return self.search_client
        from ddgs import DDGS

        return DDGS(timeout=self.timeout)

    def _collect_web_search(self, claim: dict[str, Any]) -> None:
        if len(self.evidence_list) >= self.max_sources:
            return
        parts = [
            self._clean_text(claim.get(key), 150)
            for key in ("entity_name", "location", "attribute", "claimed_value")
        ]
        query = " ".join(part for part in parts if part)
        results = None
        for attempt in range(self.retries):
            try:
                self.request_attempts += 1
                client = self._get_search_client()
                results = client.text(query, max_results=self.MAX_SOURCES)
                break
            except Exception as exc:
                if attempt + 1 == self.retries:
                    self.failures.append(f"web_search:{type(exc).__name__}")
                else:
                    self.sleeper(min(0.5 * (2**attempt), 2.0))
        if results is None:
            return

        visited_hosts = {evidence.source_host for evidence in self.evidence_list}
        for result in results:
            if len(self.evidence_list) >= self.max_sources:
                break
            if not isinstance(result, dict):
                continue
            url = result.get("href")
            if not isinstance(url, str) or not self._safe_public_url(url):
                continue
            host = (urlparse(url).hostname or "").lower()
            if host in visited_hosts:
                continue
            fetched = self._fetch_page(url)
            if fetched is None:
                continue
            text, published_at = fetched
            if not text:
                continue
            source_type, source_tier = self._source_classification(url)
            self.evidence_list.append(
                Evidence(
                    source_title=self._clean_text(result.get("title"), 200) or host,
                    source_url=url,
                    source_type=source_type,
                    source_tier=source_tier,
                    supports_claim=None,
                    relevance=text[: self.MAX_EVIDENCE_TEXT],
                    retrieved_at=datetime.now(timezone.utc),
                    published_at=published_at,
                )
            )
            visited_hosts.add(host)

    def _fetch_page(self, url: str) -> tuple[str, datetime | None] | None:
        try:
            content = self._request_bytes(
                url,
                headers={
                    "User-Agent": self._user_agent(),
                    "Accept": "text/html,application/xhtml+xml",
                },
                max_bytes=self.MAX_RESPONSE_BYTES,
                expected_content_types=("text/html", "application/xhtml+xml"),
            )
            parser = _VisibleTextParser()
            parser.feed(content.decode("utf-8", errors="replace"))
            page_text = " ".join(" ".join(parser.parts).split())
            published_at = self._parse_date(
                parser.meta.get("article:published_time")
                or parser.meta.get("datepublished")
                or parser.meta.get("date")
                or parser.meta.get("article:modified_time")
            )
            try:
                import trafilatura
                from trafilatura.metadata import extract_metadata

                document = content.decode("utf-8", errors="replace")
                extracted = trafilatura.extract(
                    document, include_comments=False, include_tables=False
                )
                metadata = extract_metadata(document)
                if metadata and metadata.date:
                    published_at = self._parse_date(metadata.date) or published_at
                if extracted:
                    page_text = extracted
            except ImportError:
                pass
            return page_text[: self.MAX_EVIDENCE_TEXT], published_at
        except (requests.RequestException, ValueError, OSError) as exc:
            self.failures.append(f"page_fetch:{type(exc).__name__}")
            return None

    @staticmethod
    def _parse_date(value: Any) -> datetime | None:
        if not isinstance(value, str) or not value.strip():
            return None
        normalized = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            for pattern in ("%Y-%m-%d", "%Y/%m/%d", "%B %d, %Y"):
                try:
                    parsed = datetime.strptime(normalized[:40], pattern)
                    break
                except ValueError:
                    continue
            else:
                return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
