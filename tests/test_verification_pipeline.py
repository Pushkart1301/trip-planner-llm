import json
from pathlib import Path
import tempfile
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from utils.evidence_collector import Evidence, SourceTier, SourceType
from utils.feedback_repository import FeedbackRepository
from utils.feedback_service import FeedbackService


CLAIM = {
    "entity_name": "Museum X",
    "location": "Mumbai",
    "claim_type": "OPENING_HOURS",
    "attribute": "opening_hours",
    "claimed_value": "closed on Mondays",
    "temporal_type": "SEMI_STABLE",
    "claim_text": "Museum X is closed on Mondays.",
}


class FakeLlm:
    def __init__(self, supports=True):
        self.supports = supports
        self.called = False

    def invoke(self, messages):
        self.called = True
        payload = json.loads(messages[1].content)
        return SimpleNamespace(
            content=json.dumps(
                {
                    "assessments": [
                        {
                            "index": index,
                            "supports_claim": self.supports,
                            "relevance": "The retrieved source directly addresses the claim.",
                        }
                        for index, _ in enumerate(payload["evidence"])
                    ],
                    "reasoning_summary": "Evidence assessment completed.",
                }
            )
        )


class FakeCollector:
    def __init__(self, evidence, failures=None, attempts=1, **kwargs):
        self.evidence = evidence
        self.failures = failures or []
        self.request_attempts = attempts

    def collect_for_claim(self, claim):
        return self.evidence


class VerificationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo = FeedbackRepository(Path(self.temp_dir.name) / "feedback.sqlite3")
        self.repo.initialize()

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_verifying_feedback(self, claim=CLAIM):
        feedback_id = self.repo.create_feedback(
            original_question="What are Museum X's hours?",
            original_answer="It is open every day.",
            feedback_text=claim["claim_text"],
            suggested_correction=claim["claim_text"],
            destination="Mumbai",
            user_id="test-user",
        )
        self.repo.update_feedback_classification(
            feedback_id=feedback_id,
            classification="FACTUAL_CORRECTION",
            claim_json=json.dumps(claim),
            policy_decision="VERIFY",
        )
        self.repo.set_feedback_status(feedback_id, "VERIFYING")
        return feedback_id

    def evidence(self, host="museum.gov", tier=SourceTier.TIER_1):
        return Evidence(
            source_title="Retrieved museum information",
            source_url=f"https://{host}/opening-hours",
            source_type=(
                SourceType.OFFICIAL
                if tier == SourceTier.TIER_1
                else SourceType.STRUCTURED_DATA
            ),
            source_tier=tier,
            supports_claim=None,
            relevance="Museum X is closed on Mondays.",
            retrieved_at=datetime.now(timezone.utc),
            published_at=datetime.now(timezone.utc),
        )

    def test_approved_verification_creates_expiring_memory(self):
        feedback_id = self.create_verifying_feedback()
        llm = FakeLlm(supports=True)
        service = FeedbackService(
            self.repo,
            llm,
            evidence_collector_factory=lambda **kwargs: FakeCollector(
                [self.evidence()], **kwargs
            ),
        )

        result = service.verify_feedback(feedback_id)

        self.assertEqual(result["decision"], "APPROVED")
        self.assertEqual(result["status"], "VERIFIED")
        self.assertIsNotNone(result["memory_id"])
        self.assertTrue(llm.called)
        self.assertEqual(self.repo.get_feedback(feedback_id)["status"], "VERIFIED")
        verification = self.repo.get_verification(result["verification_id"])
        self.assertEqual(verification["decision"], "APPROVED")
        self.assertEqual(verification["source_tier_best"], "TIER_1")
        self.assertEqual(verification["attempts"], 1)
        memory = next(
            item
            for item in self.repo.list_memories(status="ACTIVE")
            if item["id"] == result["memory_id"]
        )
        self.assertEqual(memory["entity_name"], "Museum X")
        self.assertEqual(memory["destination"], "Mumbai")
        self.assertIsNotNone(memory["expires_at"])

    def test_no_evidence_creates_uncertain_verification_without_memory_or_llm(self):
        feedback_id = self.create_verifying_feedback()
        llm = FakeLlm()
        service = FeedbackService(
            self.repo,
            llm,
            evidence_collector_factory=lambda **kwargs: FakeCollector(
                [], **kwargs
            ),
        )

        result = service.verify_feedback(feedback_id)

        self.assertEqual(result["decision"], "UNCERTAIN")
        self.assertEqual(result["status"], "UNCERTAIN")
        self.assertIsNone(result["memory_id"])
        self.assertFalse(llm.called)
        self.assertEqual(self.repo.list_memories(), [])

    def test_source_failure_prevents_memory_even_if_remaining_source_supports(self):
        feedback_id = self.create_verifying_feedback()
        llm = FakeLlm(supports=True)
        service = FeedbackService(
            self.repo,
            llm,
            evidence_collector_factory=lambda **kwargs: FakeCollector(
                [self.evidence()], failures=["web_search:Timeout"], **kwargs
            ),
        )

        result = service.verify_feedback(feedback_id)

        self.assertEqual(result["decision"], "UNCERTAIN")
        self.assertIsNone(result["memory_id"])
        self.assertEqual(self.repo.list_memories(), [])

    def test_verifier_model_failure_fails_closed_without_memory(self):
        feedback_id = self.create_verifying_feedback()
        service = FeedbackService(
            self.repo,
            evidence_collector_factory=lambda **kwargs: FakeCollector(
                [self.evidence()], **kwargs
            ),
        )

        def model_unavailable():
            raise RuntimeError("provider unavailable")

        service._get_llm_client = model_unavailable
        result = service.verify_feedback(feedback_id)

        self.assertEqual(result["decision"], "UNCERTAIN")
        self.assertIsNone(result["memory_id"])
        self.assertEqual(self.repo.list_memories(), [])

    def test_conflicting_evidence_does_not_create_memory(self):
        feedback_id = self.create_verifying_feedback()
        supporting = self.evidence("museum.gov", SourceTier.TIER_1)
        contradicting = self.evidence("osm.example", SourceTier.TIER_2)
        llm = FakeLlm(supports=True)

        class ConflictingLlm(FakeLlm):
            def invoke(self, messages):
                self.called = True
                return SimpleNamespace(
                    content=json.dumps(
                        {
                            "assessments": [
                                {
                                    "index": 0,
                                    "supports_claim": True,
                                    "relevance": "Direct support.",
                                },
                                {
                                    "index": 1,
                                    "supports_claim": False,
                                    "relevance": "Direct contradiction.",
                                },
                            ],
                            "reasoning_summary": "Sources conflict.",
                        }
                    )
                )

        service = FeedbackService(
            self.repo,
            ConflictingLlm(),
            evidence_collector_factory=lambda **kwargs: FakeCollector(
                [supporting, contradicting], **kwargs
            ),
        )

        result = service.verify_feedback(feedback_id)

        self.assertEqual(result["decision"], "UNCERTAIN")
        self.assertIsNone(result["memory_id"])
        self.assertEqual(self.repo.list_memories(), [])


if __name__ == "__main__":
    unittest.main()
