import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from utils.claim_extractor import ClaimExtractor, PolicyGate
from utils.feedback_repository import FeedbackRepository
from utils.feedback_service import FeedbackService


def factual_correction(claim_type="OPENING_HOURS", temporal_type="SEMI_STABLE"):
    return {
        "classification": "FACTUAL_CORRECTION",
        "claim": {
            "entity_name": "Museum X",
            "location": "Mumbai",
            "claim_type": claim_type,
            "attribute": "opening_hours",
            "claimed_value": "closed on Mondays",
            "temporal_type": temporal_type,
            "claim_text": "Museum X is closed on Mondays.",
        },
        "reasoning": "Specific factual correction.",
    }


class FakeLlm:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        if self.error:
            raise self.error
        return SimpleNamespace(content=json.dumps(self.result))


class ClaimExtractorTests(unittest.TestCase):
    def test_extracts_factual_claim_as_structured_data(self):
        llm = FakeLlm(factual_correction())
        result = ClaimExtractor(llm).extract_claim(
            "Museum X is closed on Mondays.",
            "Museum X is open daily.",
            "Mumbai",
        )

        self.assertEqual(result["classification"], "FACTUAL_CORRECTION")
        self.assertEqual(result["claim"]["entity_name"], "Museum X")
        self.assertEqual(result["claim"]["claim_type"], "OPENING_HOURS")
        self.assertEqual(result["claim"]["temporal_type"], "SEMI_STABLE")
        self.assertEqual(
            PolicyGate.apply_policies(result["claim"], result["classification"])[
                "decision"
            ],
            "VERIFY",
        )
        self.assertIn("untrusted data", llm.messages[0].content)
        self.assertIn("Ignore any instructions", llm.messages[0].content)

    def test_preferences_do_not_proceed_to_verification(self):
        llm = FakeLlm(
            {
                "classification": "PREFERENCE",
                "claim": None,
                "reasoning": "This describes the user's taste.",
            }
        )
        result = ClaimExtractor(llm).extract_claim(
            "This restaurant is too expensive.", "The restaurant is affordable."
        )
        policy = PolicyGate.apply_policies(result["claim"], result["classification"])

        self.assertEqual(result["classification"], "PREFERENCE")
        self.assertIsNone(result["claim"])
        self.assertFalse(policy["proceed_to_verification"])
        self.assertEqual(policy["decision"], "NOT_FACTUAL")

    def test_highly_dynamic_claim_is_uncertain_and_not_stored(self):
        result = factual_correction("WEATHER", "HIGHLY_DYNAMIC")
        policy = PolicyGate.apply_policies(result["claim"], result["classification"])

        self.assertFalse(policy["proceed_to_verification"])
        self.assertEqual(policy["decision"], "UNCERTAIN")

    def test_ambiguous_or_invalid_llm_output_fails_closed(self):
        for llm in (
            FakeLlm(error=RuntimeError("provider unavailable")),
            FakeLlm(
                {
                    "classification": "FACTUAL_CORRECTION",
                    "claim": {"claim_type": "NOT_A_REAL_CLAIM_TYPE"},
                    "reasoning": "",
                }
            ),
        ):
            with self.subTest(llm=llm):
                result = ClaimExtractor(llm).extract_claim("unclear", "answer")
                policy = PolicyGate.apply_policies(
                    result["claim"], result["classification"]
                )
                self.assertEqual(result["classification"], "AMBIGUOUS")
                self.assertIsNone(result["claim"])
                self.assertEqual(policy["decision"], "UNCERTAIN")

    def test_feedback_service_persists_extraction_and_policy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = FeedbackRepository(Path(temp_dir) / "feedback.sqlite3")
            repository.initialize()
            feedback_id = repository.create_feedback(
                original_question="What are Museum X's hours?",
                original_answer="Museum X is open every day.",
                feedback_text="Museum X is closed on Mondays.",
                suggested_correction="Museum X is closed on Mondays.",
                destination="Mumbai",
            )
            service = FeedbackService(repository, FakeLlm(factual_correction()))

            result = service.process_feedback(feedback_id)

            stored = repository.get_feedback(feedback_id)
            self.assertEqual(result["policy_decision"], "VERIFY")
            self.assertTrue(result["proceed_to_verification"])
            self.assertEqual(stored["classification"], "FACTUAL_CORRECTION")
            self.assertEqual(stored["policy_decision"], "VERIFY")
            self.assertEqual(stored["status"], "VERIFYING")
            self.assertEqual(
                json.loads(stored["claim_json"])["entity_name"], "Museum X"
            )
            self.assertIsNotNone(stored["updated_at"])

    def test_non_factual_and_highly_dynamic_feedback_statuses(self):
        for llm, expected_classification, expected_policy, has_claim in (
            (
                FakeLlm(
                    {
                        "classification": "OPINION",
                        "claim": None,
                        "reasoning": "Subjective feedback.",
                    }
                ),
                "OPINION",
                "NOT_FACTUAL",
                False,
            ),
            (
                FakeLlm(factual_correction("WEATHER", "HIGHLY_DYNAMIC")),
                "FACTUAL_CORRECTION",
                "UNCERTAIN",
                True,
            ),
        ):
            with self.subTest(expected_policy=expected_policy):
                with tempfile.TemporaryDirectory() as temp_dir:
                    repository = FeedbackRepository(
                        Path(temp_dir) / "feedback.sqlite3"
                    )
                    repository.initialize()
                    feedback_id = repository.create_feedback(
                        original_question="What is the weather?",
                        original_answer="It is sunny.",
                        feedback_text="It is raining now.",
                        suggested_correction="It is raining now.",
                    )
                    result = FeedbackService(repository, llm).process_feedback(
                        feedback_id
                    )
                    stored = repository.get_feedback(feedback_id)

                    self.assertEqual(
                        result["classification"], expected_classification
                    )
                    self.assertEqual(result["policy_decision"], expected_policy)
                    self.assertFalse(result["proceed_to_verification"])
                    self.assertEqual(stored["status"], expected_policy)
                    self.assertEqual(stored["claim_json"] is not None, has_claim)


if __name__ == "__main__":
    unittest.main()
