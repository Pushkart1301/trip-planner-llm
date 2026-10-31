from pathlib import Path
import tempfile
import unittest

from utils.feedback_repository import FeedbackRepository
from utils.memory_retriever import MemoryRetriever


class MemoryRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = FeedbackRepository(
            Path(self.temp_dir.name) / "feedback.sqlite3"
        )
        self.repository.initialize()

    def tearDown(self):
        self.temp_dir.cleanup()

    def add_memory(
        self,
        *,
        user_id: str = "anonymous",
        destination: str = "Mumbai, India",
        memory_text: str = "The museum is closed on Mondays.",
        temporal_type: str = "SEMI_STABLE",
    ) -> str:
        feedback_id = self.repository.create_feedback(
            user_id=user_id,
            original_question=f"Travel in {destination}",
            original_answer="The museum is open daily.",
            feedback_text=memory_text,
            suggested_correction=memory_text,
            destination=destination,
        )
        verification_id = self.repository.create_verification(
            feedback_id=feedback_id,
            claim={"claim_text": memory_text},
            decision="APPROVED",
            reasoning_summary="Supported by reliable evidence.",
            evidence=[{"url": "https://example.org"}],
            source_tier_best="T1",
            verified_at="2026-10-01T00:00:00+00:00",
        )
        return self.repository.create_verified_memory(
            verification_id,
            memory_text,
            "City Museum",
            destination,
            "OPENING_HOURS",
            temporal_type,
        )

    def test_retrieves_matching_active_memory_for_correct_user_only(self):
        memory_id = self.add_memory()
        retriever = MemoryRetriever(self.repository)

        matches = retriever.retrieve_for_question(
            "Plan three days in Mumbai, please.", limit=3
        )
        self.assertEqual([memory["id"] for memory in matches], [memory_id])
        self.assertEqual(
            retriever.retrieve_for_question(
                "Plan three days in Delhi.", limit=3
            ),
            [],
        )
        other_user = MemoryRetriever(self.repository, user_id="another-user")
        self.assertEqual(
            other_user.retrieve_for_question("Trip in Mumbai"),
            [],
        )

    def test_excludes_expired_memory_and_formats_prompt_data(self):
        memory_id = self.add_memory()
        retriever = MemoryRetriever(self.repository)
        matches = retriever.retrieve_relevant_memories("Mumbai")
        self.assertEqual([memory["id"] for memory in matches], [memory_id])
        prompt_block = retriever.format_for_prompt(matches)
        self.assertIn("APPROVED TRAVEL CORRECTION NOTES", prompt_block)
        self.assertIn("The museum is closed on Mondays.", prompt_block)

        memory = self.repository.list_memories()[0]
        self.repository.expire_memories(now=memory["expires_at"])
        self.assertEqual(
            retriever.retrieve_for_question("Plan a trip to Mumbai"),
            [],
        )

