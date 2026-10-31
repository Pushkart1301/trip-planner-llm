from pathlib import Path
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from utils.feedback_repository import FeedbackRepository


class FeedbackRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = FeedbackRepository(
            Path(self.temp_dir.name) / "feedback.sqlite3"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_feedback(self, *, user_id: str = "user-1") -> str:
        return self.repository.create_feedback(
            user_id=user_id,
            session_id="session-1",
            message_id="message-1",
            original_question="Plan a trip to Goa",
            original_answer="The museum is open every day.",
            feedback_text="It is closed on Mondays.",
            suggested_correction="The museum is closed on Mondays.",
            destination="Goa",
        )

    def create_verification(self, feedback_id: str, date: str) -> str:
        return self.repository.create_verification(
            feedback_id=feedback_id,
            claim={"entity": "City Museum", "attribute": "opening day"},
            decision="APPROVED",
            reasoning_summary="Supported by an official source.",
            evidence=[{"url": "https://example.org"}],
            source_tier_best="T1",
            verified_at=f"{date}T00:00:00+00:00",
        )

    def create_memory(
        self, verification_id: str, temporal_type: str = "SEMI_STABLE"
    ) -> str:
        return self.repository.create_verified_memory(
            verification_id,
            "City Museum is closed on Mondays.",
            "City Museum",
            "Goa",
            "OPENING_HOURS",
            temporal_type,
        )

    def test_schema_is_idempotent_and_feedback_is_persisted(self):
        self.repository.initialize()
        self.repository.initialize()
        feedback_id = self.create_feedback()

        feedback = self.repository.get_feedback(feedback_id)
        self.assertIsNotNone(feedback)
        self.assertEqual(feedback["status"], "PENDING")
        self.assertEqual(
            feedback["suggested_correction"], "The museum is closed on Mondays."
        )
        self.assertEqual(
            [item["id"] for item in self.repository.list_feedback(user_id="user-1")],
            [feedback_id],
        )
        self.assertEqual(self.repository.list_feedback(user_id="another-user"), [])

    def test_initialize_adds_claim_extraction_columns_to_existing_database(self):
        connection = sqlite3.connect(self.repository.db_path)
        try:
            connection.execute(
                """
                CREATE TABLE feedback (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    session_id TEXT,
                    message_id TEXT,
                    original_question TEXT NOT NULL,
                    original_answer TEXT NOT NULL,
                    feedback_text TEXT NOT NULL,
                    suggested_correction TEXT NOT NULL,
                    destination TEXT,
                    explanation TEXT,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

        self.repository.initialize()
        connection = sqlite3.connect(self.repository.db_path)
        try:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(feedback)")
            }
        finally:
            connection.close()

        self.assertTrue(
            {"classification", "claim_json", "policy_decision"}.issubset(columns)
        )

    def test_verified_memory_can_be_expired_and_superseded(self):
        self.repository.initialize()
        first_feedback_id = self.create_feedback()
        first_verification_id = self.create_verification(
            first_feedback_id, "2026-01-01"
        )
        expired_memory_id = self.create_memory(
            first_verification_id, "TIME_SENSITIVE"
        )
        expires_at = datetime.fromisoformat(
            self.repository.list_memories()[0]["expires_at"]
        )
        self.assertGreater(expires_at, datetime.now(timezone.utc) + timedelta(days=29))
        self.assertLess(expires_at, datetime.now(timezone.utc) + timedelta(days=31))

        self.assertEqual(
            self.repository.expire_memories(
                now=(expires_at + timedelta(seconds=1)).isoformat(timespec="seconds")
            ),
            1,
        )
        self.assertEqual(self.repository.list_memories(status="ACTIVE"), [])
        self.assertEqual(
            self.repository.list_memories(status="EXPIRED")[0]["id"],
            expired_memory_id,
        )

        second_verification_id = self.create_verification(
            self.create_feedback(), "2026-01-04"
        )
        old_active_memory_id = self.create_memory(second_verification_id)
        third_verification_id = self.create_verification(
            self.create_feedback(), "2026-01-05"
        )
        latest_memory_id = self.create_memory(third_verification_id)

        self.assertTrue(
            self.repository.supersede_memory(old_active_memory_id, latest_memory_id)
        )
        self.assertEqual(
            self.repository.list_memories(status="ACTIVE")[0]["id"], latest_memory_id
        )
        self.assertEqual(
            self.repository.list_memories(status="ACTIVE", user_id="another-user"), []
        )
        superseded = self.repository.list_memories(status="SUPERSEDED")[0]
        self.assertEqual(superseded["id"], old_active_memory_id)
        self.assertEqual(superseded["superseded_by"], latest_memory_id)
        self.assertIsNotNone(superseded["expires_at"])

    def test_repository_enforces_foreign_keys_and_json(self):
        self.repository.initialize()

        with self.assertRaises(sqlite3.IntegrityError):
            self.repository.create_verification(
                feedback_id="missing-feedback",
                claim={"entity": "City Museum"},
                decision="APPROVED",
                reasoning_summary="",
                evidence=[],
            )

        feedback_id = self.create_feedback()
        with self.assertRaises(ValueError):
            self.repository.create_verification(
                feedback_id=feedback_id,
                claim="{not-json}",
                decision="APPROVED",
                reasoning_summary="",
                evidence=[],
            )

        uncertain_verification_id = self.repository.create_verification(
            feedback_id=feedback_id,
            claim={"entity": "City Museum"},
            decision="UNCERTAIN",
            reasoning_summary="Evidence is inconclusive.",
            evidence=[],
        )
        with self.assertRaisesRegex(
            sqlite3.IntegrityError, "requires approved verification"
        ):
            self.create_memory(uncertain_verification_id)

        approved_verification_id = self.create_verification(
            self.create_feedback(), "2026-01-02"
        )
        with self.assertRaisesRegex(
            ValueError, "HIGHLY_DYNAMIC memories cannot be stored"
        ):
            self.create_memory(approved_verification_id, "HIGHLY_DYNAMIC")

    def test_temporal_types_receive_the_requested_expiry(self):
        self.repository.initialize()
        for temporal_type, expected_days in (
            ("STABLE", 365),
            ("SEMI_STABLE", 180),
            ("TIME_SENSITIVE", 30),
        ):
            with self.subTest(temporal_type=temporal_type):
                verification_id = self.create_verification(
                    self.create_feedback(), "2026-01-01"
                )
                before = datetime.now(timezone.utc)
                memory_id = self.create_memory(verification_id, temporal_type)
                memory = next(
                    item
                    for item in self.repository.list_memories()
                    if item["id"] == memory_id
                )
                expiry = datetime.fromisoformat(memory["expires_at"])
                self.assertGreaterEqual(
                    expiry, before + timedelta(days=expected_days)
                )
                self.assertLessEqual(
                    expiry, datetime.now(timezone.utc) + timedelta(days=expected_days)
                )

    def test_highly_dynamic_memory_is_rejected_before_insertion(self):
        self.repository.initialize()
        verification_id = self.create_verification(
            self.create_feedback(), "2026-01-01"
        )

        with self.assertRaisesRegex(
            ValueError, "HIGHLY_DYNAMIC memories cannot be stored"
        ):
            self.create_memory(verification_id, "HIGHLY_DYNAMIC")
        self.assertEqual(self.repository.list_memories(), [])


if __name__ == "__main__":
    unittest.main()
