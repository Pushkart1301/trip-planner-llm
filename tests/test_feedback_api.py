import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, mock_open, patch
from uuid import UUID

from fastapi.testclient import TestClient
from utils.feedback_service import FeedbackService


class FeedbackApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "feedback.sqlite3"
        self.env_patch = patch.dict(
            os.environ, {"VERIFICATION_DB_PATH": str(self.db_path)}
        )
        self.env_patch.start()

        from main import app

        self.app = app
        self.client = TestClient(app)
        self.client.__enter__()

        class IdleService:
            def process_feedback(self, feedback_id):
                return {"proceed_to_verification": False}

        self.app.state.feedback_service = IdleService()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.env_patch.stop()
        self.temp_dir.cleanup()

    def test_feedback_submit_status_and_filtered_list(self):
        payload = {
            "question": "3 days in Mumbai",
            "answer": "Museum X is open every day.",
            "feedback_text": "Museum X is closed on Mondays.",
            "message_id": str(UUID(int=1)),
            "suggested_correction": "Museum X is closed on Mondays.",
            "destination": "Mumbai",
            "explanation": "The museum's official site lists Monday closures.",
        }

        submitted = self.client.post("/feedback", json=payload)
        self.assertEqual(submitted.status_code, 200, submitted.text)
        submit_json = submitted.json()
        UUID(submit_json["feedback_id"])
        self.assertEqual(submit_json["status"], "PENDING")
        self.assertEqual(
            submit_json["message"],
            "Correction submitted. It will be checked against reliable sources "
            "before it can influence future answers.",
        )

        status_response = self.client.get(
            f"/feedback/{submit_json['feedback_id']}"
        )
        self.assertEqual(status_response.status_code, 200, status_response.text)
        status_json = status_response.json()
        self.assertEqual(status_json["feedback_id"], submit_json["feedback_id"])
        self.assertEqual(status_json["status"], "PENDING")
        self.assertEqual(status_json["feedback_text"], payload["feedback_text"])
        self.assertEqual(
            status_json["suggested_correction"], payload["suggested_correction"]
        )
        self.assertEqual(status_json["destination"], "Mumbai")
        self.assertTrue(status_json["created_at"])
        self.assertEqual(status_json["updated_at"], status_json["created_at"])
        stored_feedback = self.app.state.feedback_repository.get_feedback(
            submit_json["feedback_id"]
        )
        self.assertEqual(stored_feedback["message_id"], payload["message_id"])
        self.assertEqual(
            stored_feedback["explanation"],
            "The museum's official site lists Monday closures.",
        )

        listed = self.client.get("/feedback?status=PENDING")
        self.assertEqual(listed.status_code, 200, listed.text)
        list_json = listed.json()
        self.assertEqual(len(list_json["feedback"]), 1)
        self.assertEqual(
            list_json["feedback"][0]["feedback_id"], submit_json["feedback_id"]
        )

        self.assertTrue(self.db_path.read_bytes().startswith(b"SQLite format 3"))

        self.assertEqual(
            self.client.get("/feedback/does-not-exist").status_code, 404
        )

    def test_feedback_can_be_submitted_without_optional_correction(self):
        submitted = self.client.post(
            "/feedback",
            json={
                "question": "Plan a trip to Mumbai",
                "answer": "Try Museum X.",
                "feedback_text": "The opening hours are wrong.",
            },
        )
        self.assertEqual(submitted.status_code, 200, submitted.text)
        status = self.client.get(f"/feedback/{submitted.json()['feedback_id']}")
        self.assertEqual(status.status_code, 200, status.text)
        self.assertIsNone(status.json()["suggested_correction"])

    def test_feedback_is_processed_in_background(self):
        repository = self.app.state.feedback_repository

        class FakeService:
            def process_feedback(self, feedback_id):
                repository.set_feedback_status(feedback_id, "VERIFYING")
                return {"proceed_to_verification": True}

            def verify_feedback(self, feedback_id):
                repository.set_feedback_status(feedback_id, "VERIFIED")

        self.app.state.feedback_service = FakeService()
        submitted = self.client.post(
            "/feedback",
            json={
                "question": "Plan a trip to Goa",
                "answer": "The museum is open daily.",
                "feedback_text": "It is closed on Mondays.",
                "destination": "Goa",
            },
        )
        self.assertEqual(submitted.status_code, 200, submitted.text)
        feedback_id = submitted.json()["feedback_id"]
        self.assertEqual(
            repository.get_feedback(feedback_id)["status"], "VERIFIED"
        )

    def test_background_failure_is_recorded_as_uncertain(self):
        class FailingService:
            def process_feedback(self, feedback_id):
                raise RuntimeError("offline model")

        self.app.state.feedback_service = FailingService()
        submitted = self.client.post(
            "/feedback",
            json={
                "question": "Plan a trip to Goa",
                "answer": "Try the museum.",
                "feedback_text": "The address is wrong.",
            },
        )
        self.assertEqual(submitted.status_code, 200, submitted.text)
        feedback_id = submitted.json()["feedback_id"]
        self.assertEqual(
            self.app.state.feedback_repository.get_feedback(feedback_id)["status"],
            "UNCERTAIN",
        )

    def test_query_returns_message_id_and_timestamp(self):
        graph = MagicMock()
        graph.return_value.get_graph.return_value.draw_mermaid_png.return_value = b"png"
        graph.return_value.invoke.return_value = {
            "messages": [MagicMock(content="Mock trip plan")]
        }
        with patch("main.GraphBuilder", return_value=graph), patch(
            "builtins.open", mock_open()
        ):
            response = self.client.post(
                "/query", json={"question": "Plan a trip to Mumbai"}
            )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["answer"], "Mock trip plan")
        UUID(body["message_id"])
        self.assertTrue(body["timestamp"])

    def test_manual_extraction_endpoint_requires_admin_token(self):
        submitted = self.client.post(
            "/feedback",
            json={
                "question": "Museum X hours?",
                "answer": "It is open daily.",
                "feedback_text": "Museum X is closed on Mondays.",
                "suggested_correction": "Museum X is closed on Mondays.",
                "destination": "Mumbai",
            },
        )
        feedback_id = submitted.json()["feedback_id"]
        with patch.dict(os.environ, {"ADMIN_TOKEN": ""}):
            self.assertEqual(
                self.client.post(
                    f"/admin/feedback/{feedback_id}/extract",
                    headers={"X-Admin-Token": "wrong"},
                ).status_code,
                503,
            )

        class FakeLlm:
            def invoke(self, messages):
                return MagicMock(
                    content=json.dumps(
                        {
                            "classification": "FACTUAL_CORRECTION",
                            "claim": {
                                "entity_name": "Museum X",
                                "location": "Mumbai",
                                "claim_type": "OPENING_HOURS",
                                "attribute": "opening hours",
                                "claimed_value": "closed on Mondays",
                                "temporal_type": "SEMI_STABLE",
                                "claim_text": "Museum X is closed on Mondays.",
                            },
                            "reasoning": "A specific factual correction.",
                        }
                    )
                )

        self.app.state.feedback_service = FeedbackService(
            self.app.state.feedback_repository, FakeLlm()
        )
        with patch.dict(os.environ, {"ADMIN_TOKEN": "test-admin-token"}):
            unauthorized = self.client.post(
                f"/admin/feedback/{feedback_id}/extract",
                headers={"X-Admin-Token": "wrong"},
            )
            self.assertEqual(unauthorized.status_code, 401)

            extracted = self.client.post(
                f"/admin/feedback/{feedback_id}/extract",
                headers={"X-Admin-Token": "test-admin-token"},
            )
            self.assertEqual(extracted.status_code, 200, extracted.text)
            self.assertEqual(extracted.json()["policy_decision"], "VERIFY")
            self.assertTrue(extracted.json()["proceed_to_verification"])

        updated = self.client.get(f"/feedback/{feedback_id}")
        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertEqual(updated.json()["classification"], "FACTUAL_CORRECTION")
        self.assertEqual(updated.json()["policy_decision"], "VERIFY")
        self.assertEqual(updated.json()["status"], "VERIFYING")
        self.assertEqual(
            updated.json()["claim"]["entity_name"], "Museum X"
        )

    def test_manual_verification_endpoint_requires_admin_token(self):
        feedback_id = self.app.state.feedback_repository.create_feedback(
            original_question="Museum X hours?",
            original_answer="It is open daily.",
            feedback_text="Museum X is closed on Mondays.",
            suggested_correction="Museum X is closed on Mondays.",
            status="VERIFYING",
            feedback_id=str(UUID(int=2)),
        )
        self.app.state.feedback_repository.update_feedback_classification(
            feedback_id=feedback_id,
            classification="FACTUAL_CORRECTION",
            claim_json=json.dumps({"claim_text": "Museum X is closed on Mondays."}),
            policy_decision="VERIFY",
        )
        service = MagicMock()
        service.verify_feedback.return_value = {
            "feedback_id": feedback_id,
            "decision": "UNCERTAIN",
        }
        self.app.state.feedback_service = service
        self.app.state.verification_enabled = True

        with patch.dict(os.environ, {"ADMIN_TOKEN": "test-admin-token"}):
            unauthorized = self.client.post(
                f"/admin/feedback/{feedback_id}/verify",
                headers={"X-Admin-Token": "wrong"},
            )
            self.assertEqual(unauthorized.status_code, 401)
            service.verify_feedback.assert_not_called()

            verified = self.client.post(
                f"/admin/feedback/{feedback_id}/verify",
                headers={"X-Admin-Token": "test-admin-token"},
            )

        self.assertEqual(verified.status_code, 200, verified.text)
        self.assertEqual(verified.json()["decision"], "UNCERTAIN")
        service.verify_feedback.assert_called_once_with(feedback_id)


if __name__ == "__main__":
    unittest.main()
