from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from agent.agentic_workflow import GraphBuilder
from utils.feedback_repository import FeedbackRepository


class AgentMemoryIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = FeedbackRepository(
            Path(self.temp_dir.name) / "feedback.sqlite3"
        )
        self.repository.initialize()

        feedback_id = self.repository.create_feedback(
            user_id="anonymous",
            original_question="Plan a trip to Mumbai.",
            original_answer="The museum is open daily.",
            feedback_text="The museum is closed on Mondays.",
            suggested_correction="The museum is closed on Mondays.",
            destination="Mumbai",
        )
        verification_id = self.repository.create_verification(
            feedback_id=feedback_id,
            claim={"claim_text": "The museum is closed on Mondays."},
            decision="APPROVED",
            reasoning_summary="Confirmed by an official source.",
            evidence=[{"url": "https://example.org"}],
            source_tier_best="T1",
            verified_at="2026-10-01T00:00:00+00:00",
        )
        self.repository.create_verified_memory(
            verification_id,
            "The museum is closed on Mondays.",
            "City Museum",
            "Mumbai",
            "OPENING_HOURS",
            "SEMI_STABLE",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_graph_adds_relevant_notes_to_agent_system_prompt(self):
        builder = GraphBuilder.__new__(GraphBuilder)
        builder.feedback_repo = self.repository
        builder.tools = []
        builder.llm_with_tools = MagicMock()
        builder.llm_with_tools.invoke.return_value = AIMessage(content="Trip plan")

        output = builder.build_graph().invoke(
            {"messages": ["Plan a trip to Mumbai."]}
        )

        self.assertEqual(output["messages"][-1].content, "Trip plan")
        prompt = builder.llm_with_tools.invoke.call_args.args[0][0].content
        self.assertIn("The museum is closed on Mondays.", prompt)

    def test_unrelated_query_does_not_receive_notes(self):
        builder = GraphBuilder.__new__(GraphBuilder)
        builder.feedback_repo = self.repository
        builder.tools = []
        builder.llm_with_tools = MagicMock()
        builder.llm_with_tools.invoke.return_value = AIMessage(content="Trip plan")

        builder.build_graph().invoke({"messages": ["Plan a trip to Delhi."]})

        prompt = builder.llm_with_tools.invoke.call_args.args[0][0].content
        self.assertNotIn("The museum is closed on Mondays.", prompt)

