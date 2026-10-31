"""Orchestrate claim extraction and policy gating for submitted feedback."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from utils.claim_extractor import ClaimExtractor, PolicyGate
from utils.evidence_collector import EvidenceCollector, SourceTier
from utils.verifier import Verifier


logger = logging.getLogger(__name__)


class FeedbackService:
    """Extract and verify claims; verification runs only when explicitly triggered."""

    def __init__(
        self,
        repo: Any,
        llm_client: Any | None = None,
        *,
        evidence_collector_factory: Any = EvidenceCollector,
        verifier_factory: Any = Verifier,
        max_evidence_sources: int = 3,
        max_verification_retries: int = 3,
        evidence_timeout: float = 5,
    ) -> None:
        self.repo = repo
        self._llm_client = llm_client
        self.evidence_collector_factory = evidence_collector_factory
        self.verifier_factory = verifier_factory
        self.max_evidence_sources = max_evidence_sources
        self.max_verification_retries = max_verification_retries
        self.evidence_timeout = evidence_timeout

    def _get_llm_client(self) -> Any:
        if self._llm_client is None:
            from utils.model_loader import ModelLoader

            self._llm_client = ModelLoader(model_provider="groq").load_llm()
        return self._llm_client

    def process_feedback(self, feedback_id: str) -> dict[str, Any]:
        feedback = self.repo.get_feedback(feedback_id)
        if feedback is None:
            raise LookupError("Feedback not found")
        if feedback["status"] != "PENDING":
            raise ValueError("Only PENDING feedback can be processed")

        feedback_text = feedback["feedback_text"]
        suggested_correction = feedback.get("suggested_correction")
        explanation = feedback.get("explanation")
        submitted_text = "\n".join(
            part
            for part in (
                feedback_text,
                f"Suggested correction: {suggested_correction}"
                if suggested_correction
                else "",
                f"Explanation: {explanation}" if explanation else "",
            )
            if part
        )
        extraction = ClaimExtractor(self._get_llm_client()).extract_claim(
            feedback_text=submitted_text,
            original_answer=feedback["original_answer"],
            destination=feedback["destination"],
        )
        classification = extraction["classification"]
        claim = extraction["claim"]
        policy = PolicyGate.apply_policies(claim, classification)
        policy_decision = policy["decision"]

        self.repo.update_feedback_classification(
            feedback_id=feedback_id,
            classification=classification,
            claim_json=json.dumps(claim, ensure_ascii=False) if claim else None,
            policy_decision=policy_decision,
        )
        new_status = "VERIFYING" if policy["proceed_to_verification"] else policy_decision
        self.repo.set_feedback_status(feedback_id, new_status)
        logger.info(
            "Feedback claim extraction completed",
            extra={
                "feedback_id": feedback_id,
                "classification": classification,
                "policy_decision": policy_decision,
            },
        )

        return {
            "feedback_id": feedback_id,
            "classification": classification,
            "claim": claim,
            "policy_decision": policy_decision,
            "proceed_to_verification": policy["proceed_to_verification"],
            "reasoning": policy["reason"],
        }

    def verify_feedback(self, feedback_id: str) -> dict[str, Any]:
        """Collect evidence, assess it source-by-source, and apply code rules."""
        feedback = self.repo.get_feedback(feedback_id)
        if feedback is None:
            raise LookupError("Feedback not found")
        if feedback["status"] != "VERIFYING" or feedback.get("policy_decision") != "VERIFY":
            raise ValueError("Only feedback approved by the policy gate can be verified")

        claim_json = feedback.get("claim_json")
        try:
            claim = json.loads(claim_json) if claim_json else None
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Stored claim is invalid") from exc
        if not isinstance(claim, dict):
            raise ValueError("Stored claim is missing or invalid")

        policy = PolicyGate.apply_policies(claim, feedback.get("classification"))
        if not policy["proceed_to_verification"]:
            self.repo.set_feedback_status(feedback_id, policy["decision"])
            return {
                "feedback_id": feedback_id,
                "decision": policy["decision"],
                "reasoning": policy["reason"],
                "evidence_count": 0,
            }

        collector = self.evidence_collector_factory(
            max_sources=self.max_evidence_sources,
            timeout=self.evidence_timeout,
            retries=self.max_verification_retries,
        )
        evidence = collector.collect_for_claim(claim)
        llm_client = None
        llm_unavailable = False
        if evidence:
            try:
                llm_client = self._get_llm_client()
            except Exception as exc:
                llm_unavailable = True
                logger.warning(
                    "Verification model unavailable for %s (%s)",
                    feedback_id,
                    type(exc).__name__,
                )
        verifier = self.verifier_factory(llm_client)
        if llm_unavailable:
            verifier.assessment_error = "Evidence assessment is unavailable."
            assessed_evidence = evidence
        else:
            assessed_evidence = verifier.summarize_evidence(evidence, claim)
        result = verifier.decide(
            claim,
            assessed_evidence,
            collection_failures=collector.failures,
        )

        verified_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        source_tier_best = None
        if assessed_evidence:
            source_tier_best = min(
                evidence_item.source_tier for evidence_item in assessed_evidence
            ).name
        verification_id = self.repo.create_verification(
            feedback_id=feedback_id,
            claim=claim,
            decision=result["decision"],
            reasoning_summary=result["reasoning"],
            evidence=result["evidence"],
            source_tier_best=source_tier_best,
            decided_by="AUTO",
            attempts=collector.request_attempts,
            verified_at=verified_at,
        )

        memory_id = None
        if result["decision"] == "APPROVED":
            memory_id = self.repo.create_verified_memory(
                verification_id=verification_id,
                memory_text=claim["claim_text"],
                entity_name=claim["entity_name"],
                destination=claim.get("location") or feedback.get("destination"),
                claim_type=claim["claim_type"],
                temporal_type=claim["temporal_type"],
                scope="USER",
            )
            self.repo.supersede_matching_memories(
                new_memory_id=memory_id,
                entity_name=claim["entity_name"],
                destination=claim.get("location") or feedback.get("destination"),
                claim_type=claim["claim_type"],
                scope="USER",
                user_id=feedback.get("user_id"),
            )

        feedback_status = (
            "VERIFIED" if result["decision"] == "APPROVED" else result["decision"]
        )
        self.repo.set_feedback_status(feedback_id, feedback_status)
        logger.info(
            "Feedback verification completed",
            extra={
                "feedback_id": feedback_id,
                "verification_id": verification_id,
                "decision": result["decision"],
                "evidence_count": len(assessed_evidence),
            },
        )
        return {
            "feedback_id": feedback_id,
            "verification_id": verification_id,
            "decision": result["decision"],
            "status": feedback_status,
            "reasoning": result["reasoning"],
            "evidence_count": len(assessed_evidence),
            "evidence": result["evidence"],
            "source_notices": getattr(collector, "usage_notices", []),
            "memory_id": memory_id,
        }
