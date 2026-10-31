"""SQLite persistence for feedback, verification decisions, and verified memories."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterator, Mapping
from uuid import uuid4


DEFAULT_DB_PATH = Path("data") / "verified_feedback.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
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
    classification TEXT,
    claim_json TEXT,
    policy_decision TEXT,
    status TEXT NOT NULL DEFAULT 'PENDING'
        CHECK (status IN (
            'PENDING', 'VERIFYING', 'VERIFIED', 'REJECTED',
            'UNCERTAIN', 'NOT_FACTUAL'
        )),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_feedback_user_created
    ON feedback (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_status_created
    ON feedback (status, created_at DESC);

CREATE TABLE IF NOT EXISTS verification (
    id TEXT PRIMARY KEY,
    feedback_id TEXT NOT NULL REFERENCES feedback(id),
    claim_json TEXT NOT NULL,
    decision TEXT NOT NULL
        CHECK (decision IN ('APPROVED', 'REJECTED', 'UNCERTAIN')),
    reasoning_summary TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    source_tier_best TEXT,
    decided_by TEXT NOT NULL
        CHECK (decided_by IN ('AUTO', 'ADMIN')),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    verified_at TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_verification_feedback_created
    ON verification (feedback_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_verification_decision_created
    ON verification (decision, created_at DESC);

CREATE TABLE IF NOT EXISTS verified_memory (
    id TEXT PRIMARY KEY,
    verification_id TEXT NOT NULL UNIQUE REFERENCES verification(id),
    memory_text TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    destination TEXT,
    claim_type TEXT NOT NULL,
    temporal_type TEXT NOT NULL
        CHECK (temporal_type IN (
            'STABLE', 'SEMI_STABLE', 'TIME_SENSITIVE'
        )),
    status TEXT NOT NULL DEFAULT 'ACTIVE'
        CHECK (status IN ('ACTIVE', 'EXPIRED', 'SUPERSEDED', 'DISABLED')),
    superseded_by TEXT REFERENCES verified_memory(id),
    verified_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'USER'
        CHECK (scope IN ('USER', 'GLOBAL')),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_verified_memory_status_expiry
    ON verified_memory (status, expires_at);
CREATE INDEX IF NOT EXISTS idx_verified_memory_destination
    ON verified_memory (destination, entity_name, claim_type);

CREATE TRIGGER IF NOT EXISTS trg_memory_requires_approved_verification
BEFORE INSERT ON verified_memory
FOR EACH ROW
WHEN NOT EXISTS (
    SELECT 1 FROM verification
    WHERE id = NEW.verification_id AND decision = 'APPROVED'
)
BEGIN
    SELECT RAISE(ABORT, 'verified memory requires approved verification');
END;
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_text(value: str | Mapping[str, Any] | list[Any]) -> str:
    if isinstance(value, str):
        json.loads(value)
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class FeedbackRepository:
    """Small SQLite repository; connections are short-lived and thread-safe."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        configured_path = db_path or os.getenv("VERIFICATION_DB_PATH")
        self.db_path = str(configured_path or DEFAULT_DB_PATH)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create the schema and indexes safely when called more than once."""
        with self._connection() as connection:
            connection.executescript(SCHEMA)
            feedback_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(feedback)")
            }
            if "explanation" not in feedback_columns:
                connection.execute("ALTER TABLE feedback ADD COLUMN explanation TEXT")
            if "updated_at" not in feedback_columns:
                connection.execute(
                    "ALTER TABLE feedback ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''"
                )
                connection.execute(
                    "UPDATE feedback SET updated_at = created_at WHERE updated_at = ''"
                )
            for column in ("classification", "claim_json", "policy_decision"):
                if column not in feedback_columns:
                    connection.execute(f"ALTER TABLE feedback ADD COLUMN {column} TEXT")

    def create_feedback(
        self,
        *,
        original_question: str,
        original_answer: str,
        feedback_text: str,
        suggested_correction: str | None,
        user_id: str | None = None,
        session_id: str | None = None,
        message_id: str | None = None,
        destination: str | None = None,
        explanation: str | None = None,
        status: str = "PENDING",
        feedback_id: str | None = None,
    ) -> str:
        feedback_id = feedback_id or str(uuid4())
        now = _utc_now()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO feedback (
                    id, user_id, session_id, message_id, original_question,
                    original_answer, feedback_text, suggested_correction,
                    destination, explanation, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback_id,
                    user_id,
                    session_id,
                    message_id,
                    original_question,
                    original_answer,
                    feedback_text,
                    suggested_correction or "",
                    destination,
                    explanation,
                    status,
                    now,
                    now,
                ),
            )
        return feedback_id

    def get_feedback(self, feedback_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM feedback WHERE id = ?", (feedback_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def list_feedback(
        self,
        *,
        user_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if limit < 1 or offset < 0:
            raise ValueError("limit must be positive and offset cannot be negative")

        conditions: list[str] = []
        values: list[Any] = []
        if user_id is not None:
            conditions.append("user_id = ?")
            values.append(user_id)
        if status is not None:
            conditions.append("status = ?")
            values.append(status)
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM feedback
                {where_clause}
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (*values, limit, offset),
            ).fetchall()
        return [dict(row) for row in rows]

    def set_feedback_status(self, feedback_id: str, status: str) -> bool:
        with self._connection() as connection:
            result = connection.execute(
                "UPDATE feedback SET status = ?, updated_at = ? WHERE id = ?",
                (status, _utc_now(), feedback_id),
            )
        return result.rowcount == 1

    def update_feedback_classification(
        self,
        *,
        feedback_id: str,
        classification: str,
        claim_json: str | None,
        policy_decision: str,
    ) -> bool:
        if claim_json is not None:
            json.loads(claim_json)
        with self._connection() as connection:
            result = connection.execute(
                """
                UPDATE feedback
                SET classification = ?, claim_json = ?, policy_decision = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    classification,
                    claim_json,
                    policy_decision,
                    _utc_now(),
                    feedback_id,
                ),
            )
        return result.rowcount == 1

    def create_verification(
        self,
        *,
        feedback_id: str,
        claim: str | Mapping[str, Any],
        decision: str,
        reasoning_summary: str,
        evidence: str | list[Any],
        source_tier_best: str | None = None,
        decided_by: str = "AUTO",
        attempts: int = 0,
        verified_at: str | None = None,
    ) -> str:
        verification_id = str(uuid4())
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO verification (
                    id, feedback_id, claim_json, decision, reasoning_summary,
                    evidence_json, source_tier_best, decided_by, attempts,
                    verified_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    verification_id,
                    feedback_id,
                    _json_text(claim),
                    decision,
                    reasoning_summary,
                    _json_text(evidence),
                    source_tier_best,
                    decided_by,
                    attempts,
                    verified_at,
                    _utc_now(),
                ),
            )
        return verification_id

    def get_verification(self, verification_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM verification WHERE id = ?", (verification_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def list_verifications(
        self, *, feedback_id: str | None = None
    ) -> list[dict[str, Any]]:
        with self._connection() as connection:
            if feedback_id is None:
                rows = connection.execute(
                    "SELECT * FROM verification ORDER BY created_at DESC, id DESC"
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM verification
                    WHERE feedback_id = ?
                    ORDER BY created_at DESC, id DESC
                    """,
                    (feedback_id,),
                ).fetchall()
        return [dict(row) for row in rows]

    def create_verified_memory(
        self,
        verification_id: str,
        memory_text: str,
        entity_name: str,
        destination: str | None,
        claim_type: str,
        temporal_type: str,
        scope: str = "USER",
    ) -> str:
        """Create a time-limited memory after its verification is approved."""
        expiry_days_by_type = {
            "STABLE": 365,
            "SEMI_STABLE": 180,
            "TIME_SENSITIVE": 30,
        }
        if temporal_type == "HIGHLY_DYNAMIC":
            raise ValueError("HIGHLY_DYNAMIC memories cannot be stored")
        if temporal_type not in expiry_days_by_type:
            raise ValueError(f"Unsupported temporal type: {temporal_type}")

        verified_at = datetime.now(timezone.utc)
        expires_at = verified_at + timedelta(
            days=expiry_days_by_type[temporal_type]
        )
        memory_id = str(uuid4())
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO verified_memory (
                    id, verification_id, memory_text, entity_name, destination,
                    claim_type, temporal_type, status, superseded_by, verified_at,
                    expires_at, scope, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'ACTIVE', NULL, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    verification_id,
                    memory_text,
                    entity_name,
                    destination,
                    claim_type,
                    temporal_type,
                    verified_at.isoformat(),
                    expires_at.isoformat(),
                    scope,
                    _utc_now(),
                ),
            )
        return memory_id

    def list_memories(
        self,
        *,
        status: str | None = None,
        destination: str | None = None,
        scope: str | None = None,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        conditions: list[str] = []
        values: list[Any] = []
        if status is not None:
            conditions.append("status = ?")
            values.append(status)
        if destination is not None:
            conditions.append("destination = ?")
            values.append(destination)
        if scope is not None:
            conditions.append("scope = ?")
            values.append(scope)
        if user_id is not None:
            conditions.append(
                """
                (
                    scope = 'GLOBAL'
                    OR EXISTS (
                        SELECT 1
                        FROM verification AS matched_verification
                        JOIN feedback AS matched_feedback
                            ON matched_feedback.id = matched_verification.feedback_id
                        WHERE matched_verification.id = verified_memory.verification_id
                            AND matched_feedback.user_id = ?
                    )
                )
                """
            )
            values.append(user_id)
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM verified_memory
                {where_clause}
                ORDER BY verified_at DESC, id DESC
                """,
                values,
            ).fetchall()
        return [dict(row) for row in rows]

    def get_active_memories(self, *, user_id: str | None = None) -> list[dict[str, Any]]:
        """Return active, unexpired memories visible to a user."""
        self.expire_memories()
        return self.list_memories(status="ACTIVE", user_id=user_id)

    def expire_memories(self, *, now: str | None = None) -> int:
        current_time = now or _utc_now()
        with self._connection() as connection:
            result = connection.execute(
                """
                UPDATE verified_memory
                SET status = 'EXPIRED'
                WHERE status = 'ACTIVE'
                    AND expires_at IS NOT NULL
                    AND expires_at <= ?
                """,
                (current_time,),
            )
        return result.rowcount

    def set_memory_status(self, memory_id: str, status: str) -> bool:
        with self._connection() as connection:
            result = connection.execute(
                "UPDATE verified_memory SET status = ? WHERE id = ?",
                (status, memory_id),
            )
        return result.rowcount == 1

    def supersede_memory(self, old_memory_id: str, new_memory_id: str) -> bool:
        with self._connection() as connection:
            result = connection.execute(
                """
                UPDATE verified_memory
                SET status = 'SUPERSEDED', superseded_by = ?
                WHERE id = ? AND status = 'ACTIVE' AND id != ?
                """,
                (new_memory_id, old_memory_id, new_memory_id),
            )
        return result.rowcount == 1

    def supersede_matching_memories(
        self,
        *,
        new_memory_id: str,
        entity_name: str,
        destination: str | None,
        claim_type: str,
        scope: str,
        user_id: str | None = None,
    ) -> int:
        """Supersede active memories for the same user/entity/claim context."""
        if scope == "USER" and user_id is None:
            return 0
        conditions = [
            "status = 'ACTIVE'",
            "scope = ?",
            "lower(entity_name) = lower(?)",
            "lower(claim_type) = lower(?)",
            "id != ?",
        ]
        values: list[Any] = [
            scope,
            entity_name,
            claim_type,
            new_memory_id,
        ]
        if destination is None:
            conditions.append("destination IS NULL")
        else:
            conditions.append("lower(destination) = lower(?)")
            values.append(destination)
        if scope == "USER" and user_id is not None:
            conditions.append(
                """
                EXISTS (
                    SELECT 1
                    FROM verification AS matched_verification
                    JOIN feedback AS matched_feedback
                        ON matched_feedback.id = matched_verification.feedback_id
                    WHERE matched_verification.id = verified_memory.verification_id
                        AND matched_feedback.user_id = ?
                )
                """
            )
            values.append(user_id)

        with self._connection() as connection:
            result = connection.execute(
                f"""
                UPDATE verified_memory
                SET status = 'SUPERSEDED', superseded_by = ?
                WHERE {' AND '.join(conditions)}
                """,
                (new_memory_id, *values),
            )
        return result.rowcount
