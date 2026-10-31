from contextlib import asynccontextmanager
from datetime import datetime
import hmac
import json
import logging
import os
from uuid import uuid4
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from agent.agentic_workflow import GraphBuilder
from utils.feedback_repository import FeedbackRepository
from utils.feedback_service import FeedbackService
from utils.config_loader import load_config
from utils.save_to_document import save_document
from starlette.responses import JSONResponse
from dotenv import load_dotenv
from pydantic import BaseModel
load_dotenv()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    repository = FeedbackRepository()
    repository.initialize()
    verification_config = load_config().get("verification", {})
    app.state.feedback_repository = repository
    app.state.verification_enabled = verification_config.get("enabled", True)
    app.state.feedback_service = FeedbackService(
        repository,
        max_evidence_sources=verification_config.get("max_evidence_sources", 3),
        max_verification_retries=verification_config.get(
            "max_verification_retries", 3
        ),
        evidence_timeout=verification_config.get("evidence_timeout_seconds", 5),
    )
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set specific origins in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    feedback_text: str
    message_id: str | None = None
    suggested_correction: str | None = None
    destination: str | None = None
    explanation: str | None = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    status: str
    message: str


class FeedbackStatusResponse(BaseModel):
    feedback_id: str
    status: str
    feedback_text: str
    suggested_correction: str | None
    destination: str | None
    created_at: str
    updated_at: str | None
    original_question: str | None = None
    original_answer: str | None = None
    explanation: str | None = None
    classification: str | None = None
    claim: dict | None = None
    policy_decision: str | None = None


@app.post("/query")
async def query_travel_agent(query: QueryRequest, http_request: Request):
    try:
        print(query)
        graph = GraphBuilder(
            model_provider="groq",
            feedback_repo=http_request.app.state.feedback_repository,
        )
        react_app=graph()
        #react_app = graph.build_graph()

        png_graph = react_app.get_graph().draw_mermaid_png()
        with open("my_graph.png", "wb") as f:
            f.write(png_graph)

        print(f"Graph saved as 'my_graph.png' in {os.getcwd()}")
        # Assuming request is a pydantic object like: {"question": "your text"}
        messages={"messages": [query.question]}
        output = react_app.invoke(messages)

        # If result is dict with messages:
        if isinstance(output, dict) and "messages" in output:
            final_output = output["messages"][-1].content  # Last AI response
        else:
            final_output = str(output)
        
        return {
            "answer": final_output,
            "message_id": str(uuid4()),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


def process_feedback_background(
    feedback_id: str,
    service: FeedbackService,
    repository: FeedbackRepository,
) -> None:
    """Extract and verify feedback after the submission response is sent."""
    try:
        extraction = service.process_feedback(feedback_id)
        if extraction["proceed_to_verification"]:
            service.verify_feedback(feedback_id)
    except Exception:
        logger.exception("Background feedback processing failed for %s", feedback_id)
        try:
            feedback = repository.get_feedback(feedback_id)
            if feedback is None:
                logger.error(
                    "Could not mark missing feedback %s as UNCERTAIN", feedback_id
                )
            elif feedback["status"] in {"PENDING", "VERIFYING"}:
                if not repository.set_feedback_status(feedback_id, "UNCERTAIN"):
                    logger.error(
                        "Could not mark feedback %s as UNCERTAIN", feedback_id
                    )
        except Exception:
            logger.exception(
                "Failed to persist the background-processing error for %s",
                feedback_id,
            )


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
):
    """Save feedback and schedule claim extraction and evidence verification."""
    if not http_request.app.state.verification_enabled:
        raise HTTPException(status_code=503, detail="Verification is disabled")

    feedback_id = str(uuid4())
    try:
        http_request.app.state.feedback_repository.create_feedback(
            feedback_id=feedback_id,
            user_id="anonymous",
            session_id="default",
            message_id=request.message_id or str(uuid4()),
            original_question=request.question,
            original_answer=request.answer,
            feedback_text=request.feedback_text,
            suggested_correction=request.suggested_correction,
            destination=request.destination,
            explanation=request.explanation,
            status="PENDING",
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to submit feedback: {exc}"
        ) from exc

    background_tasks.add_task(
        process_feedback_background,
        feedback_id,
        http_request.app.state.feedback_service,
        http_request.app.state.feedback_repository,
    )

    return FeedbackResponse(
        feedback_id=feedback_id,
        status="PENDING",
        message=(
            "Correction submitted. It will be checked against reliable sources "
            "before it can influence future answers."
        ),
    )


@app.get("/feedback/{feedback_id}", response_model=FeedbackStatusResponse)
async def get_feedback_status(feedback_id: str, http_request: Request):
    feedback = http_request.app.state.feedback_repository.get_feedback(feedback_id)
    if feedback is None:
        raise HTTPException(status_code=404, detail="Feedback not found")

    return FeedbackStatusResponse(
        feedback_id=feedback["id"],
        status=feedback["status"],
        feedback_text=feedback["feedback_text"],
        suggested_correction=feedback["suggested_correction"] or None,
        destination=feedback["destination"],
        created_at=feedback["created_at"],
        updated_at=feedback["updated_at"],
        original_question=feedback["original_question"],
        original_answer=feedback["original_answer"],
        explanation=feedback.get("explanation"),
        classification=feedback.get("classification"),
        claim=json.loads(feedback["claim_json"]) if feedback.get("claim_json") else None,
        policy_decision=feedback.get("policy_decision"),
    )


@app.get("/feedback")
async def list_feedback(
    http_request: Request,
    user_id: str = "anonymous",
    status: str | None = Query(default=None),
):
    feedback_items = http_request.app.state.feedback_repository.list_feedback(
        user_id=user_id,
        status=status,
    )
    return {
        "feedback": [
            {
                "feedback_id": item["id"],
                "status": item["status"],
                "feedback_text": item["feedback_text"],
                "created_at": item["created_at"],
            }
            for item in feedback_items
        ]
    }


def require_admin_token(admin_token: str | None) -> None:
    configured_token = os.getenv("ADMIN_TOKEN")
    if not configured_token:
        raise HTTPException(
            status_code=503, detail="Admin actions are not configured"
        )
    if admin_token is None or not hmac.compare_digest(admin_token, configured_token):
        raise HTTPException(status_code=401, detail="Invalid admin token")


@app.post("/admin/feedback/{feedback_id}/extract")
async def trigger_extraction(
    feedback_id: str,
    http_request: Request,
    admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
):
    """Manually extract a claim and apply the policy gate; no verification occurs."""
    require_admin_token(admin_token)

    try:
        return http_request.app.state.feedback_service.process_feedback(feedback_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/admin/feedback/{feedback_id}/verify")
async def trigger_verification(
    feedback_id: str,
    http_request: Request,
    admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
):
    """Manually collect evidence and verify an extracted factual correction."""
    require_admin_token(admin_token)
    if not http_request.app.state.verification_enabled:
        raise HTTPException(status_code=503, detail="Verification is disabled")

    try:
        return http_request.app.state.feedback_service.verify_feedback(feedback_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(
            "Feedback verification failed for %s (%s)",
            feedback_id,
            type(exc).__name__,
        )
        raise HTTPException(status_code=500, detail="Feedback verification failed") from exc