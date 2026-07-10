"""Defines REST API endpoints for the AI policy assistant"""

from fastapi import APIRouter
from pydantic import BaseModel
from app.graph.agent import build_agent


router = APIRouter()


# Build the LangGraph agent once
agent = build_agent()


class QuestionRequest(BaseModel):
    """API request model"""

    user_id: str
    session_id: str
    question: str


class QuestionResponse(BaseModel):
    """API response model"""

    answer: str
    audit_log: list[str]


@router.post("/ask", response_model=QuestionResponse)


def ask_policy_question(
    request: QuestionRequest
):
    """Receives a user question,executes the LangGraph agent,returns grounded answer"""


    config = {
        "configurable": {
            # Memory identifier
            # Same session_id = same conversation
            "thread_id": request.session_id
        }
    }


    state = {
        "question": request.question,
        "requested_year": None,
        "evidence": [],
        "answer": "",
        "audit_log": [],
        "history": []
    }


    result = agent.invoke(
        state,
        config=config
    )

    return {
        "answer": result["answer"],
        "audit_log": result["audit_log"]
    }
