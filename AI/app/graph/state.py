"""Defines the shared state that travels between LangGraph nodes. Every node can read and modify this state"""

from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):

    # Original user question
    question: str

    # Year explicitly mentioned in current question
    requested_year: str | None

    # Active policy context from conversation
    active_policy_year: str | None

    # Retrieved evidence from FAISS
    evidence: List[Dict[str, Any]]

    # Final generated answer
    answer: str

    # Audit information
    audit_log: List[str]

    # Conversation history
    history: List[Dict[str, str]]
