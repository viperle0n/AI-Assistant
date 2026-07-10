"""This script defines the LangGraph workflow for the insurance policy assistant"""


from langgraph.graph import StateGraph, END
from app.graph.state import AgentState
from app.retrieval.retriever import retrieve_policy_evidence
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver


# For factual correctness, temperature=0 is the right choice
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)


memory = MemorySaver()


def build_prompt(
    question: str,
    evidence: list,
    history: list
) -> str:
    """Builds a grounded prompt for the LLM"""

    # Build previous conversation context
    conversation = ""

    for message in history:

        role = message["role"].capitalize()

        conversation += (
            f"{role}: {message['content']}\n"
        )


    # Build policy evidence context
    context = ""

    for item in evidence:

        context += (
            f"\n"
            f"Source: {item['metadata'].get('source')}\n"
            f"Year: {item['metadata'].get('year')}\n"
            f"Page: {item['metadata'].get('page')}\n\n"
            f"{item['content']}\n"
            f"\n-----------------------\n"
        )


    return f"""
You are an internal AI assistant for an insurance company.

Your primary objective is factual correctness.

Use ONLY the retrieved policy evidence.

Rules:

- Never invent facts.
- Never use outside knowledge.
- Never answer from memory alone.
- Always base answers on retrieved evidence.
- If evidence is insufficient, explicitly say so.
- Cite policy year and page whenever possible.

Previous Conversation:

{conversation}

Current Question:

{question}

Retrieved Policy Evidence:

{context}
"""
        

def generate_answer(state: AgentState):
    """Generates the final answer using the LLM"""

    prompt = build_prompt(
        state["question"],
        state["evidence"],
        state.get("history", [])
    )

    response = llm.invoke(prompt)

    state["answer"] = response.content

    state["audit_log"].append(
        "Generated grounded answer."
    )

    return state


def analyze_question(state: AgentState):
    """Determines the policy year"""

    import re


    question = state["question"]


    # Check if user mentions a year
    match = re.search(
        r"(20\d{2})",
        question
    )


    if match:
        year = match.group(1)
        state["active_policy_year"] = year

    else:
        # Try to reuse previous policy context
        year = state.get(
             "active_policy_year"
        )


    state["requested_year"] = year

    state["audit_log"].append(
        f"Active policy year: {year}"
    )


    return state


def retrieve_policy(state: AgentState):
    """Retrieves supporting policy evidence"""

    evidence = retrieve_policy_evidence(
        state["question"],
        requested_year=state.get("requested_year")
    )


    state["evidence"] = evidence


    state["audit_log"].append(
        f"Retrieved {len(evidence)} evidence chunks"
    )


    return state


def validate_evidence(state: AgentState):
    """Checks whether enough evidence was retrieved"""


    evidence = state["evidence"]


    if len(evidence) == 0:

        state["audit_log"].append(
            "No evidence found"
        )

    else:

        state["audit_log"].append(
            "Evidence validation passed"
        )


    return state



def insufficient_evidence(state: AgentState):
    """Returns a safe response when the retrieved evidence is insufficient"""

    state["answer"] = (
        "I could not find sufficient policy evidence to answer this question reliably.\n\n"
        "Please specify the policy year or consult the compliance department."
    )

    state["audit_log"].append(
        "Returned insufficient evidence response."
    )

    return state


def route_after_validation(state: AgentState):
    """Determines the next node after validation"""

    evidence = state["evidence"]

    if len(evidence) == 0:
        return "insufficient"

    return "generate"


def update_memory(state: AgentState):
    """
    Updates the conversation history stored in the agent state. Only the most recent 10 messages are kept
    (5 user-assistant exchanges) to prevent the prompt from growing indefinitely.
    """

    history = state.get("history", [])

    history.append(
        {
            "role": "user",
            "content": state["question"]
        }
    )

    history.append(
        {
            "role": "assistant",
            "content": state["answer"]
        }
    )

    # Keep only the last 10 messages (5 user-assistant pairs)
    state["history"] = history[-10:]

    state["audit_log"].append(
        "Conversation history updated."
    )

    return state






def build_agent():
    """Building the AI Agent"""
    
    graph = StateGraph(
        AgentState
    )


    graph.add_node(
        "analyze",
        analyze_question
    )


    graph.add_node(
        "generate",
        generate_answer
    )


    graph.add_node(
        "retrieve",
        retrieve_policy
    )


    graph.add_node(
        "validate",
        validate_evidence
    )

    graph.add_node(
        "insufficient",
        insufficient_evidence
    )

    graph.add_node(
        "memory",
        update_memory
    )


    graph.set_entry_point(
        "analyze"
    )


    graph.add_edge(
        "analyze",
        "retrieve"
    )

    graph.add_edge(
        "retrieve",
        "validate"
    )
    
    graph.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "generate": "generate",
            "insufficient": "insufficient"
        }
    )


    graph.add_edge(
        "generate",
        "memory"
    )

    graph.add_edge(
        "insufficient",
        "memory"
    )   


    graph.add_edge(
        "memory",
    END
    )


    return graph.compile(checkpointer=memory)
