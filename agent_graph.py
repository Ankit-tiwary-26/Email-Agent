"""
LangGraph state machine for the email triage agent.

Graph shape:

    fetch --> classify --> route --+--> act (auto)      --> log
                                    |
                                    +--> human_approval --> act (approved) --> log

`route` is a conditional edge: emails with confidence >= threshold and intent
!= "urgent" go straight to `act`; everything else goes to `human_approval` first.
"""

import json
import os

from groq import Groq
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

import tools
from approval_cli import request_approval
from classifier_prompt import build_classifier_messages
from safety_rules import is_protected, is_noreply_or_automated

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class AgentState(TypedDict):
    email: dict
    decision: dict
    human_overrode: bool
    final_action: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def classify_node(state: AgentState) -> AgentState:
    email = state["email"]
    messages = build_classifier_messages(email["subject"], email["sender"], email["body"])

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    try:
        decision = json.loads(raw)
    except json.JSONDecodeError:
        # Fail safe: if the model doesn't return valid JSON, force escalation
        decision = {
            "intent": "urgent",
            "confidence": 0.0,
            "reasoning": "Model returned non-JSON output; escalating for safety.",
            "suggested_action": "escalate",
            "draft_reply_text": "",
            "task_title": "",
        }

    state["decision"] = decision
    return state


def route_after_classify(state: AgentState) -> str:
    decision = state["decision"]
    email = state["email"]

    # Rule-based safety net runs first and can override the LLM entirely.
    # This catches cases where the LLM is confidently wrong (e.g. Google
    # security alerts classified as "fyi" with high confidence).
    protected, reason = is_protected(email["sender"], email["subject"], email["body"])
    if protected:
        decision["reasoning"] = f"[safety rule override] {reason} | original LLM reasoning: {decision['reasoning']}"
        decision["confidence"] = 0.0  # force escalation display to show this wasn't the LLM's call
        state["decision"] = decision
        return "human_approval"

    if decision["intent"] == "urgent":
        return "human_approval"
    if decision["confidence"] < CONFIDENCE_THRESHOLD:
        return "human_approval"
    return "act"


def human_approval_node(state: AgentState) -> AgentState:
    approved_decision = request_approval(state["email"], state["decision"])
    state["decision"] = approved_decision
    state["human_overrode"] = approved_decision.get("human_overrode", False)
    return state


def act_node(state: AgentState) -> AgentState:
    email = state["email"]
    decision = state["decision"]
    action = decision["suggested_action"]

    # Hard block: never draft a reply to a noreply/automated/marketing sender,
    # no matter what the LLM or human approved. If a human explicitly approves
    # draft_reply during the approval step that's fine -- this only catches
    # emails that skipped approval via the "act" fast path.
    if action == "draft_reply":
        blocked, reason = is_noreply_or_automated(email["sender"], email["subject"], email["body"])
        if blocked:
            decision["reasoning"] = (
                f"[noreply guard] {reason}; downgraded from draft_reply to archive | "
                f"original reasoning: {decision.get('reasoning', '')}"
            )
            action = "archive"
            decision["suggested_action"] = "archive"
            state["decision"] = decision

    if action == "draft_reply":
        tools.create_draft_reply(
            to_email=email["sender"],
            subject=email["subject"],
            body_text=decision.get("draft_reply_text", ""),
            thread_id=email["id"],
        )
    elif action == "create_task":
        tools.create_task(email["id"], decision.get("task_title") or email["subject"])
    elif action == "archive":
        tools.archive_email(email["id"])
    elif action == "escalate":
        # Already surfaced via human_approval; just leave it in the inbox unread
        pass

    state["final_action"] = action
    return state


def log_node(state: AgentState) -> AgentState:
    email = state["email"]
    decision = state["decision"]
    tools.log_decision(
        email_id=email["id"],
        sender=email["sender"],
        subject=email["subject"],
        intent=decision["intent"],
        confidence=decision["confidence"],
        reasoning=decision["reasoning"],
        suggested_action=decision["suggested_action"],
        final_action=state.get("final_action", decision["suggested_action"]),
        human_overrode=state.get("human_overrode", False),
    )
    tools.mark_processed(email["id"])
    return state


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("act", act_node)
    graph.add_node("log", log_node)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {"human_approval": "human_approval", "act": "act"},
    )
    graph.add_edge("human_approval", "act")
    graph.add_edge("act", "log")
    graph.add_edge("log", END)

    return graph.compile()


def process_email(email: dict) -> dict:
    """Runs a single email through the compiled graph. Returns final state."""
    app = build_graph()
    initial_state: AgentState = {
        "email": email,
        "decision": {},
        "human_overrode": False,
        "final_action": "",
    }
    return app.invoke(initial_state)
