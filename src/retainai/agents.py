"""Multi-agent graph: a supervisor routes work to three specialist agents, then writes the answer.

    question -> supervisor -> (analytics | churn_scoring | offers) -> supervisor -> ... -> respond

Each specialist runs its own tool-calling loop and reports back a short finding. The supervisor
sees every finding so far and decides the next step, or FINISH.
"""
from __future__ import annotations

import json
import operator
import re
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from . import config
from .tools import ANALYTICS_TOOLS, CHURN_TOOLS, OFFER_TOOLS

WORKERS = {
    "analytics": {
        "tools": ANALYTICS_TOOLS,
        "prompt": (
            "You are the SQL analytics agent for a telecom retention team. Answer the task with "
            "SQLite queries over the customers and churn_scores tables. Check the schema first if "
            "unsure of column names or values. Report exact numbers, briefly."
        ),
    },
    "churn_scoring": {
        "tools": CHURN_TOOLS,
        "prompt": (
            "You are the churn scoring agent. Use the churn model tools to find at-risk customers "
            "and explain why they are at risk, using the SHAP drivers. Report customer ids, "
            "probabilities, revenue at risk and the main drivers, briefly."
        ),
    },
    "offers": {
        "tools": OFFER_TOOLS,
        "prompt": (
            "You are the offer recommendation agent. Recommend retention offers ONLY from the "
            "retention policies (search them). Check each customer's profile against the offer's "
            "eligibility rules, respect combination and approval rules, and cite the offer code "
            "(e.g. R1). Never invent offers or prices."
        ),
    },
}

SUPERVISOR_PROMPT = """You are the supervisor of a customer retention team. You coordinate three agents:
- analytics: SQL questions about customers, churn rates, counts, averages, segments, risk-band counts.
- churn_scoring: churn probability for specific customers, ranking customers by revenue at risk,
  and why a customer is at risk (SHAP drivers).
- offers: anything answered by company policy: which retention offers exist or fit a customer,
  eligibility, combining offers, discount approval, and how/when to contact customers.

Routing rules: questions about policy, offers, approvals or contact guidelines go to offers, even
if they mention risk or a customer segment. Questions about specific customer ids' risk go to
churn_scoring. Always include customer ids in the final answer when the question names one.

Read the user's question and the findings so far. Choose the next agent and give it a clear,
specific task (include customer ids from earlier findings when relevant). Choose FINISH once the
findings are enough to answer. Do not call the same agent for the same task twice."""

RESPOND_PROMPT = """You are a retention analyst. Answer the user's question using ONLY the findings
from your agents below. Be concise and specific: always name the customer ids involved, give
percentages to one decimal place, and cite offer codes (e.g. R1). Only recommend offers that appear
in the findings; never invent offers, prices or rules. If the findings don't answer something, say so."""


_NEEDS_POLICY = re.compile(r"\b(offers?|eligib\w*|recommend\w*|approv\w*|contact\w*|discount\w*)\b", re.IGNORECASE)


def needs_policy(question: str) -> bool:
    """Questions about offers, approvals or contact rules must be grounded in the policy documents."""
    return bool(_NEEDS_POLICY.search(question))


class Route(BaseModel):
    next: Literal["analytics", "churn_scoring", "offers", "FINISH"] = Field(
        description="The agent to act next, or FINISH when the question can be answered.")
    task: str = Field(default="", description="Specific instruction for that agent (empty for FINISH).")


class Finding(TypedDict):
    agent: str
    task: str
    result: str
    tool_calls: list[dict]


class State(TypedDict, total=False):
    question: str
    findings: Annotated[list[Finding], operator.add]
    next: str
    task: str
    steps: int
    answer: str


def _findings_text(findings: list[Finding]) -> str:
    if not findings:
        return "(no findings yet)"
    return "\n\n".join(f"[{f['agent']}] task: {f['task']}\nresult: {f['result']}" for f in findings)


def _text(msg) -> str:
    content = msg.content
    if isinstance(content, list):  # some providers return content blocks
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content).strip()


def make_worker(name: str, llm, max_tool_rounds: int = 5):
    spec = WORKERS[name]
    tools = {t.name: t for t in spec["tools"]}
    llm_tools = llm.bind_tools(spec["tools"])

    def node(state: State) -> dict:
        msgs = [
            SystemMessage(spec["prompt"]),
            HumanMessage(
                f"User question: {state['question']}\n\nFindings from other agents so far:\n"
                f"{_findings_text(state.get('findings', []))}\n\nYour task: {state.get('task') or state['question']}"
            ),
        ]
        calls: list[dict] = []
        ai = None
        for _ in range(max_tool_rounds):
            ai = llm_tools.invoke(msgs)
            msgs.append(ai)
            if not getattr(ai, "tool_calls", None):
                break
            for tc in ai.tool_calls:
                tool = tools.get(tc["name"])
                try:
                    result = tool.invoke(tc["args"]) if tool else {"error": f"unknown tool {tc['name']}"}
                except Exception as e:
                    result = {"error": str(e)}
                calls.append({"tool": tc["name"], "args": tc["args"]})
                msgs.append(ToolMessage(json.dumps(result, default=str)[:2500], tool_call_id=tc["id"]))
        else:
            # Ran out of tool rounds: ask for a final summary without tools.
            msgs.append(HumanMessage("Summarise what you found for the task now, without calling tools."))
            ai = llm.invoke(msgs)
        return {"findings": [{"agent": name, "task": state.get("task", ""), "result": _text(ai), "tool_calls": calls}]}

    return node


def make_supervisor(llm, max_steps: int = config.MAX_SUPERVISOR_STEPS):
    router = llm.with_structured_output(Route)

    def node(state: State) -> dict:
        steps = state.get("steps", 0)
        if steps >= max_steps:
            return {"next": "FINISH", "task": "", "steps": steps}
        prompt = [
            SystemMessage(SUPERVISOR_PROMPT),
            HumanMessage(f"User question: {state['question']}\n\nFindings so far:\n{_findings_text(state.get('findings', []))}"),
        ]
        route = None
        for _ in range(2):  # retry once if the model returns a malformed route
            try:
                route = router.invoke(prompt)
                break
            except Exception as e:  # rate limits are retried by the client; this catches bad output
                last_error = e
        if route is None:
            if not state.get("findings"):
                raise last_error
            return {"next": "FINISH", "task": "", "steps": steps + 1}
        # Guardrail: never finish an offer/policy question without the offers agent,
        # otherwise the responder would have to answer from the LLM's general knowledge.
        used = {f["agent"] for f in state.get("findings", [])}
        if route.next == "FINISH" and needs_policy(state["question"]) and "offers" not in used:
            return {"next": "offers", "steps": steps + 1,
                    "task": "Using the findings so far, answer the policy/offer part of the user's question, "
                            "citing offer codes from the retention policies."}
        return {"next": route.next, "task": route.task, "steps": steps + 1}

    return node


def make_responder(llm):
    def node(state: State) -> dict:
        ai = llm.invoke([
            SystemMessage(RESPOND_PROMPT),
            HumanMessage(f"User question: {state['question']}\n\nFindings:\n{_findings_text(state.get('findings', []))}"),
        ])
        return {"answer": _text(ai)}

    return node


def build_graph(llm=None):
    if llm is None:
        from .llm import get_llm
        llm = get_llm()
    g = StateGraph(State)
    g.add_node("supervisor", make_supervisor(llm))
    for name in WORKERS:
        g.add_node(name, make_worker(name, llm))
        g.add_edge(name, "supervisor")
    g.add_node("respond", make_responder(llm))
    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        lambda s: "respond" if s["next"] == "FINISH" else s["next"],
        {"analytics": "analytics", "churn_scoring": "churn_scoring", "offers": "offers", "respond": "respond"},
    )
    g.add_edge("respond", END)
    return g.compile()


def ask(question: str, graph=None) -> dict:
    graph = graph or build_graph()
    out = graph.invoke({"question": question, "findings": [], "steps": 0})
    return {
        "answer": out.get("answer", ""),
        "agents_used": [f["agent"] for f in out.get("findings", [])],
        "tool_calls": [c for f in out.get("findings", []) for c in f["tool_calls"]],
        "findings": out.get("findings", []),
    }
