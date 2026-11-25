from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from conftest import ScriptedLLM


def test_model_quality(built_data):
    assert built_data.n_customers == 7043
    assert built_data.test_roc_auc > 0.80


def test_sql_guard_blocks_writes(built_data):
    from retainai import db
    for bad in ["DROP TABLE customers", "select 1; delete from customers",
                "UPDATE customers SET churned=0", "PRAGMA table_info(customers)"]:
        with pytest.raises(db.UnsafeQueryError):
            db.run_select(bad)
    out = db.run_select("SELECT COUNT(*) FROM customers WHERE churned = 1")
    assert out["rows"] == [[1869]]


def test_churn_tools(built_data):
    from retainai.tools import get_churn_risk, list_high_risk_customers
    r = get_churn_risk.invoke({"customer_id": "7590-VHVEG"})
    assert 0 <= r["churn_probability"] <= 1
    assert len(r["top_drivers"]) == 5
    assert "error" in get_churn_risk.invoke({"customer_id": "nope"})
    top = list_high_risk_customers.invoke({"limit": 5, "contract": "Month-to-month"})["customers"]
    assert len(top) == 5 and all(c["contract"] == "Month-to-month" for c in top)
    at_risk = [c["revenue_at_risk_per_year"] for c in top]
    assert at_risk == sorted(at_risk, reverse=True)


def test_policy_search(built_data):
    from retainai.tools import search_retention_policies
    hits = search_retention_policies.invoke({"query": "month-to-month contract upgrade discount"})
    assert hits[0]["section"].startswith("R1")


def test_graph_routes_through_agents(built_data):
    from retainai.agents import Route, ask, build_graph
    llm = ScriptedLLM(
        routes=[
            Route(next="churn_scoring", task="Score customer 7590-VHVEG"),
            Route(next="offers", task="Recommend an offer for 7590-VHVEG"),
            Route(next="FINISH"),
        ],
        worker_msgs=[
            AIMessage(content="", tool_calls=[{"name": "get_churn_risk", "args": {"customer_id": "7590-VHVEG"}, "id": "1"}]),
            AIMessage(content="7590-VHVEG is high risk: month-to-month, tenure 1."),
            AIMessage(content="", tool_calls=[{"name": "search_retention_policies", "args": {"query": "new customer"}, "id": "2"}]),
            AIMessage(content="Offer R5 (new customer welcome call)."),
        ],
        final="7590-VHVEG: high risk; offer R5.",
    )
    out = ask("Is 7590-VHVEG at risk and what should we offer?", graph=build_graph(llm))
    assert out["agents_used"] == ["churn_scoring", "offers"]
    assert [c["tool"] for c in out["tool_calls"]] == ["get_churn_risk", "search_retention_policies"]
    assert out["answer"] == "7590-VHVEG: high risk; offer R5."


def test_supervisor_step_limit(built_data):
    from retainai.agents import Route, ask, build_graph
    llm = ScriptedLLM(routes=[Route(next="analytics", task="count")] * 10,
                      worker_msgs=[AIMessage(content="done")] * 10)
    from retainai import config
    out = ask("loop forever", graph=build_graph(llm))
    assert len(out["agents_used"]) == config.MAX_SUPERVISOR_STEPS


def test_api_non_llm_endpoints(built_data):
    from fastapi.testclient import TestClient
    import api
    client = TestClient(api.app)
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/customers/9094-AZPHK/risk")
    assert r.status_code == 200 and r.json()["risk_band"] == "High"
    assert client.get("/customers/NOPE/risk").status_code == 404
    assert len(client.get("/customers/at-risk?limit=3").json()["customers"]) == 3


def test_offer_questions_always_reach_offers_agent(built_data):
    """If the supervisor tries to finish an offer question early, the guardrail routes to offers."""
    from retainai.agents import Route, ask, build_graph, needs_policy
    assert needs_policy("Which offers is 9094-AZPHK eligible for?")
    assert not needs_policy("What is the churn rate by contract?")
    llm = ScriptedLLM(
        routes=[Route(next="analytics", task="share churned"), Route(next="FINISH"), Route(next="FINISH")],
        worker_msgs=[AIMessage(content="53.7% churned."), AIMessage(content="Offer R4 fits.")],
        final="53.7%; offer R4.",
    )
    out = ask("What share churned and what offer would help?", graph=build_graph(llm))
    assert out["agents_used"] == ["analytics", "offers"]
