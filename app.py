"""Streamlit chat UI.  Run:  streamlit run app.py"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from retainai import agents  # noqa: E402

st.set_page_config(page_title="RetainAI", page_icon="📉")
st.title("RetainAI")
st.caption("Ask about churn, at-risk customers and retention offers. "
           "A supervisor agent routes your question to SQL analytics, churn scoring and offer agents.")

EXAMPLES = [
    "What is the churn rate for each contract type?",
    "Who are the 5 active customers with the most revenue at risk?",
    "Find the active customer with the most revenue at risk and recommend an offer for them.",
]


@st.cache_resource
def get_graph():
    return agents.build_graph()


if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.subheader("Try")
    for ex in EXAMPLES:
        if st.button(ex):
            st.session_state.pending = ex

for turn in st.session_state.history:
    st.chat_message("user").write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["out"]["answer"])
        with st.expander(f"Agents: {' → '.join(turn['out']['agents_used']) or 'none'}"):
            for f in turn["out"]["findings"]:
                st.markdown(f"**{f['agent']}** — {f['task']}")
                for c in f["tool_calls"]:
                    st.code(f"{c['tool']}({c['args']})", language="python")
                st.write(f["result"])

question = st.chat_input("Ask a question") or st.session_state.pop("pending", None)
if question:
    with st.spinner("Agents working..."):
        out = agents.ask(question, graph=get_graph())
    st.session_state.history.append({"q": question, "out": out})
    st.rerun()
