import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI

from normalobjects_langchain import (
    MODEL_NAME,
    SAMPLE_COMPLAINTS,
    ConversationMemory,
    ToolUsageTracker,
    build_tools,
    create_complaint_agent,
    create_tracked_tools,
    handle_complaint,
)

load_dotenv()

st.set_page_config(
    page_title="NormalObjects — Creative Complaint Handler",
    page_icon="🔦",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stChatMessage [data-testid="stChatMessageContent"] { font-size: 0.95rem; }
    .tool-badge { background:#1e1e2e; border:1px solid #444; border-radius:6px;
                  padding:2px 8px; font-size:0.75rem; color:#cba6f7; margin:2px; display:inline-block; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session-state helpers ─────────────────────────────────────────────────────

def _build_agent():
    """Instantiate LLM, tools, tracker, agent, and memory; store in session_state."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OPENAI_API_KEY is not set. Add it to your .env file and restart.")
        st.stop()

    llm = ChatOpenAI(model_name=MODEL_NAME)
    base_tools = build_tools(llm)
    tracker = ToolUsageTracker(t.name for t in base_tools)
    tracked_tools = create_tracked_tools(base_tools, tracker)
    agent_executor = create_complaint_agent(llm=llm, tools_to_use=tracked_tools)

    st.session_state.tracker = tracker
    st.session_state.agent_executor = agent_executor
    st.session_state.memory = ConversationMemory()
    st.session_state.chat_history = []  # list of {"role": str, "content": str}
    st.session_state.initialized = True


def _reset():
    for key in list(st.session_state.keys()):
        del st.session_state[key]


if "initialized" not in st.session_state:
    _build_agent()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔦 NormalObjects")
    st.caption("Creative Complaint Handler — Hawkins Division")
    st.divider()

    st.markdown("### Try a Sample Complaint")
    for i, sample in enumerate(SAMPLE_COMPLAINTS):
        label = sample if len(sample) <= 55 else sample[:52] + "..."
        if st.button(label, key=f"sample_{i}", use_container_width=True):
            st.session_state.pending_complaint = sample

    st.divider()

    st.markdown("### Tool Usage Stats")
    stats = st.session_state.tracker.get_statistics()

    col1, col2 = st.columns(2)
    col1.metric("Total Calls", stats["total_tool_calls"])
    col2.metric("Most Used", stats["most_used"] or "—")

    counts = stats["tool_counts"]
    if any(v > 0 for v in counts.values()):
        chart_df = (
            pd.DataFrame.from_dict(counts, orient="index", columns=["calls"])
            .sort_values("calls", ascending=False)
        )
        st.bar_chart(chart_df)
    else:
        st.caption("No tools used yet.")

    if stats["tool_sequences"]:
        st.markdown("**Recent tool sequence:**")
        seq_str = " → ".join(stats["tool_sequences"][-5:])
        st.caption(seq_str)

    st.divider()
    if st.button("🗑 Clear Conversation", use_container_width=True):
        _reset()
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
st.markdown("# 🌀 NormalObjects Creative Complaint Handler")
st.caption(
    "Powered by Hawkins Lab R&D • Interdimensional Customer Service Division • "
    "All complaints reviewed by licensed Demogorgons."
)
st.divider()


def _submit(complaint: str) -> None:
    """Process a complaint and append it to chat history."""
    st.session_state.chat_history.append({"role": "user", "content": complaint})

    with st.chat_message("user"):
        st.markdown(complaint)

    with st.chat_message("assistant"):
        with st.spinner("Consulting the Upside Down… 🕯️"):
            try:
                response = handle_complaint(
                    complaint,
                    st.session_state.agent_executor,
                    st.session_state.memory,
                )
            except Exception as exc:
                response = f"⚠️ Something went wrong in the Upside Down: {exc}"

        st.markdown(response)

    st.session_state.chat_history.append({"role": "assistant", "content": response})


# Render existing chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle sidebar sample-complaint button clicks
if "pending_complaint" in st.session_state:
    pending = st.session_state.pop("pending_complaint")
    _submit(pending)
    st.rerun()

# Chat input
if user_input := st.chat_input("Submit your complaint to NormalObjects…"):
    _submit(user_input)
    st.rerun()
