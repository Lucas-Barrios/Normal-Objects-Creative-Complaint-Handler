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
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    :root {
        --no-ink: #17181c;
        --no-muted: #60646f;
        --no-line: #e4e7ec;
        --no-panel: #ffffff;
        --no-soft: #f6f7f9;
        --no-accent: #7c3aed;
        --no-cyan: #0891b2;
        --no-amber: #b45309;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 6rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #151821 0%, #202433 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] * {
        color: #f5f7fb;
    }

    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] p {
        color: rgba(245, 247, 251, 0.72);
    }

    .app-hero {
        border: 1px solid var(--no-line);
        background:
            radial-gradient(circle at 12% 12%, rgba(124, 58, 237, 0.12), transparent 28%),
            radial-gradient(circle at 88% 8%, rgba(8, 145, 178, 0.12), transparent 26%),
            linear-gradient(135deg, #ffffff 0%, #f7f8fb 100%);
        border-radius: 8px;
        padding: 1.5rem 1.6rem;
        margin-bottom: 1.1rem;
    }

    .eyebrow {
        color: var(--no-accent);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .app-hero h1 {
        color: var(--no-ink);
        font-size: clamp(2rem, 3vw, 3.1rem);
        line-height: 1.05;
        letter-spacing: 0;
        margin: 0 0 0.65rem;
    }

    .hero-copy {
        color: var(--no-muted);
        font-size: 1rem;
        max-width: 780px;
        margin: 0;
    }

    .status-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 1rem 0 1.25rem;
    }

    .status-card {
        border: 1px solid var(--no-line);
        border-radius: 8px;
        background: var(--no-panel);
        padding: 0.9rem 1rem;
    }

    .status-label {
        color: var(--no-muted);
        font-size: 0.76rem;
        font-weight: 650;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
    }

    .status-value {
        color: var(--no-ink);
        font-size: 1.35rem;
        font-weight: 750;
        line-height: 1.2;
        word-break: break-word;
    }

    .tool-chip {
        display: inline-block;
        border: 1px solid rgba(124, 58, 237, 0.28);
        border-radius: 999px;
        background: rgba(124, 58, 237, 0.08);
        color: #4c1d95;
        font-size: 0.76rem;
        font-weight: 650;
        padding: 0.22rem 0.55rem;
        margin: 0.1rem 0.18rem 0.1rem 0;
    }

    .empty-state {
        border: 1px dashed #cfd5df;
        border-radius: 8px;
        background: #fbfcfd;
        padding: 1.2rem;
        color: var(--no-muted);
        margin-top: 0.75rem;
    }

    .empty-state strong {
        color: var(--no-ink);
    }

    .stChatMessage [data-testid="stChatMessageContent"] {
        border: 1px solid var(--no-line);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        font-size: 0.97rem;
        line-height: 1.55;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.35rem;
    }

    div.stButton > button {
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        background: rgba(255, 255, 255, 0.08);
        color: #f5f7fb;
        min-height: 2.6rem;
        text-align: left;
    }

    div.stButton > button:hover {
        border-color: rgba(125, 211, 252, 0.8);
        background: rgba(125, 211, 252, 0.14);
        color: #ffffff;
    }

    .stChatInput {
        border-top: 1px solid var(--no-line);
        background: rgba(255, 255, 255, 0.92);
        backdrop-filter: blur(10px);
    }

    @media (max-width: 800px) {
        .status-grid {
            grid-template-columns: 1fr;
        }

        .app-hero {
            padding: 1.1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


TOOL_LABELS = {
    "consult_demogorgon": "Demogorgon",
    "check_hawkins_records": "Hawkins Records",
    "cast_interdimensional_spell": "Spellcraft",
    "gather_party_wisdom": "Party Wisdom",
    "consult_eleven": "Eleven",
    "check_government_files": "Gov Files",
    "ask_murray_bauman": "Murray",
}


def _tool_chips(tool_names):
    """Render a compact row of tool labels."""
    if not tool_names:
        return "<span class='tool-chip'>No tools yet</span>"

    return "".join(
        f"<span class='tool-chip'>{TOOL_LABELS.get(name, name)}</span>"
        for name in tool_names
    )


def _status_card(label: str, value: str) -> str:
    """Return a small status card."""
    return (
        "<div class='status-card'>"
        f"<div class='status-label'>{label}</div>"
        f"<div class='status-value'>{value}</div>"
        "</div>"
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
    st.markdown("## NormalObjects")
    st.caption("Creative Complaint Handler")
    st.divider()

    st.markdown("### Sample Complaints")
    for i, sample in enumerate(SAMPLE_COMPLAINTS):
        label = sample if len(sample) <= 55 else sample[:52] + "..."
        if st.button(label, key=f"sample_{i}", use_container_width=True):
            st.session_state.pending_complaint = sample

    st.divider()

    st.markdown("### Tool Usage Stats")
    stats = st.session_state.tracker.get_statistics()

    col1, col2 = st.columns(2)
    col1.metric("Total Calls", stats["total_tool_calls"])
    col2.metric("Most Used", TOOL_LABELS.get(stats["most_used"], stats["most_used"] or "—"))

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
        st.markdown("**Recent tool sequence**")
        st.markdown(
            _tool_chips(stats["tool_sequences"][-5:]),
            unsafe_allow_html=True,
        )

    st.divider()
    if st.button("Clear Conversation", use_container_width=True):
        _reset()
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
stats = st.session_state.tracker.get_statistics()
conversation_turns = len(st.session_state.chat_history) // 2
most_used = TOOL_LABELS.get(stats["most_used"], stats["most_used"] or "None")
recent_tools = _tool_chips(stats["tool_sequences"][-4:])

st.markdown(
    """
    <section class="app-hero">
        <div class="eyebrow">Hawkins Division</div>
        <h1>Creative Complaint Handler</h1>
        <p class="hero-copy">
            Turn strange customer complaints into concise, character-driven responses
            using Hawkins records, classified files, psychic readings, rituals, and
            conspiracy-grade reasoning.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='status-grid'>"
    + _status_card("Conversation Turns", str(conversation_turns))
    + _status_card("Tool Calls", str(stats["total_tool_calls"]))
    + _status_card("Most Used Tool", most_used)
    + "</div>",
    unsafe_allow_html=True,
)

st.markdown("**Recent tools**")
st.markdown(recent_tools, unsafe_allow_html=True)


def _submit(complaint: str) -> None:
    """Process a complaint and append it to chat history."""
    st.session_state.chat_history.append({"role": "user", "content": complaint})

    with st.chat_message("user"):
        st.markdown(complaint)

    with st.chat_message("assistant"):
        with st.spinner("Consulting Hawkins sources..."):
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


if st.session_state.chat_history:
    st.divider()
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
else:
    st.markdown(
        """
        <div class="empty-state">
            <strong>No complaints handled yet.</strong><br>
            Pick a sample complaint from the sidebar or submit your own below.
        </div>
        """,
        unsafe_allow_html=True,
    )

# Handle sidebar sample-complaint button clicks
if "pending_complaint" in st.session_state:
    pending = st.session_state.pop("pending_complaint")
    _submit(pending)
    st.rerun()

# Chat input
if user_input := st.chat_input("Submit a complaint to NormalObjects..."):
    _submit(user_input)
    st.rerun()
