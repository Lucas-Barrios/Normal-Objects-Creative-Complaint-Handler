# NormalObjects Creative Complaint Handler

An AI-powered complaint handler for NormalObjects, a fictional company that responds to customer complaints using Stranger Things-inspired tools, character voices, conversation memory, and tool usage tracking.

## How to Run

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install the required packages:

```bash
pip install python-dotenv pandas streamlit langchain langchain-community langchain-core langchain-classic openai
```

3. Add your OpenAI API key to `.env`:

```bash
OPENAI_API_KEY=your_api_key_here
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

5. Or run the command-line performance comparison:

```bash
python3 normalobjects_langchain.py
```

This command makes live OpenAI API calls, so it may take a little time and can consume API credits.

## What the App Shows

The Streamlit app provides a modern chat interface for submitting complaints, sample complaint buttons, conversation memory, live tool usage stats, recent tool sequence chips, and a usage chart showing which creative tools the agent selected.

## File Map

| Path | Purpose |
|---|---|
| `app.py` | Modern Streamlit web app for submitting complaints, viewing responses, checking tool usage stats, and seeing recent tool sequences. |
| `normalobjects_langchain.py` | Core LangChain logic: model setup, tools, agent creation, memory, complaint handling, tracking, demo run, and performance comparison. |
| `agent_evidence.md` | Proof document showing at least three complaints handled, observed tool usage patterns, and creative solution summaries. |
| `lab_summary.md` | Short summary comparing creative agent tool use with a more structured approach. |
| `.env` | Local environment variables, including `OPENAI_API_KEY`. This file should not be committed. |
| `.gitignore` | Keeps secrets and generated files out of Git, including `.env`, `__pycache__/`, `*.pyc`, and `.DS_Store`. |
| `.claude/settings.local.json` | Local IDE/tooling settings. Not part of the main app logic. |
| `__pycache__/` | Generated Python bytecode cache. Safe to ignore. |

## Main Components

- **Creative tools:** `consult_demogorgon`, `check_hawkins_records`, `cast_interdimensional_spell`, `gather_party_wisdom`, `consult_eleven`, `check_government_files`, and `ask_murray_bauman`.
- **Memory:** `ConversationMemory` stores previous complaint/response pairs so the agent can reference earlier complaints.
- **Tracking:** `ToolUsageTracker` records how often each tool is used and the order of tool calls.
- **Performance comparison:** `run_performance_comparison()` compares runs with and without memory using response time and tool call counts.
