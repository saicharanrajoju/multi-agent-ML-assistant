# Multi-Agent ML Assistant

An autonomous, human-in-the-loop machine learning pipeline built with LangGraph and Streamlit. Upload a CSV, describe your goal, and a team of specialized AI agents profiles, cleans, engineers features, trains models, and critiques the results — pausing at each stage for your review and approval.

---

## Key Features

- **Multi-agent pipeline** — Profiler, Cleaner, Feature Engineer, Modeler, and Critic agents work in sequence, each with a focused responsibility.
- **Human-in-the-loop checkpoints** — The pipeline pauses before each major step so you can review agent decisions, read the reasoning, and optionally provide feedback before continuing.
- **Self-correcting critic loop** — The Critic agent scores model results against your goal. If unsatisfied, it routes back to the appropriate agent (up to 3 iterations) with specific, actionable fixes.
- **Secure sandboxed execution** — All generated Python code runs inside isolated E2B cloud sandboxes; nothing executes on your machine.
- **Interactive Streamlit UI** — Live pipeline progress, annotated step explanations, model visualizations, cleaned dataset download, and a built-in chat interface for asking questions about the pipeline.
- **LLM-powered reasoning** — Agents explain their decisions in plain English: why a column was dropped, why a model was chosen, what the critic found.

---

## System Architecture

The pipeline is a directed cyclic graph managed by LangGraph:

```
Profiler → Cleaner → Feature Engineer → Modeler → Critic
                ↑____________________________________________|
                        (iterates up to 3× if needed)
```

| Agent | Responsibility |
|---|---|
| **Profiler** | Analyzes the dataset — detects target column, problem type, class imbalance, null patterns, and recommended models |
| **Cleaner** | Generates and executes cleaning code — handles nulls, outliers, encoding, and type fixes |
| **Feature Engineer** | Creates new features, applies transformations, and runs unit tests on the output |
| **Modeler** | Scouts multiple algorithms on a 10% sample, then trains and evaluates the top candidates |
| **Critic** | Scores the result with a rubric, identifies weak points, and either approves or routes back with specific fixes |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) |
| LLM | [Groq](https://groq.com/) (Llama 3 via LangChain) |
| UI | [Streamlit](https://streamlit.io/) |
| Sandboxed execution | [E2B](https://e2b.dev/) |
| Observability | [LangSmith](https://smith.langchain.com/) (optional) |
| Data / ML | Pandas, Scikit-learn, NumPy, Plotly |

---

## Getting Started

### Prerequisites

- Python 3.10+
- API keys for **Groq** and **E2B** (LangSmith is optional)

### Installation

```bash
git clone https://github.com/saicharanrajoju/multi-agent-ML-assistant.git
cd multi-agent-ML-assistant
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
E2B_API_KEY=your_e2b_api_key
LANGCHAIN_API_KEY=your_langchain_api_key   # optional
LANGCHAIN_TRACING_V2=true                  # optional
LANGCHAIN_PROJECT=ml-agent-assistant       # optional
```

### Running the App

```bash
streamlit run src/ui/app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

1. **Upload or select a dataset** — drop a CSV into `datasets/` or use the sidebar file uploader.
2. **Describe your goal** — e.g. `"Predict customer churn with high F1-score"` or `"Predict house price, minimise RMSE"`.
3. **Click Run Pipeline** — the Profiler runs automatically and the pipeline pauses before Cleaning.
4. **Review and approve** — at each checkpoint, read what the agent plans to do, optionally add feedback, then approve to continue.
5. **Inspect results** — explore tabs for the data profile, annotated cleaning/feature steps, model metrics, confusion matrix, feature importance, critique scorecard, and a chat interface.

---

## Project Structure

```
ml-agent-assistant/
├── main.py                          # CLI entry point (alternative to UI)
├── src/
│   ├── agents/                      # Agent logic (profiler, cleaner, feature_eng, modeler, critic)
│   ├── prompts/                     # System and user prompts for each agent
│   ├── tools/                       # Sandbox executor, file utils, code validator, leakage detector
│   ├── ui/
│   │   ├── app.py                   # Streamlit application
│   │   ├── components/              # Sidebar, approval panel, results panel, pipeline status, chat
│   │   ├── styles.py                # CSS design system
│   │   └── ui_components.py         # Reusable HTML component helpers
│   ├── graph.py                     # LangGraph workflow definition
│   ├── state.py                     # Shared AgentState schema
│   └── llm_helper.py                # LLM call wrapper with fallback
├── datasets/                        # Input CSV files
├── outputs/                         # Generated artifacts (reports, models, code, checkpoints)
└── requirements.txt
```

---

## Dataset Requirements

| Constraint | Limit |
|---|---|
| Minimum rows | 50 |
| Maximum rows | 100,000 |
| Minimum columns | 2 (at least one feature + target) |
| Maximum columns | 500 |
| Required | At least one numeric column |

---

## License

[MIT License](LICENSE)
