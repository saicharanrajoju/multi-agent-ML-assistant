import streamlit as st
import os
import sys
import time
import traceback
import contextlib
import io
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.graph import graph
from src.state import AgentState
from src.tools.code_executor import close_sandbox_for_run

# Import newly split UI components
from src.ui.components.sidebar import render_sidebar
from src.ui.components.pipeline_status import render_pipeline_progress, render_pipeline_logs
from src.ui.components.approval_panel import render_approval_panel
from src.ui.components.results_panel import render_results_panel
from src.ui.components.diagnosis_panel import render_diagnosis_panel
from src.ui.styles import apply_custom_styles
from src.ui.ui_components import banner, empty_state

# Page config
st.set_page_config(
    page_title="ML Agent Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_custom_styles()

# Initialize session state (Only mutate in this top-level wrapper)
defaults = {
    "pipeline_state": None,
    "pipeline_config": None,
    "pipeline_running": False,
    "pipeline_complete": False,
    "current_step": 0,
    "logs": [],
    "waiting_for_approval": False,
    "next_node": None,
    "full_debug_log": "--- System Log init ---\n",
    "balloons_shown": False,
    "last_error": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Need access to agent tracking for progress
from src.ui.components.pipeline_status import AGENT_NAME_TO_STEP, AGENT_STEPS

def _update_step_from_agent(agent_name: str):
    if agent_name in AGENT_NAME_TO_STEP:
        st.session_state.current_step = AGENT_NAME_TO_STEP[agent_name] + 1

def _run_graph_stream(initial_or_none, config):
    try:
        stdout_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture):
            for event in graph.stream(initial_or_none, config, stream_mode="values"):
                agent = event.get("current_agent", "")
                if agent:
                    msg = f"✅ {agent.upper()} completed"
                    st.session_state.logs.append(msg)
                    st.session_state.pipeline_state = event
                    _update_step_from_agent(agent)

        captured_log = stdout_capture.getvalue()
        st.session_state.full_debug_log += captured_log

        # Check if waiting at interrupt
        state = graph.get_state(config)
        if state.next:
            st.session_state.waiting_for_approval = True
            st.session_state.next_node = state.next[0]
            msg = f"⏸️ Waiting for approval before: {state.next[0]}"
            st.session_state.logs.append(msg)
            st.session_state.full_debug_log += f"[{time.strftime('%H:%M:%S')}] {msg}\n"
        else:
            st.session_state.pipeline_complete = True
            st.session_state.pipeline_running = False
            st.session_state.logs.append("🎉 Pipeline complete!")
            st.session_state.full_debug_log += f"[{time.strftime('%H:%M:%S')}] Pipeline complete!\n"
            run_id = (st.session_state.pipeline_config or {}).get("configurable", {}).get("thread_id", "default")
            close_sandbox_for_run(run_id)
        return True
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        st.session_state.logs.append(error_msg)
        st.session_state.pipeline_running = False
        st.session_state.last_error = str(e)

        tb = traceback.format_exc()
        st.session_state.full_debug_log += f"\n[{time.strftime('%H:%M:%S')}] EXCEPTION:\n{tb}\n"

        run_id = (st.session_state.pipeline_config or {}).get("configurable", {}).get("thread_id", "default")
        close_sandbox_for_run(run_id)
        return False

def _full_reset():
    run_id = (st.session_state.get("pipeline_config") or {}).get("configurable", {}).get("thread_id", "default")
    close_sandbox_for_run(run_id)
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

def _retry_last_step():
    if st.session_state.pipeline_config:
        st.session_state.pipeline_running = True
        st.session_state.last_error = None
        st.session_state.logs.append("🔁 Retrying from last checkpoint...")
        with st.spinner("Retrying..."):
            _run_graph_stream(None, st.session_state.pipeline_config)
        st.rerun()

def _approve_and_continue(feedback: str):
    st.session_state.waiting_for_approval = False
    st.session_state.logs.append(f"✅ Approved: {st.session_state.next_node}")

    graph.update_state(st.session_state.pipeline_config, {"human_feedback": feedback})
    if feedback:
        st.session_state.logs.append(f"📝 Feedback submitted: {feedback}")

    with st.spinner(f"Running {st.session_state.next_node}..."):
        _run_graph_stream(None, st.session_state.pipeline_config)
    st.rerun()

def _submit_feedback_and_continue(feedback: str):
    st.session_state.waiting_for_approval = False
    graph.update_state(st.session_state.pipeline_config, {"human_feedback": feedback})
    if feedback:
        st.session_state.logs.append(f"📝 Feedback submitted: {feedback}")

    with st.spinner(f"Running {st.session_state.next_node} with feedback..."):
        _run_graph_stream(None, st.session_state.pipeline_config)
    st.rerun()

def _stop_pipeline():
    st.session_state.pipeline_running = False
    st.session_state.waiting_for_approval = False
    st.session_state.logs.append("🛑 Pipeline stopped by user")
    run_id = (st.session_state.get("pipeline_config") or {}).get("configurable", {}).get("thread_id", "default")
    close_sandbox_for_run(run_id)
    st.rerun()


# --- Main Wrapper Execution ---
datasets_dir = os.path.join(PROJECT_ROOT, "datasets")
os.makedirs(datasets_dir, exist_ok=True)

# 1. SIDEBAR
dataset_path = render_sidebar(datasets_dir=datasets_dir, on_reset=_full_reset)

# 2. HEADER
st.markdown(
    '<h1 style="margin-bottom:0.2rem">ML Agent Assistant</h1>'
    '<p style="color:var(--text-muted);font-size:0.9rem;margin-top:0;margin-bottom:1.25rem">'
    'Automated ML pipeline with human-in-the-loop oversight</p>',
    unsafe_allow_html=True,
)

# 3. PIPELINE STATUS
render_pipeline_progress(
    current_step=st.session_state.current_step,
    is_running=st.session_state.pipeline_running,
    is_complete=st.session_state.pipeline_complete
)

# 4. START PIPELINE CONTROLS
goal_suggestions = {
    "WA_Fn-UseC_-Telco-Customer-Churn.csv": "Predict customer churn with high F1-score and low false positive rate",
    "telco_churn.csv": "Predict customer churn with high F1-score and low false positive rate",
    "titanic.csv": "Predict passenger survival with high recall to identify all survivors",
    "adult_income.csv": "Predict whether income exceeds $50K per year with high F1-score, handling class imbalance",
}

# Assume choice is dataset_path
default_goal = goal_suggestions.get(dataset_path, "")

if dataset_path:
    st.markdown(
        banner(f"<strong>{dataset_path}</strong> selected", kind="info"),
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        empty_state("", "No dataset selected", "Choose a CSV from the sidebar or upload your own.", "← Select a dataset to begin"),
        unsafe_allow_html=True,
    )

user_goal = st.text_area(
    "Modeling goal",
    value=default_goal,
    placeholder="e.g., Predict customer churn with high F1-score and low false positive rate",
    height=88,
)

if st.button("Run Pipeline", type="primary", use_container_width=True, disabled=st.session_state.pipeline_running):
    if not user_goal:
        st.error("Please enter a goal.")
    elif not dataset_path:
        st.error("Please select a dataset.")
    else:
        _validation_error = None
        try:
            _preview_path = os.path.join(datasets_dir, dataset_path)
            _df_check = pd.read_csv(_preview_path, nrows=500)
            _total_rows = sum(1 for _ in open(_preview_path)) - 1
            if _total_rows < 50:
                _validation_error = f"Dataset too small ({_total_rows} rows). Need at least 50 rows to train a model."
            elif _total_rows > 100_000:
                _validation_error = f"Dataset too large ({_total_rows:,} rows). Please reduce to under 100,000 rows to avoid sandbox timeouts."
            elif _df_check.shape[1] < 2:
                _validation_error = "Dataset must have at least 2 columns (features + target)."
            elif _df_check.shape[1] > 500:
                _validation_error = f"Dataset has {_df_check.shape[1]} columns — too wide. Reduce to under 500."
            elif len(_df_check.select_dtypes(include='number').columns) == 0:
                _validation_error = "No numeric columns found. The pipeline requires at least one numeric feature."
        except Exception as _ve:
            _validation_error = f"Could not read dataset: {str(_ve)}"

        if _validation_error:
            st.error(_validation_error)
        else:
            st.session_state.pipeline_running = True
        st.session_state.pipeline_complete = False
        st.session_state.current_step = 0
        st.session_state.logs = []
        st.session_state.waiting_for_approval = False
        st.session_state.balloons_shown = False
        st.session_state.last_error = None

        thread_id = f"streamlit-run-{int(time.time())}"
        st.session_state.pipeline_config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            # Input
            "dataset_path": dataset_path,
            "user_goal": user_goal,
            # Profiling
            "profile_report": "",
            "column_info": {},
            "data_issues": [],
            "dataset_summary": {},
            "target_column": "",
            # Reasoning context
            "problem_type": "binary_classification",
            "recommended_metric": "f1",
            "reasoning_context": {},
            # Cleaning
            "cleaning_code": "",
            "cleaning_summary": {},
            "cleaning_approved": False,
            "cleaning_result": "",
            # Feature engineering
            "feature_code": "",
            "feature_approved": False,
            "feature_result": "",
            "unit_test_results": {},
            # Modeling
            "model_code": "",
            "model_approved": False,
            "model_result": "",
            "visualization_data": {},
            "model_unit_test_results": {},
            "scout_ranking": [],
            # Critique
            "critique_report": "",
            "improvement_suggestions": [],
            "code_fixes": [],
            "iteration_history": [],
            "iteration_count": 0,
            "should_iterate": False,
            "scorecard": {},
            # Narrations
            "profiler_narration": "",
            "cleaning_narration": "",
            "feature_narration": "",
            "model_narration": "",
            # Pre-execution review
            "pre_exec_corrections": [],
            # Control
            "messages": [],
            "current_agent": "",
            "human_feedback": "",
            "error": "",
        }
        st.session_state.pipeline_state = initial_state
        st.session_state.logs.append("🚀 Pipeline started!")
        st.session_state.full_debug_log += f"[{time.strftime('%H:%M:%S')}] Pipeline started!\n"

        with st.spinner("Running pipeline... This may take a few minutes."):
            _run_graph_stream(initial_state, st.session_state.pipeline_config)

        st.rerun()

# 5. PIPELINE LOGS & ERRORS
render_pipeline_logs(
    logs=st.session_state.logs,
    last_error=st.session_state.last_error,
    is_running=st.session_state.pipeline_running,
    on_reset=_full_reset,
    on_retry=_retry_last_step
)

# 6. APPROVAL PANEL (if waiting)
if st.session_state.waiting_for_approval:
    render_approval_panel(
        next_node=st.session_state.next_node,
        pipeline_state=st.session_state.pipeline_state,
        on_approve=_approve_and_continue,
        on_submit_feedback=_submit_feedback_and_continue,
        on_stop=_stop_pipeline
    )
elif st.session_state.pipeline_complete:
    st.markdown(
        banner("All agents completed. Review the results in the tabs below.", kind="success", title="Pipeline complete"),
        unsafe_allow_html=True,
    )
    if not st.session_state.balloons_shown:
        st.balloons()
        st.session_state.balloons_shown = True
elif not st.session_state.pipeline_running and not st.session_state.logs:
    st.markdown(
        empty_state("", "Ready to run", "Select a dataset, describe your modeling goal, then click Run Pipeline.", "Results will appear here once the pipeline completes"),
        unsafe_allow_html=True,
    )

# 7. TABS / RESULTS PANEL
render_results_panel(
    state_vals=st.session_state.pipeline_state,
    project_root=PROJECT_ROOT
)

# 8. DIAGNOSIS PANEL (save + LLM assessment)
render_diagnosis_panel(
    state=st.session_state.pipeline_state,
    project_root=PROJECT_ROOT
)

# --- DEBUG CONSOLE ---
with st.expander("System log", expanded=False):
    st.caption("Copy this log to share tracebacks and errors with the developer.")
    if "full_debug_log" in st.session_state:
        st.download_button(
            label="Download full log",
            data=st.session_state.full_debug_log,
            file_name=f"debug_log_{int(time.time())}.txt",
            mime="text/plain",
        )
        st.code(st.session_state.full_debug_log, language="text")

# Footer
st.markdown(
    '<p style="text-align:center;font-size:0.78rem;color:var(--text-muted);margin-top:2rem">'
    'Built with LangGraph · Groq · E2B · LangSmith &nbsp;|&nbsp; DTSC 5082 Group 2'
    '</p>',
    unsafe_allow_html=True,
)
