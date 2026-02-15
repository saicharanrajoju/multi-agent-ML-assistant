from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # --- Input ---
    dataset_path: str                              # path to the uploaded CSV
    user_goal: str                                 # e.g. "predict customer churn with high F1-score"

    # --- Data Profiling ---
    profile_report: str                            # markdown summary from profiler
    column_info: dict                              # column names, types, stats, nulls
    data_issues: list[str]                         # identified problems (missing vals, imbalance, etc.)
    data_issues: list[str]                         # identified problems (missing vals, imbalance, etc.)
    dataset_summary: dict                          # machine-readable dataset summary from profiler
    target_column: str                             # detected target column name

    # --- Cleaning ---
    cleaning_code: str                             # generated Python cleaning code
    cleaning_summary: dict                         # what the cleaner did, columns after cleaning
    cleaning_approved: bool                        # did human approve?
    cleaning_result: str                           # stdout/stderr from execution

    # --- Feature Engineering ---
    feature_code: str                              # generated feature engineering code
    feature_approved: bool
    feature_result: str

    # --- Modeling ---
    model_code: str                                # generated model training code
    model_approved: bool
    model_result: str                              # metrics, confusion matrix, etc.
    visualization_data: dict                       # model metrics, confusion matrix, feature importance for UI

    # --- Critique ---
    # --- Critique ---
    critique_report: str                           # detailed analysis from critic
    improvement_suggestions: list[str]             # actionable suggestions
    code_fixes: list                               # specific code fixes from critic [{"description": ..., "file": ..., "problem_code": ..., "fixed_code": ..., "reason": ...}]
    iteration_history: list                        # [{"iteration": 1, "critique": "...", "fixes": [...], "scorecard": ...}, ...]
    iteration_count: int                           # how many critique loops so far
    should_iterate: bool                           # does critic want another round?
    scorecard: dict                                # critic's scoring rubric results

    # --- Deployment ---
    deployment_code: str                           # FastAPI + Docker code
    deployment_approved: bool
    api_endpoint: str                              # deployed URL if applicable

    # --- Conversation & Control ---
    messages: Annotated[list, add_messages]         # for LangGraph message passing
    current_agent: str                             # which agent is currently active
    human_feedback: str                            # feedback from human review step
    error: str                                     # error message if something fails
