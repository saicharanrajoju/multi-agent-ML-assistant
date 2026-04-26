from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # --- Input ---
    dataset_path: str                              # path to the uploaded CSV
    user_goal: str                                 # e.g. "predict customer churn with high F1-score"

    # --- Data Profiling ---
    profile_report: str                            # markdown summary from profiler
    column_info: dict                              # column names, types, stats, nulls
    data_issues: list[str]                         # identified problems (missing vals, imbalance, etc.)
    dataset_summary: dict                          # machine-readable dataset summary from profiler
    target_column: str                             # detected target column name

    # --- Reasoning Context (set by Profiler, read by all downstream agents) ---
    problem_type: str          # "binary_classification", "multiclass_classification", "regression"
    recommended_metric: str    # "f1", "recall", "precision", "rmse", "mae", "r2" — parsed from user goal
    reasoning_context: dict    # structured reasoning: imbalance_strategy, recommended_models, null_patterns, encoding_map, feature_strategies

    # --- Cleaning ---
    cleaning_code: str                             # generated Python cleaning code
    cleaning_summary: dict                         # what the cleaner did, columns after cleaning
    cleaning_approved: bool                        # did human approve?
    cleaning_result: str                           # stdout/stderr from execution

    # --- Feature Engineering ---
    feature_code: str                              # generated feature engineering code
    feature_approved: bool
    feature_result: str
    unit_test_results: dict                        # pass/fail results from post-cleaning and post-feature unit tests

    # --- Modeling ---
    model_code: str                                # generated model training code
    model_approved: bool
    model_result: str                              # metrics, confusion matrix, etc.
    visualization_data: dict                       # model metrics, confusion matrix, feature importance for UI
    model_unit_test_results: dict                  # artifact existence + metric sanity checks
    scout_ranking: list                            # [(model_name, score), ...] from the 10% sample scout

    # --- Critique ---
    critique_report: str                           # detailed analysis from critic
    improvement_suggestions: list[str]             # actionable suggestions
    code_fixes: list                               # specific code fixes from critic [{"description": ..., "file": ..., "problem_code": ..., "fixed_code": ..., "reason": ...}]
    iteration_history: list                        # [{"iteration": 1, "critique": "...", "fixes": [...], "scorecard": ...}, ...]
    iteration_count: int                           # how many critique loops so far
    should_iterate: bool                           # does critic want another round?
    scorecard: dict                                # critic's scoring rubric results

    # --- Teacher Narrations ---
    profiler_narration: str                        # plain-English explanation of profiler decisions
    cleaning_narration: str                        # plain-English explanation of cleaning decisions
    feature_narration: str                         # plain-English explanation of feature engineering decisions
    model_narration: str                           # plain-English explanation of model selection decisions

    # --- Pre-Execution Review ---
    pre_exec_corrections: list                     # issues found and corrected by pre_exec_reviewer per agent

    # --- Conversation & Control ---
    messages: Annotated[list, add_messages]         # for LangGraph message passing
    current_agent: str                             # which agent is currently active
    human_feedback: str                            # feedback from human review step
    error: str                                     # error message if something fails
