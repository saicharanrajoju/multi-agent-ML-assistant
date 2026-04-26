"""
Pipeline Results Saver

Saves the complete pipeline state to a JSON report for external analysis.
"""

import json
import os
import time
import streamlit as st


def save_pipeline_report(state: dict, project_root: str) -> tuple[str, str]:
    """
    Save the full pipeline state to outputs/pipeline_report_<timestamp>.json.
    Returns (file_path, json_string).
    """
    viz = state.get("visualization_data", {}) or {}
    scorecard = state.get("scorecard", {}) or {}
    best = viz.get("best_model", {}) or {}
    comparison = viz.get("model_comparison", {}) or {}
    cv = viz.get("cross_validation", {}) or {}
    tuning = viz.get("tuning", {}) or {}
    threshold = viz.get("threshold", {}) or {}
    unit_tests = state.get("model_unit_test_results", {}) or {}
    fe_tests = state.get("unit_test_results", {}) or {}
    cleaning_summary = state.get("cleaning_summary", {}) or {}
    dataset_summary = state.get("dataset_summary", {}) or {}
    reasoning_context = state.get("reasoning_context", {}) or {}

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),

        # ── Run metadata ──────────────────────────────────────────────
        "run": {
            "dataset": os.path.basename(state.get("dataset_path", "unknown")),
            "user_goal": state.get("user_goal", ""),
            "problem_type": state.get("problem_type", ""),
            "target_column": state.get("target_column", ""),
            "recommended_metric": state.get("recommended_metric", ""),
        },

        # ── Dataset profile ───────────────────────────────────────────
        "data_profile": {
            "shape": dataset_summary.get("shape"),
            "numeric_columns": dataset_summary.get("numeric_columns", []),
            "categorical_columns": dataset_summary.get("categorical_columns", []),
            "missing_values": dataset_summary.get("missing_values", {}),
            "skewed_columns": dataset_summary.get("skewed_columns", []),
            "class_imbalance_ratio": dataset_summary.get("class_imbalance_ratio"),
            "data_issues": state.get("data_issues", []),
        },

        # ── Profiler reasoning ────────────────────────────────────────
        "profiler_reasoning": {
            "imbalance_strategy": reasoning_context.get("imbalance_strategy"),
            "recommended_models": reasoning_context.get("recommended_models", []),
            "encoding_map": reasoning_context.get("encoding_map", {}),
            "null_patterns": reasoning_context.get("null_patterns", {}),
            "feature_strategies": reasoning_context.get("feature_strategies", []),
            "n_rows": reasoning_context.get("n_rows"),
            "n_cols": reasoning_context.get("n_cols"),
        },

        # ── Cleaning ──────────────────────────────────────────────────
        "cleaning": {
            "shape_after": cleaning_summary.get("shape_after"),
            "columns_after": cleaning_summary.get("columns_after", []),
            "dtypes_after": cleaning_summary.get("dtypes_after", {}),
            "no_missing": cleaning_summary.get("no_missing"),
            "numeric_features": cleaning_summary.get("numeric_features", []),
            "unit_tests": state.get("unit_test_results", {}),
            "execution_output": (state.get("cleaning_result") or "")[:2000],
        },

        # ── Feature engineering ───────────────────────────────────────
        "feature_engineering": {
            "unit_tests": fe_tests,
            "execution_output": (state.get("feature_result") or "")[:2000],
        },

        # ── Modeling ──────────────────────────────────────────────────
        "modeling": {
            "best_model": best.get("name"),
            "confusion_matrix": best.get("confusion_matrix"),
            "classification_report": best.get("classification_report"),
            "feature_importance": best.get("feature_importance", {}),
            "model_comparison": {
                "model_names": comparison.get("model_names", []),
                "accuracy": comparison.get("accuracy", []),
                "f1_score": comparison.get("f1_score", []),
                "recall": comparison.get("recall", []),
                "precision": comparison.get("precision", []),
                "auc_roc": comparison.get("auc_roc", []),
                "pr_auc": comparison.get("pr_auc", []),
            },
            "cross_validation": {
                "mean": cv.get("mean"),
                "std": cv.get("std"),
                "scores": cv.get("cv_scores", []),
            },
            "tuning": {
                "metric": tuning.get("metric"),
                "before": tuning.get("before"),
                "after": tuning.get("after"),
                "delta": tuning.get("delta"),
            },
            "threshold": {
                "optimal": threshold.get("optimal"),
                "metric_at_default": threshold.get("metric_at_default"),
                "metric_at_optimal": threshold.get("metric_at_optimal"),
            },
            "artifacts_saved": {
                "model": unit_tests.get("model_artifact_exists"),
                "preprocessor": unit_tests.get("preprocessor_artifact_exists"),
                "metrics_file": unit_tests.get("metrics_file_exists"),
                "metric_reasonable": unit_tests.get("metric_reasonable"),
            },
            "scout_ranking": state.get("scout_ranking", []),
            "execution_output": (state.get("model_result") or "")[:3000],
        },

        # ── Critique ──────────────────────────────────────────────────
        "critique": {
            "scorecard": scorecard,
            "iteration_count": state.get("iteration_count", 0),
            "should_iterate": state.get("should_iterate", False),
            "improvement_suggestions": state.get("improvement_suggestions", []),
            "code_fixes": state.get("code_fixes", []),
            "full_critique_report": state.get("critique_report", ""),
            "iteration_history": state.get("iteration_history", []),
        },

        # ── Generated code ────────────────────────────────────────────
        "generated_code": {
            "cleaning_code": state.get("cleaning_code", ""),
            "feature_engineering_code": state.get("feature_code", ""),
            "model_training_code": state.get("model_code", ""),
        },
    }

    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    filename = f"pipeline_report_{int(time.time())}.json"
    path = os.path.join(output_dir, filename)
    json_str = json.dumps(report, indent=2, default=str)
    with open(path, "w") as f:
        f.write(json_str)
    return path, json_str


def render_diagnosis_panel(state: dict, project_root: str):
    """Render the Save Results section below the results tabs. Logic unchanged — visual only."""
    if not state:
        return

    viz = state.get("visualization_data", {}) or {}
    if not viz.get("best_model"):
        return  # pipeline hasn't reached modeling yet

    st.markdown('<hr style="border:none;border-top:1px solid var(--border);margin:1.5rem 0 1rem">', unsafe_allow_html=True)
    st.markdown(
        '<div style="margin-bottom:0.4rem">'
        '<h3 style="margin:0;font-size:1rem;font-weight:600;color:var(--text-pri)">Export results</h3>'
        '<p style="margin:2px 0 0;font-size:0.82rem;color:var(--text-muted)">Save all metrics, generated code, critique, and artifacts as a JSON report.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.button("Save full results report", type="primary", width="stretch"):
        with st.spinner("Saving..."):
            try:
                path, json_str = save_pipeline_report(state, project_root)
                st.session_state["saved_report_path"] = path
                st.session_state["saved_report_json"] = json_str
            except Exception as e:
                st.markdown(
                    f'<div class="ds-banner ds-banner-error"><span class="ds-banner-icon">✕</span>'
                    f'<div class="ds-banner-body"><span class="ds-banner-title">Save failed</span>{e}</div></div>',
                    unsafe_allow_html=True,
                )

    if st.session_state.get("saved_report_path"):
        path = st.session_state["saved_report_path"]
        json_str = st.session_state["saved_report_json"]
        st.markdown(
            f'<div class="ds-banner ds-banner-success"><span class="ds-banner-icon">✓</span>'
            f'<div class="ds-banner-body">Saved to <code>{os.path.relpath(path)}</code></div></div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            label="Download full report (JSON)",
            data=json_str,
            file_name=os.path.basename(path),
            mime="application/json",
            width="stretch",
        )
