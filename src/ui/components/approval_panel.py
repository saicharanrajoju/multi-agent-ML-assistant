import streamlit as st
from typing import Callable
from src.ui.ui_components import banner, kpi_row, section_header, divider, pill


def render_approval_panel(
    next_node: str,
    pipeline_state: dict,
    on_approve: Callable[[str], None],
    on_submit_feedback: Callable[[str], None],
    on_stop: Callable[[], None],
):
    """
    Approval checkpoint panel.
    Signatures, callbacks, and all data reads unchanged — visual layer only.
    """
    state_vals = pipeline_state or {}

    st.markdown(divider(), unsafe_allow_html=True)

    node_labels = {
        "cleaner":          "Ready to clean data",
        "feature_engineer": "Ready to engineer features",
        "modeler":          "Ready to train models",
        "deployer":         "Ready to deploy",
    }
    node_label = node_labels.get(next_node, next_node.replace("_", " ").title())

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1rem">'
        f'<div style="width:8px;height:8px;border-radius:50%;background:var(--warn);flex-shrink:0"></div>'
        f'<h2 style="margin:0;font-size:1.05rem">{node_label}</h2>'
        f'{pill("Review required", "amber")}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Cleaner checkpoint ────────────────────────────────────────────────────
    if next_node == "cleaner":
        rc = state_vals.get("reasoning_context", {})
        problem_type = state_vals.get("problem_type", "unknown")
        recommended_metric = state_vals.get("recommended_metric", "unknown")

        st.markdown(section_header("What the profiler decided"), unsafe_allow_html=True)

        imbalance_val = rc.get("imbalance_strategy", "none").replace("_", " ").title()
        st.markdown(
            kpi_row([
                {"label": "Problem type",    "value": problem_type.replace("_", " ").title()},
                {"label": "Optimizing for",  "value": recommended_metric.upper(),   "pill": recommended_metric.upper(), "pill_color": "blue"},
                {"label": "Imbalance",        "value": imbalance_val,               "pill": imbalance_val,             "pill_color": "gray"},
            ]),
            unsafe_allow_html=True,
        )

        rec_models = rc.get("recommended_models", [])
        if rec_models:
            st.markdown(
                banner(f"Models queued: <strong>{', '.join(rec_models)}</strong>", kind="info"),
                unsafe_allow_html=True,
            )

        enc_map = rc.get("encoding_map", {})
        if enc_map:
            with st.expander(f"Encoding strategy — {len(enc_map)} columns"):
                enc_df_data = [{"Column": col, "Strategy": strategy} for col, strategy in enc_map.items()]
                st.dataframe(enc_df_data, use_container_width=True, hide_index=True)

        feat_strats = rc.get("feature_strategies", [])
        if feat_strats:
            with st.expander(f"Feature engineering plan — {len(feat_strats)} strategies"):
                for s in feat_strats:
                    st.markdown(f'<div style="font-size:0.85rem;padding:0.2rem 0;color:var(--text-sec)">› {s}</div>', unsafe_allow_html=True)

        with st.expander("Data profile summary"):
            st.markdown(
                f'<div style="font-size:0.82rem;color:var(--text-sec);white-space:pre-wrap">'
                f'{state_vals.get("profile_report", "No report yet")[:2000]}'
                f'</div>',
                unsafe_allow_html=True,
            )

        issues = state_vals.get("data_issues", [])
        if issues:
            st.markdown(
                banner(
                    "".join(f'<div style="margin-top:3px">· {i}</div>' for i in issues),
                    kind="warning",
                    title=f"{len(issues)} issue{'s' if len(issues) != 1 else ''} found",
                ),
                unsafe_allow_html=True,
            )

    # ── Feature engineer checkpoint ───────────────────────────────────────────
    elif next_node == "feature_engineer":
        iteration_count = state_vals.get("iteration_count", 0)

        if iteration_count > 0:
            iteration_history = state_vals.get("iteration_history", [])
            last_entry = iteration_history[-1] if iteration_history else {}
            last_scorecard = last_entry.get("scorecard", {})
            suggestions = last_entry.get("suggestions", [])
            severity = last_entry.get("severity", "moderate")
            n_fixes = last_entry.get("n_fixes", 0)

            sev_map = {"critical": "error", "high": "error", "moderate": "warning", "low": "info"}
            kind = sev_map.get(severity, "warning")
            st.markdown(
                banner(
                    f"Iteration {iteration_count} — {n_fixes} fix{'es' if n_fixes != 1 else ''} required",
                    kind=kind,
                    title=f"Critic routed back · severity: {severity}",
                ),
                unsafe_allow_html=True,
            )

            if suggestions:
                st.markdown(section_header("What the critic wants fixed"), unsafe_allow_html=True)
                for i, s in enumerate(suggestions[:4], 1):
                    st.markdown(
                        f'<div style="display:flex;gap:0.6rem;padding:0.4rem 0;border-bottom:1px solid var(--border)">'
                        f'<span style="font-size:0.78rem;font-weight:600;color:var(--accent);min-width:1.2rem">{i}.</span>'
                        f'<span style="font-size:0.85rem;color:var(--text-sec)">{s}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            if last_scorecard:
                weak = sorted(
                    [(k, v) for k, v in last_scorecard.items() if k != "overall" and isinstance(v, (int, float))],
                    key=lambda x: x[1],
                )[:3]
                if weak:
                    st.markdown(
                        kpi_row([
                            {
                                "label": cat.replace("_", " ").title(),
                                "value": f"{val}/10",
                                "pill": "needs work" if val < 7 else "ok",
                                "pill_color": "red" if val < 5 else ("amber" if val < 7 else "green"),
                            }
                            for cat, val in weak
                        ]),
                        unsafe_allow_html=True,
                    )
            st.markdown(divider(), unsafe_allow_html=True)

        if iteration_count == 0:
            st.markdown(section_header("Cleaning complete"), unsafe_allow_html=True)
            cleaning_result = state_vals.get("cleaning_result", "")
            if cleaning_result:
                keywords = ["SAVED", "SPECIAL NULLS", "IMPUTATION", "OUTLIERS", "DROPPED", "TARGET COLUMN", "shape"]
                key_lines = [l for l in cleaning_result.split("\n") if any(k in l for k in keywords)]
                if key_lines:
                    rows_html = "".join(
                        f'<div style="font-family:var(--font-mono);font-size:0.78rem;'
                        f'padding:0.25rem 0;border-bottom:1px solid var(--border);'
                        f'color:var(--text-sec)">{l.strip()}</div>'
                        for l in key_lines[:15]
                    )
                    st.markdown(
                        f'<div style="background:var(--bg);border:1px solid var(--border);'
                        f'border-radius:var(--radius-card);padding:0.75rem 1rem">{rows_html}</div>',
                        unsafe_allow_html=True,
                    )
            with st.expander("Full cleaning output"):
                st.code(cleaning_result, language="text")
            with st.expander("Cleaning code"):
                st.code(state_vals.get("cleaning_code", "No code yet"), language="python")

            _info_only = {"duplicate_row_pct", "row_count"}
            unit_tests = state_vals.get("unit_test_results", {})
            if unit_tests:
                if unit_tests.get("all_passed"):
                    dup_pct = unit_tests.get("duplicate_row_pct")
                    dup_note = f" ({dup_pct}% duplicate rows — normal for survey data)" if dup_pct and dup_pct > 0 else ""
                    st.markdown(
                        banner(f"All unit tests passed{dup_note}", kind="success"),
                        unsafe_allow_html=True,
                    )
                else:
                    failed_checks = {k: v for k, v in unit_tests.items()
                                     if v is False and k != "all_passed" and k not in _info_only}
                    if failed_checks:
                        hints = {
                            "target_is_numeric": "Target column must be 0/1 integers.",
                            "no_missing_values": "Null values remain in cleaned data.",
                            "no_duplicate_columns": "Duplicate column names — check OHE output.",
                        }
                        body = "".join(
                            f'<div style="margin-top:3px">· {k.replace("_"," ").title()}'
                            f'{": " + hints[k] if k in hints else ""}</div>'
                            for k in failed_checks
                        )
                        st.markdown(
                            banner(body, kind="error", title="Unit test failures"),
                            unsafe_allow_html=True,
                        )

    # ── Modeler checkpoint ────────────────────────────────────────────────────
    elif next_node == "modeler":
        st.markdown(section_header("Feature engineering complete"), unsafe_allow_html=True)
        feature_result = state_vals.get("feature_result", "")
        if feature_result:
            keywords = ["CREATED", "DROPPED", "LOG TRANSFORM", "INTERACTION", "SAVED", "FINAL SHAPE"]
            key_lines = [l.strip() for l in feature_result.split("\n")
                         if any(k in l.upper() for k in keywords) and l.strip()]
            if key_lines:
                rows_html = "".join(
                    f'<div style="font-family:var(--font-mono);font-size:0.78rem;'
                    f'padding:0.25rem 0;border-bottom:1px solid var(--border);'
                    f'color:var(--text-sec)">'
                    f'{"🗑 " if "DROP" in l.upper() else ("✓ " if "SAVED" in l.upper() else "  ")}{l}'
                    f'</div>'
                    for l in key_lines[:12]
                )
                st.markdown(
                    f'<div style="background:var(--bg);border:1px solid var(--border);'
                    f'border-radius:var(--radius-card);padding:0.75rem 1rem">{rows_html}</div>',
                    unsafe_allow_html=True,
                )

        unit_tests = state_vals.get("unit_test_results", {})
        if unit_tests:
            if unit_tests.get("all_passed"):
                grew = unit_tests.get("feature_count_grew")
                st.markdown(
                    banner(
                        "Feature count grew as expected" if grew else "Features verified",
                        kind="success",
                    ),
                    unsafe_allow_html=True,
                )
            else:
                failed_checks = {k: v for k, v in unit_tests.items()
                                 if v is False and k != "all_passed"}
                if failed_checks:
                    body = "".join(
                        f'<div style="margin-top:3px">· {k.replace("_"," ").title()}</div>'
                        for k in failed_checks
                    )
                    st.markdown(
                        banner(body, kind="error", title="Unit test failures"),
                        unsafe_allow_html=True,
                    )

        with st.expander("Feature engineering code"):
            st.code(state_vals.get("feature_code", "No code yet"), language="python")

    # ── Action row ────────────────────────────────────────────────────────────
    st.markdown(divider(), unsafe_allow_html=True)

    st.markdown(
        '<p style="font-size:0.8rem;color:var(--text-muted);margin-bottom:0.4rem">'
        'Optional — provide instructions for the next agent</p>',
        unsafe_allow_html=True,
    )
    feedback = st.text_area(
        "Feedback",
        height=72,
        key="approval_feedback",
        label_visibility="collapsed",
        placeholder="e.g., 'Focus on recall', 'Drop the tenure column', 'Use only tree-based models'",
    )

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        if st.button("Approve & continue", type="primary", use_container_width=True):
            on_approve(feedback)
    with c2:
        if st.button("Submit feedback & continue", use_container_width=True):
            on_submit_feedback(feedback)
    with c3:
        if st.button("Stop pipeline", use_container_width=True):
            on_stop()
