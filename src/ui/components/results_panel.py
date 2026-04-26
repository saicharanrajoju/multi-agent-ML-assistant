import re
import streamlit as st
import os
import plotly.graph_objects as go
import plotly.figure_factory as ff
from src.ui.ui_components import banner, empty_state, section_header, kpi_row


# ── Teacher narration helper ──────────────────────────────────────────────────

def _render_narration(narration: str, label: str = "Agent reasoning"):
    """Render a teacher narration in an expander using design system banner."""
    if not narration:
        return
    with st.expander(label, expanded=False):
        st.markdown(
            f'<div class="ds-banner ds-banner-info">'
            f'<span class="ds-banner-icon">ℹ</span>'
            f'<div class="ds-banner-body">{narration}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Reasoning annotation helpers ─────────────────────────────────────────────

def _cleaning_why(line: str, rc: dict, issues: list) -> str:
    """Return a one-sentence 'why' for a cleaning stdout line."""
    u = line.upper()
    null_patterns = rc.get("null_patterns", {})

    if "DROPPED" in u and "COLUMN" in u:
        m = re.search(r"DROPPED[^:]*:\s*([^\s—–-]+)", line, re.IGNORECASE)
        col = m.group(1).strip("`,") if m else None
        reason = null_patterns.get(col, "")
        if reason:
            return f"Why: `{col}` null pattern — {reason}. Dropping avoids imputing an unreliable column."
        # Fall back to checking issues list for a matching mention
        col_issue = next((iss for iss in (issues or []) if col and col.lower() in iss.lower()), "")
        if col_issue:
            return f"Why: {col_issue}"
        return "Why: too many missing values or zero variance — keeping it would add noise without signal."

    if "OUTLIERS" in u or "CAPPED" in u or "CLIP" in u:
        return ("Why: extreme values detected via IQR bounds. Capped (not removed) to preserve row count "
                "while preventing outliers from distorting the model's loss function.")

    if "IMPUTATION" in u or "IMPUTED" in u or "FILL" in u:
        if "MEDIAN" in u:
            return "Why: median imputation is robust to outliers (unlike mean), so extreme values don't skew the fill."
        if "MODE" in u:
            return "Why: mode imputation for categorical — replaces missing with the most common value in that column."
        return "Why: missing values must be filled before training; ML models can't handle NaN inputs."

    if "ENCODING" in u or "LABEL" in u or "ONE-HOT" in u or "OHE" in u:
        return ("Why: ML models require numeric inputs. Categorical strings are converted to integers "
                "(label encoding for ordinal / OHE for nominal) so the model can compute distances and gradients.")

    if "TARGET COLUMN" in u or "TARGET ENCODE" in u:
        return ("Why: the target must be 0/1 integers for binary classification. "
                "String labels (e.g. 'M'/'B') are mapped to 0 and 1.")

    if "SPECIAL NULLS" in u or "SENTINEL" in u:
        return ("Why: some datasets encode missing data as specific values (e.g. -1, 'Unknown', 999). "
                "These are detected and treated as NaN before imputation.")

    if "DUPLICATE" in u:
        return "Why: exact duplicate rows add no new information and can cause the model to overfit to repeated examples."

    if "SKEW" in u or "LOG" in u:
        return ("Why: highly skewed numeric features distort distance-based models and linear coefficients. "
                "Log-transforming compresses the tail so the distribution is closer to normal.")

    return ""


def _feature_why(line: str, rc: dict, dataset_summary: dict) -> str:
    """Return a one-sentence 'why' for a feature engineering stdout line."""
    u = line.upper()
    top_corr = rc.get("correlations_with_target") or dataset_summary.get("correlations_with_target", {})
    skewed = dataset_summary.get("skewed_columns", [])

    if "DROPPED" in u and ("MULTICOLLINEAR" in u or "CORR" in u):
        # Try to extract correlation value from the line
        m = re.search(r"(\d+\.\d+)", line)
        corr_val = m.group(1) if m else None
        if corr_val:
            return (f"Why: correlation of {corr_val} exceeds the 0.92 threshold. "
                    "Two near-identical features inflate coefficients and make the model unstable "
                    "— keeping one preserves the information without the redundancy.")
        return ("Why: near-duplicate features (|correlation| > 0.92) add no new information "
                "and inflate model complexity.")

    if "LOG" in u and ("TRANSFORM" in u or "LOG1P" in u or "_LOG" in u):
        m = re.search(r"skew[^\d]*(\d+\.?\d*)", line, re.IGNORECASE)
        skew_val = m.group(1) if m else None
        # Check if the column appears in the profiler's known skewed list
        col_m = re.search(r"(?:applied|transform)[ed\s]*:?\s*([a-z_][a-z0-9_ ]*?)(?:\s|,|\(|$)", line, re.IGNORECASE)
        col_name = col_m.group(1).strip() if col_m else None
        in_skewed = col_name and any(col_name.lower() in s.lower() for s in skewed)
        skew_hint = f" (skewness = {skew_val})" if skew_val else (" — profiler flagged as highly skewed" if in_skewed else "")
        return (f"Why: the feature has a long right tail{skew_hint}. np.log1p compresses extreme values "
                "so the model treats a 10x difference proportionally, not as a fixed gap.")

    if "INTERACTION" in u or "_X_" in u:
        top_names = list(top_corr.keys())[:2] if top_corr else []
        hint = f" ({' × '.join(top_names)})" if top_names else ""
        return (f"Why: multiplying top-correlated features{hint} captures non-linear joint effects "
                "the model can't learn from each feature alone.")

    if "BIN" in u or "QCUT" in u:
        return ("Why: discretising a highly-skewed continuous column into quantile bins lets "
                "the model capture threshold effects (e.g. 'very high' vs 'moderate') "
                "that a raw number might obscure.")

    if "FREQUENCY ENCOD" in u or "FREQ_ENCOD" in u:
        return ("Why: replacing a high-cardinality categorical with its frequency turns "
                "rare/common categories into a meaningful numeric signal without exploding dimensionality.")

    if "MISSING INDICATOR" in u or "WAS_MISSING" in u:
        return ("Why: missingness itself can be informative — a value being absent might correlate "
                "with the target. A binary flag preserves that signal after imputation.")

    if "CREATED" in u or "NEW FEATURE" in u:
        return "Why: derived feature engineered to capture a relationship not directly present in the raw data."

    if "SAVED" in u or "FINAL SHAPE" in u:
        return ""  # no caption needed for the save confirmation line

    return ""


def _render_annotated_steps(lines: list, why_fn, *why_args):
    """Render step lines with an optional 'why' caption below each."""
    for line in lines:
        u = line.upper()
        if "WARNING" in u or "LEAKAGE" in u:
            st.markdown(
                f'<div class="ds-banner ds-banner-warning">'
                f'<span class="ds-banner-icon">⚠</span>'
                f'<div class="ds-banner-body">{line}</div></div>',
                unsafe_allow_html=True,
            )
        elif "SAVED" in u or "FINAL" in u or "TARGET" in u:
            st.markdown(
                f'<div class="ds-banner ds-banner-success">'
                f'<span class="ds-banner-icon">✓</span>'
                f'<div class="ds-banner-body">{line}</div></div>',
                unsafe_allow_html=True,
            )
            continue
        else:
            if "DROPPED" in u:
                icon, color = "−", "var(--error)"
            elif "IMPUTATION" in u or "IMPUTED" in u or "FILL" in u:
                icon, color = "~", "var(--accent)"
            elif "OUTLIERS" in u or "CAPPED" in u:
                icon, color = "↔", "var(--warn)"
            elif "LOG" in u or "TRANSFORM" in u:
                icon, color = "ƒ", "var(--accent)"
            elif "INTERACTION" in u or "_X_" in u:
                icon, color = "×", "var(--accent)"
            elif "ENCODING" in u or "LABEL" in u:
                icon, color = "#", "var(--text-sec)"
            else:
                icon, color = "›", "var(--text-muted)"
            st.markdown(
                f'<div style="display:flex;gap:0.6rem;padding:0.3rem 0;border-bottom:1px solid var(--border)">'
                f'<span style="font-family:var(--font-mono);font-size:0.8rem;color:{color};'
                f'min-width:1rem;flex-shrink:0">{icon}</span>'
                f'<span style="font-family:var(--font-mono);font-size:0.8rem;color:var(--text-sec)">{line}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        why = why_fn(line, *why_args)
        if why:
            st.markdown(
                f'<div style="font-size:0.78rem;color:var(--text-muted);padding:0.2rem 0 0.4rem 1.6rem">{why}</div>',
                unsafe_allow_html=True,
            )


def render_results_panel(state_vals: dict, project_root: str):
    tab_profile, tab_cleaning, tab_feature_eng, tab_reasoning, tab_results, tab_critique, tab_chat = st.tabs([
        "Data Profile",
        "Cleaning",
        "Feature Engineering",
        "Reasoning",
        "Model Results",
        "Critique & Scorecard",
        "Ask the AI",
    ])

    with tab_profile:
        state_vals = state_vals or {}
        report       = state_vals.get("profile_report", "")
        ds           = state_vals.get("dataset_summary", {}) or {}
        rc           = state_vals.get("reasoning_context", {}) or {}
        col_info     = state_vals.get("column_info", {}) or {}
        target_col   = state_vals.get("target_column", "")
        problem_type = state_vals.get("problem_type", "")
        user_goal    = state_vals.get("user_goal", "")
        issues       = state_vals.get("data_issues", [])

        if report or ds:
            _render_narration(state_vals.get("profiler_narration", ""))

            # ── 1. Goal & problem type banner ─────────────────────────────────
            if user_goal or problem_type:
                prob_display = problem_type.replace("_", " ").title() if problem_type else "Unknown"
                metric_display = state_vals.get("recommended_metric", "").upper()
                goal_html = f'<strong>Goal:</strong> {user_goal}<br>' if user_goal else ""
                st.markdown(
                    banner(
                        f'{goal_html}'
                        f'<strong>Problem type:</strong> {prob_display} &nbsp;·&nbsp; '
                        f'<strong>Optimising for:</strong> {metric_display} &nbsp;·&nbsp; '
                        f'<strong>Target column:</strong> <code>{target_col}</code>',
                        kind="info",
                    ),
                    unsafe_allow_html=True,
                )

            # ── 2. Dataset KPI row ────────────────────────────────────────────
            shape   = ds.get("shape", [])
            n_rows  = shape[0] if shape else rc.get("n_rows", "N/A")
            n_cols  = shape[1] if len(shape) > 1 else rc.get("n_cols", "N/A")
            n_num   = len(ds.get("numeric_columns", []))
            n_cat   = len(ds.get("categorical_columns", []))
            missing = ds.get("missing_values", {})
            n_miss  = sum(v for v in missing.values() if isinstance(v, (int, float)) and v > 0)

            st.markdown(section_header("Dataset overview"), unsafe_allow_html=True)
            st.markdown(
                kpi_row([
                    {"label": "Rows",             "value": f"{n_rows:,}" if isinstance(n_rows, int) else str(n_rows)},
                    {"label": "Columns",          "value": str(n_cols)},
                    {"label": "Numeric features", "value": str(n_num)},
                    {"label": "Categorical",      "value": str(n_cat)},
                    {"label": "Missing values",   "value": "None" if n_miss == 0 else str(n_miss),
                     "pill": "clean" if n_miss == 0 else "check",
                     "pill_color": "green" if n_miss == 0 else "amber"},
                ]),
                unsafe_allow_html=True,
            )

            # ── 3. Column inventory ───────────────────────────────────────────
            num_cols  = ds.get("numeric_columns", [])
            cat_cols  = ds.get("categorical_columns", [])
            corr_map  = ds.get("correlations_with_target", {})
            all_cols  = list(dict.fromkeys(num_cols + cat_cols + ([target_col] if target_col else [])))

            if all_cols:
                st.markdown(section_header("Column inventory", "Every column, its type, and its relationship to the target"), unsafe_allow_html=True)

                # Header row
                st.markdown(
                    '<div style="display:grid;grid-template-columns:1.8fr 0.8fr 0.7fr 1.2fr 2.2fr;'
                    'gap:0.5rem;padding:0.3rem 0.5rem;background:rgba(59,35,20,0.08);'
                    'border-radius:6px 6px 0 0;font-size:0.75rem;font-weight:700;color:var(--text-muted)">'
                    '<span>Column</span><span>Type</span><span>Role</span>'
                    '<span>Correlation with target</span><span>Description</span></div>',
                    unsafe_allow_html=True,
                )

                rows_html = ""
                for col in all_cols:
                    is_target = col == target_col
                    col_type  = "numeric" if col in num_cols else ("categorical" if col in cat_cols else ("numeric" if is_target else "unknown"))
                    role      = "target" if is_target else "feature"
                    corr_val  = corr_map.get(col)
                    if corr_val is not None and not is_target:
                        try:
                            cv = float(corr_val)
                            bar_width = int(abs(cv) * 80)
                            bar_color = "#2D7A4A" if cv > 0 else "#C62828"
                            corr_html = (
                                f'<div style="display:flex;align-items:center;gap:0.4rem">'
                                f'<div style="width:{bar_width}px;height:8px;background:{bar_color};'
                                f'border-radius:2px;min-width:2px"></div>'
                                f'<span style="font-size:0.75rem;color:var(--text-muted)">{cv:+.3f}</span>'
                                f'</div>'
                            )
                        except (TypeError, ValueError):
                            corr_html = '<span style="font-size:0.75rem;color:var(--text-muted)">—</span>'
                    elif is_target:
                        corr_html = '<span style="font-size:0.75rem;color:var(--accent-dark)">← this is the target</span>'
                    else:
                        corr_html = '<span style="font-size:0.75rem;color:var(--text-muted)">—</span>'

                    # Description from LLM column_info or sensible fallback
                    info_entry = col_info.get(col, {})
                    if isinstance(info_entry, dict):
                        desc = info_entry.get("description", info_entry.get("type", ""))
                    else:
                        desc = str(info_entry)
                    if not desc:
                        if is_target:
                            desc = "The value the model is trained to predict"
                        elif col_type == "numeric":
                            desc = "Numeric feature"
                        else:
                            desc = "Categorical feature"

                    type_color  = "var(--accent-dark)" if col_type == "numeric" else "var(--text-muted)"
                    role_color  = "var(--success)" if is_target else "var(--text-muted)"
                    row_bg      = "rgba(59,35,20,0.04)" if is_target else "transparent"
                    has_missing = col in missing and missing[col] > 0
                    miss_badge  = (f'<span style="font-size:0.68rem;background:rgba(180,83,9,0.15);'
                                   f'color:#B45309;border-radius:3px;padding:1px 5px;margin-left:4px">'
                                   f'{int(missing[col])} null</span>') if has_missing else ""

                    rows_html += (
                        f'<div style="display:grid;grid-template-columns:1.8fr 0.8fr 0.7fr 1.2fr 2.2fr;'
                        f'gap:0.5rem;padding:0.35rem 0.5rem;border-bottom:1px solid var(--border);'
                        f'background:{row_bg};align-items:center">'
                        f'<span style="font-family:var(--font-mono);font-size:0.8rem;color:var(--text-pri);font-weight:{"700" if is_target else "400"}">'
                        f'{col}{miss_badge}</span>'
                        f'<span style="font-size:0.78rem;color:{type_color}">{col_type}</span>'
                        f'<span style="font-size:0.78rem;color:{role_color};font-weight:{"600" if is_target else "400"}">{role}</span>'
                        f'<div>{corr_html}</div>'
                        f'<span style="font-size:0.78rem;color:var(--text-muted)">{desc}</span>'
                        f'</div>'
                    )
                st.markdown(
                    f'<div style="background:var(--surface);border:1px solid var(--border);'
                    f'border-radius:var(--radius-card);overflow:hidden">{rows_html}</div>',
                    unsafe_allow_html=True,
                )

            # ── 4. Missing values detail ──────────────────────────────────────
            missing_nonzero = {k: v for k, v in missing.items() if isinstance(v, (int, float)) and v > 0}
            if missing_nonzero:
                st.markdown(section_header("Missing values by column"), unsafe_allow_html=True)
                total_rows = n_rows if isinstance(n_rows, int) else 1
                mv_rows = "".join(
                    f'<div style="display:flex;align-items:center;gap:0.75rem;padding:0.3rem 0;border-bottom:1px solid var(--border)">'
                    f'<span style="font-family:var(--font-mono);font-size:0.8rem;min-width:140px;color:var(--text-pri)">{col}</span>'
                    f'<div style="flex:1;background:rgba(59,35,20,0.08);border-radius:4px;height:10px">'
                    f'<div style="width:{min(100, int(cnt/total_rows*100))}%;height:10px;background:#B45309;border-radius:4px"></div></div>'
                    f'<span style="font-size:0.78rem;color:var(--text-muted);min-width:80px">{int(cnt):,} ({cnt/total_rows*100:.1f}%)</span>'
                    f'</div>'
                    for col, cnt in sorted(missing_nonzero.items(), key=lambda x: x[1], reverse=True)
                )
                st.markdown(
                    f'<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-card);padding:0.5rem 1rem">{mv_rows}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(banner("No missing values — dataset is complete.", kind="success"), unsafe_allow_html=True)

            # ── 5. Top correlations chart ─────────────────────────────────────
            if corr_map:
                import plotly.graph_objects as go
                sorted_corr = sorted(
                    [(k, float(v)) for k, v in corr_map.items() if k != target_col and isinstance(v, (int, float))],
                    key=lambda x: abs(x[1]), reverse=True
                )[:12]
                if sorted_corr:
                    st.markdown(section_header("Feature correlations with target", f"Pearson |r| with {target_col}"), unsafe_allow_html=True)
                    names_c  = [x[0] for x in sorted_corr]
                    vals_c   = [x[1] for x in sorted_corr]
                    colors_c = ["#2D7A4A" if v > 0 else "#C62828" for v in vals_c]
                    fig_corr = go.Figure(go.Bar(
                        x=vals_c, y=names_c, orientation='h',
                        marker_color=colors_c,
                        text=[f"{v:+.3f}" for v in vals_c],
                        textposition='outside',
                        textfont=dict(color="#3B2314", size=10),
                    ))
                    fig_corr.update_layout(
                        xaxis_title="Pearson correlation",
                        xaxis=dict(tickfont=dict(color="#3B2314", size=10), title_font=dict(color="#3B2314"),
                                   showgrid=True, gridcolor="rgba(59,35,20,0.15)", zeroline=True,
                                   zerolinecolor="#3B2314", zerolinewidth=1),
                        yaxis=dict(tickfont=dict(color="#3B2314", size=10), title_font=dict(color="#3B2314"), showgrid=False),
                        height=max(260, 32 * len(names_c)),
                        paper_bgcolor="#FFE8D6", plot_bgcolor="#FFE8D6",
                        font=dict(family="Inter, sans-serif", size=11, color="#3B2314"),
                        margin=dict(l=0, r=60, t=8, b=0),
                    )
                    st.plotly_chart(fig_corr, width="stretch")

            # ── 6. Issues ─────────────────────────────────────────────────────
            if issues:
                st.markdown('<hr style="border:none;border-top:1px solid var(--border);margin:1rem 0">', unsafe_allow_html=True)
                body = "".join(f'<div style="margin-top:3px">· {i}</div>' for i in issues)
                st.markdown(
                    banner(body, kind="warning", title=f"{len(issues)} issue{'s' if len(issues) != 1 else ''} identified"),
                    unsafe_allow_html=True,
                )

            # ── 7. Raw profiling output (collapsable) ─────────────────────────
            st.markdown('<hr style="border:none;border-top:1px solid var(--border);margin:1rem 0">', unsafe_allow_html=True)
            with st.expander("Raw profiling output (detailed statistics)", expanded=False):
                st.markdown(report)
                st.download_button(
                    "Download profile report",
                    data=report,
                    file_name="data_profile_report.md",
                    mime="text/markdown",
                )
        else:
            st.markdown(
                empty_state("", "No profile yet", "The data profile will appear here after the Profiler agent runs.", "Run the pipeline to get started"),
                unsafe_allow_html=True,
            )


    with tab_cleaning:
        state_vals = state_vals or {}
        cleaning_result = state_vals.get("cleaning_result", "")
        cleaning_code = state_vals.get("cleaning_code", "")
        cleaning_summary = state_vals.get("cleaning_summary", {})
        unit_tests = state_vals.get("unit_test_results", {})

        if cleaning_result or cleaning_code:
            st.markdown(section_header("Data cleaning"), unsafe_allow_html=True)
            _render_narration(state_vals.get("cleaning_narration", ""))

            # Unit test status banner
            if unit_tests:
                info_only_keys = {"duplicate_row_pct", "row_count"}
                failed = {k: v for k, v in unit_tests.items()
                          if v is False and k != "all_passed" and k not in info_only_keys}

                if unit_tests.get("all_passed"):
                    st.markdown(banner("Cleaned data verified — all checks passed", kind="success"), unsafe_allow_html=True)
                elif failed:
                    hints = {
                        "target_is_numeric": "Target column must be 0/1 integers.",
                        "no_missing_values": "Null values remain in cleaned data.",
                        "no_duplicate_columns": "Duplicate column names — check OHE output.",
                        "target_column_present": "Target column was accidentally dropped.",
                    }
                    body = "".join(
                        f'<div style="margin-top:3px">· {k.replace("_"," ").title()}'
                        f'{": " + hints[k] if k in hints else ""}</div>'
                        for k in failed
                    )
                    st.markdown(banner(body, kind="error", title="Unit test failures"), unsafe_allow_html=True)

                dup_pct = unit_tests.get("duplicate_row_pct")
                if dup_pct is not None and dup_pct > 0:
                    st.markdown(
                        banner(f"{dup_pct}% of rows share identical feature values — normal for survey/census data.", kind="info"),
                        unsafe_allow_html=True,
                    )

            # Summary metrics
            if cleaning_summary:
                shape_after = cleaning_summary.get("shape_after", [])
                cols_after = cleaning_summary.get("columns_after", [])
                no_missing = cleaning_summary.get("no_missing", False)
                target_type = cleaning_summary.get("target_type", "")

                st.markdown(section_header("Summary"), unsafe_allow_html=True)
                st.markdown(
                    kpi_row([
                        {"label": "Rows",          "value": f"{shape_after[0]:,}" if shape_after else "N/A"},
                        {"label": "Columns",       "value": str(shape_after[1]) if len(shape_after) > 1 else "N/A"},
                        {"label": "Missing values","value": "None" if no_missing else "Remain", "pill": "clean" if no_missing else "check", "pill_color": "green" if no_missing else "amber"},
                        {"label": "Target type",   "value": target_type or "N/A"},
                    ]),
                    unsafe_allow_html=True,
                )

                if cols_after:
                    st.markdown(section_header("Columns in cleaned dataset"), unsafe_allow_html=True)
                    pills = "".join(
                        f'<span style="display:inline-block;font-family:var(--font-mono);font-size:0.75rem;'
                        f'background:rgba(59,35,20,0.10);color:var(--text-sec);padding:2px 8px;border-radius:4px;'
                        f'margin:2px 3px">{c}</span>'
                        for c in cols_after
                    )
                    st.markdown(f'<div style="line-height:2">{pills}</div>', unsafe_allow_html=True)

            # Key cleaning steps with reasoning
            rc = state_vals.get("reasoning_context", {})
            issues = state_vals.get("data_issues", [])
            if cleaning_result and "FAILED" not in cleaning_result:
                keywords = ["SPECIAL NULLS", "DROPPED", "IMPUTATION", "OUTLIERS",
                            "SAVED", "TARGET COLUMN", "FINAL SHAPE", "ENCODING", "WARNING",
                            "DUPLICATE", "SENTINEL", "SKEW", "LOG"]
                key_lines = [l.strip() for l in cleaning_result.split("\n")
                             if any(k in l.upper() for k in keywords) and l.strip()]
                with st.expander(f"Steps performed ({len(key_lines)} actions)", expanded=False):
                    if key_lines:
                        st.markdown('<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-card);padding:0.5rem 1rem;margin-bottom:0.75rem">', unsafe_allow_html=True)
                        _render_annotated_steps(key_lines, _cleaning_why, rc, issues)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.code(cleaning_result[:1000], language="text")
            elif cleaning_result and "FAILED" in cleaning_result:
                st.markdown(banner("Cleaning failed — see full output below", kind="error"), unsafe_allow_html=True)
                st.code(cleaning_result, language="text")

            with st.expander("Full cleaning output"):
                st.code(cleaning_result or "No output captured", language="text")

            with st.expander("Cleaning code"):
                st.code(cleaning_code or "No code generated", language="python")

            if cleaning_code:
                st.download_button(
                    "Download cleaning code",
                    data=cleaning_code,
                    file_name="cleaning_code.py",
                    mime="text/plain",
                )

            # ── Cleaned dataset preview & download ────────────────────────────
            _clean_run_id  = (st.session_state.get("pipeline_config") or {}).get("configurable", {}).get("thread_id", "default")
            _clean_csv     = os.path.join(project_root, "outputs", "checkpoints", _clean_run_id, "cleaned_data.csv")
            if os.path.exists(_clean_csv):
                st.markdown('<hr style="border:none;border-top:1px solid var(--border);margin:1rem 0">', unsafe_allow_html=True)
                with st.expander("Cleaned dataset — preview & download", expanded=False):
                    try:
                        import pandas as pd
                        _df_clean = pd.read_csv(_clean_csv)
                        st.markdown(
                            f'<p style="font-size:0.82rem;color:var(--text-muted);margin-bottom:0.5rem">'
                            f'{len(_df_clean):,} rows × {_df_clean.shape[1]} columns after cleaning</p>',
                            unsafe_allow_html=True,
                        )
                        st.dataframe(_df_clean.head(50), width="stretch", hide_index=True)
                        _clean_csv_bytes = _df_clean.to_csv(index=False).encode()
                        st.download_button(
                            "Download cleaned dataset (.csv)",
                            data=_clean_csv_bytes,
                            file_name="cleaned_data.csv",
                            mime="text/csv",
                        )
                    except Exception as _ce:
                        st.error(f"Could not load cleaned dataset: {_ce}")
        else:
            st.markdown(
                empty_state("", "No cleaning results yet", "Cleaning results will appear here after the Cleaner agent runs."),
                unsafe_allow_html=True,
            )


    with tab_feature_eng:
        feature_code = state_vals.get("feature_code", "")
        feature_result = state_vals.get("feature_result", "")
        feat_unit_tests = state_vals.get("unit_test_results", {})

        if feature_code or feature_result:
            st.markdown(section_header("Feature engineering"), unsafe_allow_html=True)
            _render_narration(state_vals.get("feature_narration", ""))

            # Unit test banner
            if feat_unit_tests:
                if feat_unit_tests.get("all_passed"):
                    st.markdown(banner("Engineered features verified — all checks passed", kind="success"), unsafe_allow_html=True)
                else:
                    failed = {k: v for k, v in feat_unit_tests.items()
                              if v is False and k != "all_passed"}
                    if failed:
                        body = "".join(f'<div style="margin-top:3px">· {k.replace("_"," ").title()}</div>' for k in failed)
                        st.markdown(banner(body, kind="error", title="Unit test failures"), unsafe_allow_html=True)

            # Metrics row
            grew = feat_unit_tests.get("feature_count_grew")
            no_missing = feat_unit_tests.get("no_missing_values")
            no_inf = feat_unit_tests.get("no_infinite_values")

            def _bool_val(v): return "Yes" if v else ("No" if v is False else "N/A")
            def _bool_color(v): return "green" if v else ("red" if v is False else "gray")
            st.markdown(
                kpi_row([
                    {"label": "Feature count grew", "value": _bool_val(grew),       "pill": _bool_val(grew),       "pill_color": _bool_color(grew)},
                    {"label": "No missing values",  "value": _bool_val(no_missing), "pill": _bool_val(no_missing), "pill_color": _bool_color(no_missing)},
                    {"label": "No infinite values", "value": _bool_val(no_inf),     "pill": _bool_val(no_inf),     "pill_color": _bool_color(no_inf)},
                ]),
                unsafe_allow_html=True,
            )

            # Key steps from stdout with reasoning
            if feature_result and "FAILED" not in feature_result:
                fe_rc = state_vals.get("reasoning_context", {})
                fe_ds = state_vals.get("dataset_summary", {})
                keywords = ["CREATED", "DROPPED", "LOG", "INTERACTION", "_X_",
                            "BINNED", "ENCODED", "SAVED", "FINAL SHAPE", "WARNING",
                            "LEAKAGE", "MISSING INDICATOR", "FREQUENCY"]
                key_lines = [l.strip() for l in feature_result.split("\n")
                             if any(k in l.upper() for k in keywords) and l.strip()]
                with st.expander(f"Steps performed ({len(key_lines)} actions)", expanded=False):
                    if key_lines:
                        st.markdown('<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-card);padding:0.5rem 1rem;margin-bottom:0.75rem">', unsafe_allow_html=True)
                        _render_annotated_steps(key_lines, _feature_why, fe_rc, fe_ds)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.code(feature_result[:1000], language="text")
            elif feature_result and "FAILED" in feature_result:
                st.markdown(banner("Feature engineering failed — see full output below", kind="error"), unsafe_allow_html=True)
                st.code(feature_result, language="text")

            with st.expander("Full feature engineering output"):
                st.code(feature_result or "No output captured", language="text")

            with st.expander("Feature engineering code"):
                st.code(feature_code or "No code generated", language="python")

            if feature_code:
                st.download_button(
                    "Download feature engineering code",
                    data=feature_code,
                    file_name="feature_engineering_code.py",
                    mime="text/plain",
                )

            # ── Correlation heatmap ───────────────────────────────────────────
            target_col = state_vals.get("target_column", "")
            run_id = st.session_state.get("pipeline_config", {})
            run_id = (run_id or {}).get("configurable", {}).get("thread_id", "default")
            featured_csv = os.path.join(
                project_root, "outputs", "checkpoints", run_id, "featured_data.csv"
            )
            if os.path.exists(featured_csv) and target_col:
                try:
                    import pandas as pd
                    _df = pd.read_csv(featured_csv)
                    numeric_cols = _df.select_dtypes(include="number").columns.tolist()
                    if target_col in numeric_cols:
                        corr_with_target = _df[numeric_cols].corr()[target_col].drop(target_col).abs()
                        top_cols = corr_with_target.nlargest(min(20, len(corr_with_target))).index.tolist()
                        display_cols = top_cols + [target_col]
                        corr_matrix = _df[display_cols].corr()

                        st.markdown(section_header("Feature correlation heatmap", f"Top {len(top_cols)} features by |correlation| with {target_col}"), unsafe_allow_html=True)
                        fig_heat = ff.create_annotated_heatmap(
                            z=corr_matrix.values.round(2).tolist(),
                            x=list(corr_matrix.columns),
                            y=list(corr_matrix.index),
                            colorscale="Oranges",
                            reversescale=False,
                            showscale=True,
                            annotation_text=corr_matrix.values.round(2).astype(str).tolist(),
                            font_colors=["#3B2314", "#FFE8D6"],
                        )
                        _n = len(display_cols)
                        _label_len = max((len(c) for c in display_cols), default=10)
                        _left_margin   = max(120, _label_len * 7)
                        _bottom_margin = max(120, _label_len * 6)
                        fig_heat.update_layout(
                            height=max(420, 28 * _n),
                            margin=dict(l=_left_margin, r=20, t=20, b=_bottom_margin),
                            paper_bgcolor="#FFE8D6",
                            font=dict(family="Inter, sans-serif", size=10, color="#3B2314"),
                        )
                        fig_heat.update_xaxes(
                            tickangle=-45,
                            tickfont=dict(color="#3B2314", size=10),
                            title_font=dict(color="#3B2314"),
                        )
                        fig_heat.update_yaxes(
                            tickfont=dict(color="#3B2314", size=10),
                            title_font=dict(color="#3B2314"),
                        )
                        fig_heat.update_traces(colorbar=dict(
                            tickfont=dict(color="#3B2314"),
                            title=dict(text="r", font=dict(color="#3B2314", size=10)),
                        ))
                        st.plotly_chart(fig_heat, width="stretch")
                except Exception:
                    pass
        else:
            st.markdown(
                empty_state("", "No feature engineering results yet", "Results will appear here after the Feature Engineer agent runs."),
                unsafe_allow_html=True,
            )

    with tab_reasoning:
        state_vals = state_vals or {}
        rc = state_vals.get("reasoning_context", {}) or {}
        problem_type = state_vals.get("problem_type", "")
        recommended_metric = state_vals.get("recommended_metric", "")

        if problem_type:
            st.markdown(section_header("Pipeline reasoning", "A plain-English explanation of every decision the system made"), unsafe_allow_html=True)
            _render_narration(state_vals.get("profiler_narration", ""))

            target_col_r  = state_vals.get("target_column", "target")
            n_rows_r      = rc.get("n_rows", 0)
            n_cols_r      = rc.get("n_cols", 0)
            imb_r         = rc.get("imbalance_ratio", "N/A")
            imb_strat_r   = rc.get("imbalance_strategy", "none")
            rec_models_r  = rc.get("recommended_models", [])
            enc_map_r     = rc.get("encoding_map", {})
            feat_strats_r = rc.get("feature_strategies", [])
            metric_display_r = recommended_metric.upper()

            def _card(title, body_html):
                return (
                    f'<div style="background:var(--surface);border:1px solid var(--border);'
                    f'border-radius:var(--radius-card);padding:1rem 1.2rem;margin-bottom:0.9rem">'
                    f'<p style="font-size:0.72rem;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:0.07em;color:var(--text-muted);margin:0 0 0.5rem">{title}</p>'
                    f'<p style="font-size:0.88rem;color:var(--text-sec);line-height:1.7;margin:0">{body_html}</p>'
                    f'</div>'
                )

            # ── Card 1: About this dataset ────────────────────────────────────
            prob_sentence = {
                "binary_classification": f'The target column <code>{target_col_r}</code> has two possible values, so this is a <strong>binary classification</strong> problem — the model predicts yes or no.',
                "multiclass_classification": f'The target column <code>{target_col_r}</code> has multiple categories, so this is a <strong>multi-class classification</strong> problem.',
                "regression": f'The target column <code>{target_col_r}</code> is a continuous number, so this is a <strong>regression</strong> problem — the model predicts a numeric value.',
            }.get(problem_type, f'<strong>{problem_type.replace("_"," ").title()}</strong> problem detected.')

            try:
                imb_float = float(imb_r)
                if imb_float < 0.05:
                    imb_sentence = f' The positive class makes up only <strong>{imb_float*100:.1f}%</strong> of the dataset — this is extreme class imbalance, which requires special handling.'
                elif imb_float < 0.15:
                    imb_sentence = f' The class imbalance ratio is <strong>{imb_float:.2f}</strong> — significantly imbalanced.'
                elif imb_float < 0.4:
                    imb_sentence = f' The class imbalance ratio is <strong>{imb_float:.2f}</strong> — mildly imbalanced.'
                else:
                    imb_sentence = f' The class imbalance ratio is <strong>{imb_float:.2f}</strong> — classes are reasonably balanced.'
            except (TypeError, ValueError):
                imb_sentence = ""

            st.markdown(_card(
                "About this dataset",
                f'This dataset has <strong>{n_rows_r:,} rows</strong> and <strong>{n_cols_r} columns</strong>. '
                f'{prob_sentence}{imb_sentence}'
            ), unsafe_allow_html=True)

            # ── Card 2: Evaluation metric & imbalance strategy ────────────────
            metric_why_r = {
                "f1":      "F1 balances precision and recall — it's the right choice when both missing cases and false alarms matter.",
                "recall":  "Recall minimises missed positives. Chosen because your goal prioritised catching as many true cases as possible.",
                "precision": "Precision minimises false alarms. Chosen because your goal prioritised only flagging high-confidence cases.",
                "roc_auc": "ROC-AUC measures ranking ability across all thresholds — useful when the exact cutoff isn't fixed.",
                "accuracy": "Accuracy counts the fraction of correct predictions. Suitable for balanced classes.",
                "rmse":    "RMSE penalises large errors more than small ones — the standard metric for regression.",
                "mae":     "MAE treats all errors equally — chosen from your goal description.",
                "r2":      "R² measures how much variance the model explains — standard for evaluating regression quality.",
            }.get(recommended_metric, f"Metric <code>{recommended_metric}</code> chosen from your goal.")

            imb_why_r = {
                "none":                        "The classes are balanced, so no special handling was needed — the model was trained normally.",
                "class_weight_balanced":       "Mild imbalance detected. <code>class_weight='balanced'</code> was applied so the model penalises misclassifying minority cases more heavily.",
                "smote_plus_class_weight":     "Significant imbalance detected. SMOTE was used to generate synthetic minority samples, and class weights were applied on top. This prevents the model from learning to just predict the majority class.",
                "threshold_tuning_plus_pr_auc":"Extreme imbalance detected. The decision threshold was tuned away from 0.5, and PR-AUC was used instead of accuracy — because accuracy is misleading when positives are rare.",
            }.get(imb_strat_r, f"Strategy <code>{imb_strat_r}</code> selected.")

            st.markdown(_card(
                "Evaluation metric & imbalance handling",
                f'Optimising for <strong>{metric_display_r}</strong>. {metric_why_r}'
                f'<br><br>{imb_why_r}'
            ), unsafe_allow_html=True)

            # ── Card 3: Model candidates ──────────────────────────────────────
            if rec_models_r:
                if n_rows_r < 500:
                    size_sentence_r = f'With only {n_rows_r} rows this is a small dataset, so simpler models were prioritised to avoid overfitting.'
                elif n_rows_r < 50_000:
                    size_sentence_r = f'With {n_rows_r:,} rows — a medium-sized dataset — both linear baselines and tree-based ensembles were tried.'
                else:
                    size_sentence_r = f'With {n_rows_r:,} rows, gradient boosting models were prioritised as they typically dominate on large tabular data.'

                model_explain_r = {
                    "LogisticRegression": "fast interpretable linear baseline",
                    "RandomForestClassifier": "ensemble of trees, handles non-linearity well",
                    "XGBClassifier": "powerful gradient boosting",
                    "LGBMClassifier": "fast gradient boosting",
                    "GradientBoostingClassifier": "robust gradient boosting baseline",
                    "LinearRegression": "fast linear regression baseline",
                    "Ridge": "regularised linear — stable on small data",
                    "Lasso": "linear with built-in feature selection",
                    "RandomForestRegressor": "ensemble regression",
                    "XGBRegressor": "gradient boosting for regression",
                    "LGBMRegressor": "fast gradient boosting for regression",
                }
                model_lines_r = "<br>".join(
                    f'<strong>{m}</strong> — {model_explain_r.get(m, "selected based on dataset characteristics")}'
                    for m in rec_models_r
                )
                st.markdown(_card(
                    "Model candidates",
                    f'{size_sentence_r} The following model families were trained and compared on a stratified hold-out test set:<br><br>'
                    f'{model_lines_r}'
                ), unsafe_allow_html=True)

            # ── Card 4: Encoding & feature engineering ────────────────────────
            enc_parts_r = []
            for col_e, strat_e in list((enc_map_r or {}).items())[:8]:
                if isinstance(strat_e, dict):
                    t = strat_e.get("type", "unknown")
                    if t == "ordinal":
                        order = strat_e.get("order", [])
                        enc_parts_r.append(f'<code>{col_e}</code> → ordinal encoding ({" → ".join(order[:4])}{"…" if len(order) > 4 else ""})')
                    else:
                        enc_parts_r.append(f'<code>{col_e}</code> → {t}')
                else:
                    enc_parts_r.append(f'<code>{col_e}</code> → {str(strat_e).replace("_"," ")}')

            feat_short_r = {
                "log_transform":                    "log-transform applied to skewed columns to compress long tails",
                "create_interaction_terms":         "interaction terms created from top correlated feature pairs",
                "bin_skewed":                       "skewed columns binned into quantile buckets",
                "remove_high_correlation":          "near-duplicate features removed (correlation pruning)",
                "create_missing_indicator":         "missing-value indicator flags added",
            }
            feat_parts_r = []
            for fs in (feat_strats_r or []):
                label_r = next((v for k, v in feat_short_r.items() if k in fs), fs)
                feat_parts_r.append(label_r)

            if enc_parts_r or feat_parts_r:
                enc_body = (
                    "Categorical columns were encoded as: " + "; ".join(enc_parts_r) + "."
                    if enc_parts_r else "Standard encoding was applied."
                )
                feat_body = (
                    "Feature engineering included: " + "; ".join(feat_parts_r) + ". "
                    "These steps transform raw columns into richer signals the model can learn from."
                    if feat_parts_r else ""
                )
                st.markdown(_card(
                    "Encoding & feature engineering",
                    enc_body + ("<br><br>" + feat_body if feat_body else "")
                ), unsafe_allow_html=True)

            # ── Card 5: Training outcomes (tuning + threshold) ────────────────
            viz_data_r    = state_vals.get("visualization_data", {}) or {}
            tuning_r      = viz_data_r.get("tuning", {})
            threshold_r   = viz_data_r.get("threshold", {})

            outcome_parts = []
            if tuning_r:
                before_r = tuning_r.get("before", 0)
                after_r  = tuning_r.get("after", 0)
                delta_r  = after_r - before_r
                m_r      = tuning_r.get("metric", recommended_metric).upper()
                if delta_r > 0.02:
                    outcome_parts.append(f'Hyperparameter tuning (RandomizedSearchCV, 25 iterations) improved {m_r} from {before_r:.4f} to <strong>{after_r:.4f}</strong> — a meaningful gain of +{delta_r:.4f}.')
                elif delta_r > 0:
                    outcome_parts.append(f'Hyperparameter tuning gave a small improvement: {m_r} moved from {before_r:.4f} to {after_r:.4f} (+{delta_r:.4f}).')
                else:
                    outcome_parts.append(f'Hyperparameter tuning found no improvement — the default hyperparameters were already near-optimal ({m_r}: {before_r:.4f}).')

            if threshold_r:
                opt_r     = threshold_r.get("optimal", 0.5)
                def_score = threshold_r.get("metric_at_default", 0)
                opt_score = threshold_r.get("metric_at_optimal", 0)
                if opt_r < 0.45:
                    outcome_parts.append(f'The decision threshold was <strong>lowered to {opt_r:.2f}</strong> (from the default 0.5) to capture more positives — expected for imbalanced data. Score improved from {def_score:.4f} to {opt_score:.4f}.')
                elif opt_r > 0.55:
                    outcome_parts.append(f'The decision threshold was <strong>raised to {opt_r:.2f}</strong> to reduce false positives. Score improved from {def_score:.4f} to {opt_score:.4f}.')
                else:
                    outcome_parts.append(f'The decision threshold stayed near 0.5 ({opt_r:.2f}) — probability calibration was already well-aligned.')

            if outcome_parts:
                st.markdown(_card(
                    "Training outcomes",
                    "<br><br>".join(outcome_parts)
                ), unsafe_allow_html=True)

        else:
            st.markdown(
                empty_state("", "No reasoning context yet", "The profiler's decisions will appear here after the Profiler agent runs."),
                unsafe_allow_html=True,
            )


    with tab_results:
        state_vals = state_vals or {}
        viz_data = state_vals.get("visualization_data", {})

        _CHART_LAYOUT = dict(
            paper_bgcolor="#FFE8D6",
            plot_bgcolor="#FFE8D6",
            font=dict(family="Inter, sans-serif", size=12, color="#3B2314"),
            margin=dict(l=0, r=0, t=32, b=0),
        )
        # Axis style applied to every chart — makes tick labels reliably readable
        def _ax(**kw):
            return dict(tickfont=dict(color="#3B2314", size=10), title_font=dict(color="#3B2314", size=11), **kw)
        # 5-color palette: hue-varied, all high-contrast on #FFE8D6
        _CHART_COLORS = ["#3B2314", "#B45309", "#FF9F1C", "#2D7A4A", "#8B1A1A"]

        if viz_data:
            st.markdown(section_header("Model results"), unsafe_allow_html=True)
            _render_narration(state_vals.get("model_narration", ""))

            problem_type = state_vals.get("problem_type", "binary_classification")
            is_regression = "regression" in problem_type

            # Top level metrics
            if "best_model" in viz_data:
                bm = viz_data["best_model"]
                best_model_name = bm.get("name", "Unknown")

                if "model_comparison" in viz_data:
                    comp = viz_data["model_comparison"]
                    try:
                        def get_metric(comp, key, idx):
                            val = comp[key]
                            if isinstance(val, (list, tuple)):
                                return val[idx]
                            return val

                        model_names = comp["model_names"]
                        if not isinstance(model_names, list):
                            model_names = [model_names]
                        idx = model_names.index(best_model_name)
                        if is_regression:
                            r2_v   = bm.get("r2")
                            rmse_v = bm.get("rmse")
                            mae_v  = bm.get("mae")
                            st.markdown(
                                kpi_row([
                                    {"label": "Best model", "value": best_model_name, "pill": "winner", "pill_color": "blue"},
                                    {"label": "R²",   "value": f"{r2_v:.3f}"   if r2_v   is not None else "N/A"},
                                    {"label": "RMSE", "value": f"{rmse_v:.3f}" if rmse_v is not None else "N/A"},
                                    {"label": "MAE",  "value": f"{mae_v:.3f}"  if mae_v  is not None else "N/A"},
                                ]),
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                kpi_row([
                                    {"label": "Best model", "value": best_model_name, "pill": "winner", "pill_color": "blue"},
                                    {"label": "F1 score",   "value": f"{get_metric(comp, 'f1_score', idx):.3f}"},
                                    {"label": "AUC-ROC",    "value": f"{get_metric(comp, 'auc_roc', idx):.3f}"},
                                    {"label": "Accuracy",   "value": f"{get_metric(comp, 'accuracy', idx):.3f}"},
                                ]),
                                unsafe_allow_html=True,
                            )
                    except (ValueError, KeyError, IndexError):
                        st.markdown(kpi_row([{"label": "Best model", "value": best_model_name}]), unsafe_allow_html=True)

            # Why this model won
            if "best_model" in viz_data:
                bm = viz_data["best_model"]
                recommended_metric = state_vals.get("recommended_metric", "f1")
                imbalance_strategy = (state_vals.get("reasoning_context") or {}).get("imbalance_strategy", "none")
                scout_ranking_pre = state_vals.get("scout_ranking", [])
                best_name = bm.get("name", "")

                reasons = []
                if scout_ranking_pre:
                    ranked_names = [r[0] for r in scout_ranking_pre if r[1] > -999]
                    if best_name in ranked_names:
                        rank = ranked_names.index(best_name) + 1
                        scout_score = next((r[1] for r in scout_ranking_pre if r[0] == best_name), None)
                        reasons.append(f"Scout rank #{rank} on the 10% sample (score: {scout_score:.4f})" if scout_score else f"Scout rank #{rank}")
                metric_map = {
                    "recall": "optimised for recall (minimising false negatives)",
                    "f1": "optimised for F1 (balanced precision/recall)",
                    "precision": "optimised for precision (minimising false positives)",
                    "roc_auc": "optimised for AUC-ROC (ranking ability across all thresholds)",
                    "accuracy": "optimised for accuracy",
                    "rmse": "optimised for RMSE (regression)",
                    "r2": "optimised for R² (regression)",
                }
                reasons.append(metric_map.get(recommended_metric, f"optimised for {recommended_metric}"))
                if imbalance_strategy and imbalance_strategy != "none":
                    reasons.append(f"class imbalance handled via {imbalance_strategy}")
                family_hints = {
                    "LGBM": "gradient boosting — fast, handles mixed feature types well",
                    "XGB": "gradient boosting — strong on tabular data with interaction features",
                    "RandomForest": "bagging ensemble — robust to outliers, low tuning sensitivity",
                    "LogisticRegression": "linear baseline — interpretable, good for linearly separable data",
                    "Ridge": "regularised linear model — stable on small datasets",
                }
                family_hint = next((v for k, v in family_hints.items() if k.lower() in best_name.lower()), "")
                if family_hint:
                    reasons.append(family_hint)
                if reasons:
                    st.markdown(
                        banner(f"<strong>Why {best_name}?</strong> " + " · ".join(reasons) + ".", kind="info"),
                        unsafe_allow_html=True,
                    )

            st.markdown('<hr style="border:none;border-top:1px solid var(--border);margin:1rem 0">', unsafe_allow_html=True)

            # Scout ranking
            scout_ranking = state_vals.get("scout_ranking", [])
            if scout_ranking:
                st.markdown(section_header("Model scout — 10% sample ranking", "Stratified benchmark before full training"), unsafe_allow_html=True)
                names = [r[0] for r in scout_ranking if r[1] > -999]
                scores = [r[1] for r in scout_ranking if r[1] > -999]
                colors = ["#FF9F1C" if i < 2 else "rgba(59,35,20,0.22)" for i in range(len(names))]
                fig = go.Figure(go.Bar(
                    x=scores, y=names, orientation='h',
                    marker_color=colors,
                    text=[f"{s:.4f}" for s in scores],
                    textposition='outside',
                    textfont=dict(color="#3B2314", size=10),
                ))
                fig.update_layout(
                    xaxis_title="CV score (3-fold)",
                    height=max(250, 40 * len(names)),
                    xaxis=_ax(showgrid=True, gridcolor="rgba(59,35,20,0.15)"),
                    yaxis=_ax(showgrid=False),
                    **_CHART_LAYOUT,
                )
                st.plotly_chart(fig, width="stretch")

            # Model comparison chart
            if "model_comparison" in viz_data:
                st.markdown(section_header("Model comparison"), unsafe_allow_html=True)
                comparison = viz_data["model_comparison"]

                if is_regression:
                    col_r2, col_rmse = st.columns(2)
                    _mn = comparison.get("model_names", [])
                    if not isinstance(_mn, list): _mn = [_mn]
                    with col_r2:
                        r2_vals = comparison.get("r2", [])
                        if r2_vals:
                            if not isinstance(r2_vals, list): r2_vals = [r2_vals]
                            fig_r2 = go.Figure(go.Bar(
                                x=_mn, y=r2_vals,
                                marker_color=_CHART_COLORS[0],
                                text=[f"{v:.3f}" for v in r2_vals],
                                textposition='outside',
                                textfont=dict(color="#3B2314", size=10),
                            ))
                            _r2_min = min(r2_vals)
                            _r2_pad = max(0.02, (max(r2_vals) - _r2_min) * 0.4)
                            fig_r2.update_layout(
                                yaxis_title="R² (higher = better)",
                                yaxis=_ax(
                                    range=[max(0, _r2_min - _r2_pad), min(1, max(r2_vals) + _r2_pad)],
                                    showgrid=True, gridcolor="rgba(59,35,20,0.15)"
                                ),
                                xaxis=_ax(showgrid=False, tickangle=-35),
                                height=320, **_CHART_LAYOUT,
                            )
                            st.plotly_chart(fig_r2, width="stretch")
                    with col_rmse:
                        rmse_vals = comparison.get("rmse", [])
                        if rmse_vals:
                            if not isinstance(rmse_vals, list): rmse_vals = [rmse_vals]
                            fig_rm = go.Figure(go.Bar(
                                x=_mn, y=rmse_vals,
                                marker_color=_CHART_COLORS[1],
                                text=[f"{v:.3f}" for v in rmse_vals],
                                textposition='outside',
                                textfont=dict(color="#3B2314", size=10),
                            ))
                            fig_rm.update_layout(
                                yaxis_title="RMSE (lower = better)",
                                yaxis=_ax(showgrid=True, gridcolor="rgba(59,35,20,0.15)"),
                                xaxis=_ax(showgrid=False, tickangle=-35),
                                height=320, **_CHART_LAYOUT,
                            )
                            st.plotly_chart(fig_rm, width="stretch")
                else:
                    fig = go.Figure()
                    metrics = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
                    for metric, color in zip(metrics, _CHART_COLORS):
                        if metric in comparison:
                            y_val = comparison[metric]
                            if not isinstance(y_val, (list, tuple)):
                                y_val = [y_val]
                            x_val = comparison["model_names"]
                            if not isinstance(x_val, (list, tuple)):
                                x_val = [x_val]
                            fig.add_trace(go.Bar(
                                name=metric.replace("_", " ").title(),
                                x=x_val, y=y_val,
                                marker_color=color,
                            ))
                    fig.update_layout(
                        barmode='group',
                        yaxis_title="Score",
                        yaxis=_ax(range=[0, 1], showgrid=True, gridcolor="rgba(59,35,20,0.15)"),
                        xaxis=_ax(showgrid=False, tickangle=-35),
                        height=420,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                    font=dict(color="#3B2314", size=11)),
                        **_CHART_LAYOUT,
                    )
                    st.plotly_chart(fig, width="stretch")

            col_left, col_right = st.columns(2)

            with col_left:
                if is_regression:
                    bm_reg = viz_data.get("best_model", {})
                    actuals = bm_reg.get("actuals")
                    preds   = bm_reg.get("predictions")
                    if actuals and preds:
                        st.markdown(section_header(f"Actual vs Predicted — {bm_reg.get('name', '')}"), unsafe_allow_html=True)
                        lo = min(min(actuals), min(preds))
                        hi = max(max(actuals), max(preds))
                        fig_avp = go.Figure()
                        fig_avp.add_trace(go.Scatter(
                            x=actuals, y=preds, mode='markers',
                            marker=dict(color="#B45309", size=5, opacity=0.6),
                            name="Predictions",
                        ))
                        fig_avp.add_trace(go.Scatter(
                            x=[lo, hi], y=[lo, hi], mode='lines',
                            line=dict(color="#3B2314", dash="dash", width=1),
                            name="Perfect fit",
                        ))
                        fig_avp.update_layout(
                            xaxis_title="Actual",
                            yaxis_title="Predicted",
                            xaxis=_ax(showgrid=True, gridcolor="rgba(59,35,20,0.15)"),
                            yaxis=_ax(showgrid=True, gridcolor="rgba(59,35,20,0.15)"),
                            legend=dict(font=dict(color="#3B2314", size=10)),
                            height=380, **_CHART_LAYOUT,
                        )
                        st.plotly_chart(fig_avp, width="stretch")
                    else:
                        # Fallback: show metric tile when actuals/predictions not saved
                        r2_v   = bm_reg.get("r2")
                        rmse_v = bm_reg.get("rmse")
                        mae_v  = bm_reg.get("mae")
                        if r2_v is not None:
                            st.markdown(section_header(f"Test set metrics — {bm_reg.get('name', '')}"), unsafe_allow_html=True)
                            pill_color = "green" if r2_v > 0.6 else ("amber" if r2_v > 0.3 else "red")
                            pill_label = "Good" if r2_v > 0.6 else ("OK" if r2_v > 0.3 else "Low")
                            st.markdown(
                                kpi_row([
                                    {"label": "R²",   "value": f"{r2_v:.4f}",   "pill": pill_label, "pill_color": pill_color},
                                    {"label": "RMSE", "value": f"{rmse_v:.4f}" if rmse_v is not None else "N/A"},
                                    {"label": "MAE",  "value": f"{mae_v:.4f}"  if mae_v  is not None else "N/A"},
                                ]),
                                unsafe_allow_html=True,
                            )
                else:
                    if "best_model" in viz_data:
                        cm = viz_data["best_model"].get("confusion_matrix")
                        if cm:
                            n_classes = len(cm)
                            target = state_vals.get("target_column", "target")
                            if n_classes == 2:
                                labels = [f"Not {target}", target]
                            else:
                                labels = [f"Class {i}" for i in range(n_classes)]
                            st.markdown(section_header(f"Confusion matrix — {viz_data['best_model']['name']}"), unsafe_allow_html=True)
                            fig_cm = ff.create_annotated_heatmap(
                                z=cm, x=labels, y=labels,
                                colorscale="Blues", showscale=True,
                                font_colors=["#3B2314", "#FFE8D6"],
                            )
                            fig_cm.update_layout(
                                xaxis=_ax(title="Predicted"),
                                yaxis=_ax(title="Actual"),
                                height=max(340, 80 * n_classes),
                                **_CHART_LAYOUT,
                            )
                            fig_cm.update_traces(colorbar=dict(tickfont=dict(color="#3B2314")))
                            st.plotly_chart(fig_cm, width="stretch")

            with col_right:
                if "cross_validation" in viz_data:
                    cv = viz_data["cross_validation"]
                    scores = cv.get("cv_scores", [])
                    if scores:
                        mean_val = cv.get("mean", sum(scores)/len(scores))
                        std_val = cv.get("std", 0)
                        if is_regression:
                            # cv_scores are neg_rmse from sklearn — flip to positive for display
                            display_scores = [abs(s) for s in scores]
                            y_min = max(0, min(display_scores) * 0.9)
                            y_max = max(display_scores) * 1.1
                            y_title = "RMSE (lower = better)"
                        else:
                            display_scores = scores
                            y_min = max(0, mean_val - 0.2)
                            y_max = min(1.0, mean_val + 0.2)
                            y_title = "Score"
                        st.markdown(section_header("Cross-validation consistency"), unsafe_allow_html=True)
                        fig_cv = go.Figure()
                        fig_cv.add_trace(go.Bar(
                            x=[f"Fold {i+1}" for i in range(len(display_scores))],
                            y=display_scores,
                            marker_color="#B45309",
                            name="Fold score",
                        ))
                        fig_cv.add_hline(
                            y=mean_val, line_dash="dash", line_color="#3B2314",
                            annotation_text=f"Mean {mean_val:.3f} ± {std_val:.3f}",
                            annotation_position="top right",
                            annotation_font=dict(color="#3B2314"),
                        )
                        fig_cv.update_layout(
                            yaxis_title=y_title,
                            yaxis=_ax(range=[y_min, y_max], showgrid=True, gridcolor="rgba(59,35,20,0.15)"),
                            xaxis=_ax(showgrid=False),
                            height=380,
                            **_CHART_LAYOUT,
                        )
                        st.plotly_chart(fig_cv, width="stretch")

            # ── Residuals plot (regression only, full-width) ──────────────────
            if is_regression:
                bm_reg = viz_data.get("best_model", {})
                actuals_res = bm_reg.get("actuals")
                preds_res   = bm_reg.get("predictions")
                if actuals_res and preds_res:
                    residuals = [float(p) - float(a) for p, a in zip(preds_res, actuals_res)]
                    st.markdown('<hr style="border:none;border-top:1px solid var(--border);margin:1rem 0">', unsafe_allow_html=True)
                    st.markdown(section_header("Residuals plot", "Predicted − Actual vs Predicted — should scatter randomly around zero"), unsafe_allow_html=True)
                    fig_res = go.Figure()
                    fig_res.add_trace(go.Scatter(
                        x=list(preds_res), y=residuals, mode='markers',
                        marker=dict(color="#B45309", size=5, opacity=0.5),
                        name="Residuals",
                    ))
                    fig_res.add_hline(y=0, line_dash="dash", line_color="#3B2314", line_width=1.5)
                    fig_res.update_layout(
                        xaxis_title="Predicted value",
                        yaxis_title="Residual (Predicted − Actual)",
                        xaxis=_ax(showgrid=True, gridcolor="rgba(59,35,20,0.15)"),
                        yaxis=_ax(showgrid=True, gridcolor="rgba(59,35,20,0.15)",
                                  zeroline=True, zerolinecolor="#3B2314", zerolinewidth=1),
                        height=360, **_CHART_LAYOUT,
                    )
                    st.plotly_chart(fig_res, width="stretch")
                    mean_res = sum(residuals) / len(residuals) if residuals else 0
                    max_abs  = max(abs(r) for r in residuals) if residuals else 1
                    if abs(mean_res) < 0.05 * max_abs:
                        st.markdown(banner("Residuals centered near zero — no systematic bias detected.", kind="success"), unsafe_allow_html=True)
                    else:
                        st.markdown(banner(f"Mean residual: {mean_res:.3f} — the model may be systematically over- or under-predicting.", kind="warning"), unsafe_allow_html=True)

            # ── Precision-Recall curve (binary classification, imbalanced data) ─
            if not is_regression and "best_model" in viz_data:
                pr_curve = viz_data["best_model"].get("pr_curve", {})
                if pr_curve and pr_curve.get("precision") and pr_curve.get("recall"):
                    avg_prec = pr_curve.get("avg_precision")
                    ap_str   = f" — Average Precision = {avg_prec:.3f}" if avg_prec is not None else ""
                    st.markdown('<hr style="border:none;border-top:1px solid var(--border);margin:1rem 0">', unsafe_allow_html=True)
                    st.markdown(section_header(f"Precision-Recall curve{ap_str}", "Better than ROC for imbalanced datasets — top-right corner = best"), unsafe_allow_html=True)
                    fig_pr = go.Figure()
                    # Shaded area under PR curve
                    fig_pr.add_trace(go.Scatter(
                        x=pr_curve["recall"], y=pr_curve["precision"],
                        mode='lines', line=dict(color="#B45309", width=2.5),
                        name=f"PR curve (AP={avg_prec:.3f})" if avg_prec is not None else "PR curve",
                        fill='tozeroy', fillcolor='rgba(180,83,9,0.12)',
                    ))
                    # Baseline = positive class rate
                    try:
                        baseline = float((state_vals.get("reasoning_context") or {}).get("imbalance_ratio", 0.5))
                    except (TypeError, ValueError):
                        baseline = 0.5
                    fig_pr.add_hline(
                        y=baseline, line_dash="dot", line_color="#3B2314",
                        annotation_text=f"Baseline ({baseline:.2f})",
                        annotation_font=dict(color="#3B2314"),
                    )
                    fig_pr.update_layout(
                        xaxis_title="Recall", yaxis_title="Precision",
                        xaxis=_ax(range=[0, 1], showgrid=True, gridcolor="rgba(59,35,20,0.15)"),
                        yaxis=_ax(range=[0, 1.05], showgrid=True, gridcolor="rgba(59,35,20,0.15)"),
                        legend=dict(font=dict(color="#3B2314", size=11)),
                        height=380, **_CHART_LAYOUT,
                    )
                    st.plotly_chart(fig_pr, width="stretch")
                    if avg_prec is not None:
                        if avg_prec > 0.7:
                            st.markdown(banner(f"AP = {avg_prec:.3f} — strong precision-recall trade-off. Model reliably identifies positives.", kind="success"), unsafe_allow_html=True)
                        elif avg_prec > 0.4:
                            st.markdown(banner(f"AP = {avg_prec:.3f} — moderate. Some precision lost at high recall — consider threshold adjustment.", kind="info"), unsafe_allow_html=True)
                        else:
                            st.markdown(banner(f"AP = {avg_prec:.3f} — low. Model struggles with class separation. Review features and imbalance strategy.", kind="warning"), unsafe_allow_html=True)

            # ── Learning curves (bias/variance diagnostic) ────────────────────
            lc_data = viz_data.get("learning_curve", {})
            if lc_data and lc_data.get("train_sizes") and lc_data.get("train_scores_mean"):
                train_sizes  = lc_data["train_sizes"]
                train_mean   = lc_data["train_scores_mean"]
                val_mean     = lc_data["val_scores_mean"]
                train_std    = lc_data.get("train_scores_std", [0] * len(train_mean))
                val_std      = lc_data.get("val_scores_std",   [0] * len(val_mean))
                st.markdown('<hr style="border:none;border-top:1px solid var(--border);margin:1rem 0">', unsafe_allow_html=True)
                st.markdown(section_header("Learning curves", "Training vs validation score as data grows — bias/variance diagnostic"), unsafe_allow_html=True)
                fig_lc = go.Figure()
                # Shaded uncertainty bands
                fig_lc.add_trace(go.Scatter(
                    x=train_sizes + train_sizes[::-1],
                    y=[m + s for m, s in zip(train_mean, train_std)] + [m - s for m, s in zip(train_mean[::-1], train_std[::-1])],
                    fill='toself', fillcolor='rgba(59,35,20,0.08)', line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False,
                ))
                fig_lc.add_trace(go.Scatter(
                    x=train_sizes + train_sizes[::-1],
                    y=[m + s for m, s in zip(val_mean, val_std)] + [m - s for m, s in zip(val_mean[::-1], val_std[::-1])],
                    fill='toself', fillcolor='rgba(180,83,9,0.12)', line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False,
                ))
                # Score lines
                fig_lc.add_trace(go.Scatter(
                    x=train_sizes, y=train_mean, mode='lines+markers',
                    line=dict(color="#3B2314", width=2), marker=dict(size=6),
                    name="Training score",
                ))
                fig_lc.add_trace(go.Scatter(
                    x=train_sizes, y=val_mean, mode='lines+markers',
                    line=dict(color="#B45309", width=2), marker=dict(size=6),
                    name="Validation score",
                ))
                fig_lc.update_layout(
                    xaxis_title="Training set size",
                    yaxis_title="Score",
                    xaxis=_ax(showgrid=True, gridcolor="rgba(59,35,20,0.15)"),
                    yaxis=_ax(showgrid=True, gridcolor="rgba(59,35,20,0.15)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                font=dict(color="#3B2314", size=11)),
                    height=380, **_CHART_LAYOUT,
                )
                st.plotly_chart(fig_lc, width="stretch")
                # Bias/variance interpretation
                final_train = train_mean[-1] if train_mean else 0
                final_val   = val_mean[-1]   if val_mean   else 0
                gap = final_train - final_val
                if gap > 0.15:
                    st.markdown(banner(
                        f"High variance (overfitting) — training ({final_train:.3f}) >> validation ({final_val:.3f}), gap = {gap:.3f}. "
                        "Try regularisation, feature selection, or collecting more data.", kind="warning"),
                        unsafe_allow_html=True,
                    )
                elif final_val < 0.6 and gap < 0.05:
                    st.markdown(banner(
                        f"High bias (underfitting) — both scores are low (validation = {final_val:.3f}). "
                        "Add more predictive features, try a more complex model, or reduce regularisation.", kind="warning"),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(banner(
                        f"Balanced bias-variance — training ({final_train:.3f}) and validation ({final_val:.3f}) are close. "
                        "Model generalises well.", kind="success"),
                        unsafe_allow_html=True,
                    )

            # Hyperparameter tuning
            tuning = viz_data.get("tuning", {})
            if tuning and tuning.get("delta"):
                st.markdown('<hr style="border:none;border-top:1px solid var(--border);margin:1rem 0">', unsafe_allow_html=True)
                st.markdown(section_header("Hyperparameter tuning", "RandomizedSearchCV — 25 iterations"), unsafe_allow_html=True)
                metric_name = tuning.get("metric", "score").upper()
                before = tuning.get("before", 0)
                after = tuning.get("after", 0)
                delta = tuning.get("delta", 0)
                st.markdown(
                    kpi_row([
                        {"label": f"Default {metric_name}", "value": f"{before:.4f}"},
                        {"label": f"Tuned {metric_name}",   "value": f"{after:.4f}", "pill": f"+{delta:.4f}", "pill_color": "green" if delta > 0 else "gray"},
                        {"label": "Improvement",            "value": f"{delta*100:.2f}%"},
                    ]),
                    unsafe_allow_html=True,
                )
                if delta > 0.005:
                    st.markdown(banner(f"RandomizedSearchCV improved {metric_name} by {delta*100:.1f}% — a meaningful gain for this dataset.", kind="success"), unsafe_allow_html=True)
                else:
                    st.markdown(banner("Minimal improvement — default hyperparameters were already well-calibrated.", kind="info"), unsafe_allow_html=True)

            # Threshold optimisation (classification only)
            threshold = viz_data.get("threshold", {})
            if not is_regression and threshold and threshold.get("optimal"):
                st.markdown('<hr style="border:none;border-top:1px solid var(--border);margin:1rem 0">', unsafe_allow_html=True)
                st.markdown(section_header("Decision threshold optimisation"), unsafe_allow_html=True)
                optimal = threshold.get("optimal", 0.5)
                default_score = threshold.get("metric_at_default", 0)
                optimal_score = threshold.get("metric_at_optimal", 0)
                gain = optimal_score - default_score
                st.markdown(
                    kpi_row([
                        {"label": "Score at default (0.5)",        "value": f"{default_score:.4f}"},
                        {"label": f"Score at optimal ({optimal:.2f})", "value": f"{optimal_score:.4f}", "pill": f"+{gain:.4f}", "pill_color": "green" if gain > 0 else "gray"},
                        {"label": "Optimal threshold",             "value": f"{optimal:.3f}"},
                    ]),
                    unsafe_allow_html=True,
                )
                if optimal < 0.45:
                    st.markdown(banner("Threshold below 0.5 — lower cutoff needed to capture more positives. Expected for imbalanced data.", kind="info"), unsafe_allow_html=True)
                elif optimal > 0.55:
                    st.markdown(banner("Threshold above 0.5 — conservative model; raising the bar reduces false positives.", kind="info"), unsafe_allow_html=True)
                else:
                    st.markdown(banner("Threshold near 0.5 — probability calibration is already well-aligned.", kind="info"), unsafe_allow_html=True)

            # ── Business impact calculator ─────────────────────────────────────
            if not is_regression and "best_model" in viz_data:
                _pr = viz_data["best_model"].get("pr_curve", {})
                _cm = viz_data["best_model"].get("confusion_matrix")
                if _pr and _pr.get("precision") and _pr.get("recall") and _cm:
                    st.markdown('<hr style="border:none;border-top:1px solid var(--border);margin:1rem 0">', unsafe_allow_html=True)
                    st.markdown(section_header("Business impact calculator", "Translate sensitivity into real-world outcomes for any population size"), unsafe_allow_html=True)

                    _target_name = state_vals.get("target_column", "positive case")

                    # Estimate prevalence from confusion matrix
                    try:
                        _test_pos = _cm[1][0] + _cm[1][1]
                        _test_total = _cm[0][0] + _cm[0][1] + _cm[1][0] + _cm[1][1]
                        _prevalence = _test_pos / _test_total if _test_total > 0 else 0.1
                        _opt_recall = _cm[1][1] / _test_pos if _test_pos > 0 else 0.7
                    except (IndexError, TypeError, ZeroDivisionError):
                        _prevalence = 0.1
                        _opt_recall = 0.7

                    _col_pop, _col_sens = st.columns([1, 2])
                    with _col_pop:
                        _population = st.number_input(
                            "Population size", min_value=100, max_value=10_000_000,
                            value=10_000, step=500, key="biz_population",
                            help="Number of people you want to screen",
                        )
                    with _col_sens:
                        _recall_target = st.slider(
                            "Sensitivity (recall) target", min_value=0.0, max_value=1.0,
                            value=float(min(0.99, max(0.01, _opt_recall))),
                            step=0.01, key="biz_recall",
                            help="Higher = catch more positives but generate more false alarms",
                        )

                    # Interpolate precision at chosen recall from PR curve
                    _pr_pairs = sorted(zip(_pr["recall"], _pr["precision"]), key=lambda x: x[0])
                    _prec = None
                    for _i in range(len(_pr_pairs) - 1):
                        _r0, _p0 = _pr_pairs[_i]
                        _r1, _p1 = _pr_pairs[_i + 1]
                        if _r0 <= _recall_target <= _r1:
                            _prec = _p0 if _r1 == _r0 else _p0 + (_p1 - _p0) * (_recall_target - _r0) / (_r1 - _r0)
                            break
                    if _prec is None:
                        _prec = min(_pr_pairs, key=lambda x: abs(x[0] - _recall_target))[1]
                    _prec = max(0.001, min(1.0, _prec))

                    # Real-world impact numbers
                    _tot_pos = _population * _prevalence
                    _tot_neg = _population * (1 - _prevalence)
                    _tp = round(_recall_target * _tot_pos)
                    _fn = round(_tot_pos) - _tp
                    _fp = round(_tp * (1 - _prec) / _prec)
                    _fp = max(0, min(_fp, round(_tot_neg)))
                    _tn = max(0, round(_tot_neg) - _fp)

                    st.markdown(
                        kpi_row([
                            {"label": f"True {_target_name}s caught",     "value": f"{_tp:,}",  "pill": "caught",     "pill_color": "green"},
                            {"label": f"Missed {_target_name}s",           "value": f"{_fn:,}",  "pill": "missed",     "pill_color": "red"},
                            {"label": "False alarms",                       "value": f"{_fp:,}",  "pill": "false alarm","pill_color": "amber"},
                            {"label": "Correctly cleared",                  "value": f"{_tn:,}",  "pill": "cleared",    "pill_color": "gray"},
                        ]),
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        banner(
                            f'<strong>At {_recall_target*100:.0f}% sensitivity:</strong> Out of '
                            f'<strong>{_population:,}</strong> people screened, you flag '
                            f'<strong>{_tp + _fp:,}</strong> as positive — catching '
                            f'<strong>{_tp:,} of {round(_tot_pos):,} true {_target_name}s</strong> '
                            f'with <strong>{_fp:,} false alarms</strong>. '
                            f'Of every 100 flagged, <strong>{_prec*100:.0f} are real {_target_name}s</strong>.',
                            kind="info",
                        ),
                        unsafe_allow_html=True,
                    )
                    if _recall_target > 0.9:
                        st.markdown(banner(f"High sensitivity — catching most {_target_name}s, but expect more false alarms. Ideal for high-stakes screening where missing a case is costly.", kind="info"), unsafe_allow_html=True)
                    elif _recall_target < 0.5:
                        st.markdown(banner(f"Low sensitivity — fewer false alarms, but missing many {_target_name}s. Best when precision matters more than coverage.", kind="warning"), unsafe_allow_html=True)
                    else:
                        st.markdown(banner(f"Balanced — moderate trade-off between {_target_name} detection and false alarm rate.", kind="info"), unsafe_allow_html=True)

            # Feature importance
            if "best_model" in viz_data:
                fi = viz_data["best_model"].get("feature_importance")
                if fi:
                    st.markdown('<hr style="border:none;border-top:1px solid var(--border);margin:1rem 0">', unsafe_allow_html=True)
                    st.markdown(section_header("Top feature importances"), unsafe_allow_html=True)
                    def _clean_feat(n):
                        # Strip sklearn Pipeline prefixes like num__, cat__, remainder__
                        import re as _re
                        return _re.sub(r'^[a-z]+__', '', str(n))
                    names = [_clean_feat(n) for n in fi["feature_names"][:15]]
                    values = fi["importance_values"][:15]
                    names_sorted = names[::-1]
                    values_sorted = values[::-1]
                    fig_fi = go.Figure(go.Bar(
                        x=values_sorted, y=names_sorted,
                        orientation='h',
                        marker=dict(color=values_sorted, colorscale=[[0, "#E8C08A"], [1, "#7A3800"]], showscale=False),
                    ))
                    fig_fi.update_layout(
                        xaxis_title="Importance score",
                        xaxis=_ax(showgrid=True, gridcolor="rgba(59,35,20,0.15)"),
                        yaxis=_ax(showgrid=False),
                        height=max(300, 30 * len(names_sorted)),
                        **_CHART_LAYOUT,
                    )
                    st.plotly_chart(fig_fi, width="stretch")

            # Iteration comparison
            iteration_history = state_vals.get("iteration_history", [])
            if len(iteration_history) > 1:
                st.markdown('<hr style="border:none;border-top:1px solid var(--border);margin:1rem 0">', unsafe_allow_html=True)
                st.markdown(section_header("Iteration comparison", "Each iteration = one Critic → Feature Engineer → Modeler loop"), unsafe_allow_html=True)

                desired_cats = ["data_leakage", "code_quality", "metric_alignment",
                                "feature_engineering", "model_selection", "deployment_readiness"]
                blues_iter = _CHART_COLORS
                fig_iter = go.Figure()
                cat_labels_iter = None
                for entry in iteration_history:
                    sc = entry.get("scorecard", {})
                    if not sc:
                        continue
                    cats = [c for c in desired_cats if c in sc]
                    vals = [sc[c] for c in cats]
                    cat_labels_iter = [c.replace("_", " ").title() for c in cats]
                    color = blues_iter[entry["iteration"] % len(blues_iter)]
                    fig_iter.add_trace(go.Bar(
                        name=f"Iteration {entry['iteration']}",
                        x=cat_labels_iter, y=vals,
                        marker_color=color,
                    ))

                if fig_iter.data:
                    fig_iter.update_layout(
                        barmode='group',
                        yaxis=_ax(range=[0, 10], title="Score (/10)", showgrid=True, gridcolor="rgba(59,35,20,0.15)"),
                        xaxis=_ax(showgrid=False, tickangle=-25),
                        height=320,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                    font=dict(color="#3B2314", size=11)),
                        **_CHART_LAYOUT,
                    )
                    st.plotly_chart(fig_iter, width="stretch")

                metric_rows = []
                for entry in iteration_history:
                    viz = entry.get("viz_snapshot", {})
                    bm = viz.get("best_model", {})
                    sc = entry.get("scorecard", {})
                    if is_regression:
                        metric_rows.append({
                            "Iteration": entry["iteration"],
                            "Overall Score": sc.get("overall", "N/A"),
                            "Severity": entry.get("severity", "N/A").title(),
                            "Fixes": entry.get("n_fixes", 0),
                            "R²":   round(bm.get("r2",   0), 4) if bm.get("r2")   is not None else "N/A",
                            "RMSE": round(bm.get("rmse", 0), 4) if bm.get("rmse") is not None else "N/A",
                            "MAE":  round(bm.get("mae",  0), 4) if bm.get("mae")  is not None else "N/A",
                            "Best Model": bm.get("name", "N/A"),
                        })
                    else:
                        metric_rows.append({
                            "Iteration": entry["iteration"],
                            "Overall Score": sc.get("overall", "N/A"),
                            "Severity": entry.get("severity", "N/A").title(),
                            "Fixes": entry.get("n_fixes", 0),
                            "F1": round(bm.get("test_f1", 0), 3) if bm.get("test_f1") else "N/A",
                            "Recall": round(bm.get("test_recall", 0), 3) if bm.get("test_recall") else "N/A",
                            "AUC-ROC": round(bm.get("test_roc_auc", 0), 3) if bm.get("test_roc_auc") else "N/A",
                            "Best Model": bm.get("name", "N/A"),
                        })
                import pandas as pd
                st.dataframe(pd.DataFrame(metric_rows), width="stretch", hide_index=True)

            with st.expander("Raw training output & code"):
                st.text(state_vals.get("model_result", ""))
                st.code(state_vals.get("model_code", "No code"), language="python")

            model_code = state_vals.get("model_code", "")
            if model_code:
                st.download_button(
                    "Download model training code",
                    data=model_code,
                    file_name="model_training_code.py",
                    mime="text/x-python",
                )
        else:
            model_result = state_vals.get("model_result", "")
            if model_result:
                st.markdown(section_header("Model training output"), unsafe_allow_html=True)
                st.code(model_result, language="text")
                with st.expander("Model training code"):
                    st.code(state_vals.get("model_code", "No code"), language="python")
            else:
                st.markdown(
                    empty_state("", "No model results yet", "Model results and charts will appear here after the Modeler agent runs."),
                    unsafe_allow_html=True,
                )


    with tab_critique:
        state_vals = state_vals or {}
        critique = state_vals.get("critique_report", "")
        scorecard = state_vals.get("scorecard", {})

        if critique or scorecard:
            st.markdown(section_header("Critique & scorecard"), unsafe_allow_html=True)

            if scorecard:
                categories = list(scorecard.keys())
                if "overall" in categories:
                    categories.remove("overall")
                desired_order = ["data_leakage", "code_quality", "metric_alignment",
                                 "feature_engineering", "model_selection", "deployment_readiness"]
                ordered_cats = [c for c in desired_order if c in categories]
                for c in categories:
                    if c not in ordered_cats:
                        ordered_cats.append(c)
                values = [scorecard.get(c, 0) for c in ordered_cats]
                labels = [c.replace("_", " ").title() for c in ordered_cats]
                overall = scorecard.get("overall", "N/A")

                # Overall score KPI + verdict banner
                try:
                    ov = float(overall)
                    if ov >= 8.5:
                        verdict_kind, verdict_text = "success", "Production-ready — minor polish only."
                    elif ov >= 7.0:
                        verdict_kind, verdict_text = "info", "Good pipeline — another iteration may improve weaker dimensions."
                    elif ov >= 5.0:
                        verdict_kind, verdict_text = "warning", "Functional but has real issues — review the fixes before deploying."
                    else:
                        verdict_kind, verdict_text = "error", "Significant problems detected — do not deploy without addressing all fixes."
                except (ValueError, TypeError):
                    verdict_kind, verdict_text = "info", ""

                c_score, c_verdict = st.columns([1, 3])
                with c_score:
                    st.markdown(
                        f'<div class="ds-kpi" style="text-align:center">'
                        f'<div class="ds-kpi-label">Overall score</div>'
                        f'<div class="ds-kpi-value" style="font-size:2.5rem">{overall}</div>'
                        f'<div style="font-size:0.75rem;color:var(--text-muted)">/10</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with c_verdict:
                    if verdict_text:
                        st.markdown(banner(verdict_text, kind=verdict_kind), unsafe_allow_html=True)

                # Horizontal bar scorecard (replacing radar)
                if values and labels:
                    _score_desc = {
                        "Data Leakage": "Does the pipeline accidentally use future/target info during training?",
                        "Code Quality": "Is the generated code robust, with proper error handling and clean logic?",
                        "Metric Alignment": "Is the chosen metric right for the problem?",
                        "Feature Engineering": "Are features informative, non-redundant, and leakage-free?",
                        "Model Selection": "Was the best model family chosen and properly tuned?",
                        "Deployment Readiness": "Are model artifacts and API code production-complete?",
                    }
                    bar_colors = ["#2D7A4A" if v >= 7 else ("#B87800" if v >= 5 else "#C62828") for v in values]
                    fig_sc = go.Figure(go.Bar(
                        x=values,
                        y=labels,
                        orientation='h',
                        marker_color=bar_colors,
                        text=[f"{v}/10" for v in values],
                        textposition='outside',
                    ))
                    fig_sc.update_layout(
                        xaxis=_ax(range=[0, 10], showgrid=True, gridcolor="rgba(59,35,20,0.15)", tickvals=[0,2,4,6,8,10]),
                        yaxis=_ax(showgrid=False),
                        height=max(220, 42 * len(labels)),
                        paper_bgcolor="#FFE8D6",
                        plot_bgcolor="#FFE8D6",
                        font=dict(family="Inter, sans-serif", size=12, color="#3B2314"),
                        margin=dict(l=0, r=60, t=8, b=0),
                    )
                    st.plotly_chart(fig_sc, width="stretch")

                    st.markdown(section_header("Dimension breakdown"), unsafe_allow_html=True)
                    for cat, val in zip(labels, values):
                        desc = _score_desc.get(cat, "")
                        color = "var(--success)" if val >= 7 else ("var(--warn)" if val >= 5 else "var(--error)")
                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:0.75rem;padding:0.4rem 0;border-bottom:1px solid var(--border)">'
                            f'<span style="font-family:var(--font-mono);font-size:0.85rem;font-weight:700;color:{color};min-width:2.5rem">{val}/10</span>'
                            f'<div><div style="font-size:0.85rem;font-weight:600;color:var(--text-pri)">{cat}</div>'
                            f'<div style="font-size:0.78rem;color:var(--text-muted)">{desc}</div></div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            st.markdown('<hr style="border:none;border-top:1px solid var(--border);margin:1rem 0">', unsafe_allow_html=True)
            st.markdown(section_header("Critique report"), unsafe_allow_html=True)
            st.markdown(critique)

            suggestions = state_vals.get("improvement_suggestions", [])
            if suggestions:
                st.markdown('<hr style="border:none;border-top:1px solid var(--border);margin:1rem 0">', unsafe_allow_html=True)
                st.markdown(section_header("Improvement suggestions"), unsafe_allow_html=True)
                rows_html = "".join(
                    f'<div style="display:flex;gap:0.6rem;padding:0.4rem 0;border-bottom:1px solid var(--border)">'
                    f'<span style="font-size:0.78rem;font-weight:700;color:var(--accent-dark);min-width:1.5rem">{i}.</span>'
                    f'<span style="font-size:0.85rem;color:var(--text-sec)">{s}</span>'
                    f'</div>'
                    for i, s in enumerate(suggestions, 1)
                )
                st.markdown(
                    f'<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-card);padding:0.5rem 1rem">{rows_html}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<p style="font-size:0.82rem;color:var(--text-muted);margin-top:0.75rem">Iterations completed: <strong>{state_vals.get("iteration_count", 0)}</strong></p>',
                unsafe_allow_html=True,
            )

            history = state_vals.get("iteration_history", [])
            if history:
                st.markdown('<hr style="border:none;border-top:1px solid var(--border);margin:1rem 0">', unsafe_allow_html=True)
                st.markdown(section_header("Iteration history"), unsafe_allow_html=True)
                for entry in history:
                    sev = entry.get("severity", "N/A")
                    with st.expander(f"Iteration {entry['iteration']} — {entry.get('n_fixes', 0)} fixes · severity: {sev}"):
                        if entry.get("suggestions"):
                            for s in entry["suggestions"]:
                                st.markdown(f'<div style="font-size:0.85rem;padding:0.2rem 0;color:var(--text-sec)">· {s}</div>', unsafe_allow_html=True)
                        if entry.get("scorecard"):
                            sc_e = entry["scorecard"]
                            kpi_items = [{"label": k.replace("_"," ").title(), "value": f"{v}/10"} for k, v in sc_e.items() if k != "overall" and isinstance(v, (int, float))]
                            if kpi_items:
                                st.markdown(kpi_row(kpi_items[:4]), unsafe_allow_html=True)

            if critique:
                st.download_button(
                    "Download critique report",
                    data=critique,
                    file_name="critique_report.md",
                    mime="text/markdown",
                )
        else:
            st.markdown(
                empty_state("", "No critique yet", "The critique report and scorecard will appear here after the Critic agent runs."),
                unsafe_allow_html=True,
            )

    with tab_chat:
        from src.ui.components.chat_panel import render_chat_panel
        render_chat_panel(state_vals)
