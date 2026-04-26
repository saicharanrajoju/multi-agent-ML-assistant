import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage
from src.llm_helper import call_llm_with_fallback


# ── Context builder ───────────────────────────────────────────────────────────

def build_chat_context(state_vals: dict) -> str:
    """Assemble a grounded factual context string from the full pipeline run."""
    if not state_vals:
        return "No pipeline analysis has been run yet."

    parts = []

    # ── 1. Dataset & problem setup ────────────────────────────────────────────
    problem_type  = state_vals.get("problem_type", "unknown")
    target_col    = state_vals.get("target_column", "unknown")
    rec_metric    = state_vals.get("recommended_metric", "unknown")
    user_goal     = state_vals.get("user_goal", "")
    rc            = state_vals.get("reasoning_context", {}) or {}
    ds            = state_vals.get("dataset_summary", {}) or {}

    n_rows     = rc.get("n_rows") or ds.get("shape", [None])[0]
    n_cols     = rc.get("n_cols") or (ds.get("shape", [None, None])[1] if len(ds.get("shape", [])) > 1 else None)
    imb_ratio  = rc.get("imbalance_ratio", "N/A")
    imb_strat  = rc.get("imbalance_strategy", "none")

    parts.append("=== DATASET & PROBLEM ===")
    if user_goal:
        parts.append(f"User goal: {user_goal}")
    parts.append(f"Problem type: {problem_type}")
    parts.append(f"Target column: {target_col}")
    parts.append(f"Optimisation metric: {rec_metric}")
    if n_rows:
        parts.append(f"Dataset size: {n_rows:,} rows × {n_cols} columns" if n_cols else f"Dataset size: {n_rows:,} rows")
    parts.append(f"Class imbalance ratio: {imb_ratio}")
    parts.append(f"Imbalance strategy: {imb_strat}")

    # ── 2. Profiler reasoning ─────────────────────────────────────────────────
    rec_models      = rc.get("recommended_models", [])
    feat_strats     = rc.get("feature_strategies", [])
    enc_map         = rc.get("encoding_map", {})
    null_patterns   = rc.get("null_patterns", {})
    id_cols         = rc.get("id_columns", [])
    const_cols      = rc.get("constant_columns", [])
    datetime_cols   = rc.get("datetime_columns", [])
    text_cols       = rc.get("text_columns", [])
    skewed_cols     = ds.get("skewed_columns", [])
    top_corr        = ds.get("correlations_with_target", {})

    parts.append("\n=== PROFILER DECISIONS ===")
    if rec_models:
        parts.append(f"Models tried: {', '.join(rec_models)}")
    if feat_strats:
        parts.append(f"Feature strategies: {', '.join(feat_strats)}")
    if enc_map:
        enc_summary = "; ".join(
            f"{col}: {v['type'] if isinstance(v, dict) else v}" for col, v in list(enc_map.items())[:10]
        )
        parts.append(f"Encoding decisions: {enc_summary}")
    if id_cols:
        parts.append(f"ID columns dropped: {', '.join(id_cols)}")
    if const_cols:
        parts.append(f"Constant columns dropped: {', '.join(const_cols)}")
    if datetime_cols:
        parts.append(f"Datetime columns (temporal features extracted): {', '.join(datetime_cols)}")
    if text_cols:
        parts.append(f"Text columns (TF-IDF applied): {', '.join(text_cols)}")
    if skewed_cols:
        parts.append(f"Skewed columns (log-transformed): {', '.join(skewed_cols[:8])}")
    if top_corr:
        top_str = ", ".join(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
                            for k, v in list(top_corr.items())[:6])
        parts.append(f"Top correlations with target: {top_str}")

    # ── 3. Cleaning summary ───────────────────────────────────────────────────
    cleaning_summary = state_vals.get("cleaning_summary", {}) or {}
    if cleaning_summary:
        shape_after = cleaning_summary.get("shape_after", [])
        parts.append("\n=== DATA CLEANING ===")
        if shape_after:
            parts.append(f"Shape after cleaning: {shape_after[0]:,} rows × {shape_after[1]} columns")
        parts.append(f"Missing values after cleaning: {'none' if cleaning_summary.get('no_missing') else 'some remain'}")
        if null_patterns:
            np_str = "; ".join(f"{k}: {v}" for k, v in list(null_patterns.items())[:6])
            parts.append(f"Null handling: {np_str}")

    # ── 4. Feature engineering ────────────────────────────────────────────────
    feature_result = state_vals.get("feature_result", "")
    if feature_result and "FAILED" not in feature_result:
        parts.append("\n=== FEATURE ENGINEERING ===")
        keywords = ["CREATED", "INTERACTION", "LOG1P", "BIN", "DROPPED", "FREQ ENCODE",
                    "FEATURES BEFORE", "FEATURES AFTER", "NEW FEATURES CREATED",
                    "DATETIME", "TEXT TF-IDF"]
        fe_lines = [l.strip() for l in feature_result.split("\n")
                    if any(k in l.upper() for k in keywords) and l.strip()]
        if fe_lines:
            parts.extend(fe_lines[:20])

    # ── 5. Model results ──────────────────────────────────────────────────────
    viz_data = state_vals.get("visualization_data", {}) or {}
    bm       = viz_data.get("best_model", {}) or {}
    mc       = viz_data.get("model_comparison", {}) or {}
    cv       = viz_data.get("cross_validation", {}) or {}
    tuning   = viz_data.get("tuning", {}) or {}
    threshold= viz_data.get("threshold", {}) or {}
    lc_data  = viz_data.get("learning_curve", {}) or {}

    is_regression = "regression" in problem_type

    if bm:
        parts.append("\n=== MODEL RESULTS ===")
        parts.append(f"Winning model: {bm.get('name', 'unknown')}")
        if is_regression:
            for k in ["r2", "rmse", "mae"]:
                if bm.get(k) is not None:
                    parts.append(f"{k.upper()}: {bm[k]:.4f}")
        else:
            for k in ["test_f1", "test_recall", "test_precision", "test_roc_auc", "test_accuracy"]:
                if bm.get(k) is not None:
                    label = k.replace("test_", "").upper()
                    parts.append(f"{label}: {bm[k]:.4f}")

        pr_curve = bm.get("pr_curve", {})
        if pr_curve and pr_curve.get("avg_precision") is not None:
            parts.append(f"Average Precision (PR-AUC): {pr_curve['avg_precision']:.4f}")

    if mc:
        model_names = mc.get("model_names", [])
        if isinstance(model_names, list) and len(model_names) > 1:
            if is_regression:
                r2_vals = mc.get("r2", [])
                if r2_vals:
                    ranking = sorted(zip(model_names, r2_vals), key=lambda x: x[1], reverse=True)
                    parts.append("Model R² ranking: " + ", ".join(f"{n}: {v:.3f}" for n, v in ranking))
            else:
                f1_vals = mc.get("f1_score", [])
                if f1_vals:
                    ranking = sorted(zip(model_names, f1_vals), key=lambda x: x[1], reverse=True)
                    parts.append("Model F1 ranking: " + ", ".join(f"{n}: {v:.3f}" for n, v in ranking))

    if cv:
        cv_scores = cv.get("cv_scores", [])
        if cv_scores:
            parts.append(f"Cross-validation: mean={cv.get('mean', 0):.4f}, std={cv.get('std', 0):.4f}, folds={len(cv_scores)}")

    if tuning:
        parts.append(f"Hyperparameter tuning: before={tuning.get('before', 0):.4f}, after={tuning.get('after', 0):.4f}, delta={tuning.get('delta', 0):.4f}")

    if threshold:
        parts.append(f"Optimal decision threshold: {threshold.get('optimal', 0.5):.3f} (default 0.5 score: {threshold.get('metric_at_default', 0):.4f}, optimal score: {threshold.get('metric_at_optimal', 0):.4f})")

    if lc_data and lc_data.get("train_scores_mean") and lc_data.get("val_scores_mean"):
        final_train = lc_data["train_scores_mean"][-1]
        final_val   = lc_data["val_scores_mean"][-1]
        gap         = final_train - final_val
        diagnosis   = "high variance (overfitting)" if gap > 0.15 else ("high bias (underfitting)" if final_val < 0.6 and gap < 0.05 else "well-generalising")
        parts.append(f"Learning curve: training score={final_train:.3f}, validation score={final_val:.3f} → {diagnosis}")

    # ── 6. Feature importances ────────────────────────────────────────────────
    fi = bm.get("feature_importance", {})
    if fi and fi.get("feature_names"):
        names  = fi["feature_names"][:10]
        values = fi["importance_values"][:10]
        fi_str = ", ".join(f"{n}: {v:.4f}" for n, v in zip(names, values))
        parts.append(f"\n=== TOP FEATURES ===\n{fi_str}")

    # ── 7. Critique scorecard ─────────────────────────────────────────────────
    scorecard    = state_vals.get("scorecard", {}) or {}
    critique     = state_vals.get("critique_report", "") or ""
    suggestions  = state_vals.get("improvement_suggestions", []) or []

    if scorecard:
        parts.append("\n=== CRITIQUE SCORECARD (out of 10) ===")
        sc_lines = [f"{k.replace('_', ' ').title()}: {v}/10"
                    for k, v in scorecard.items() if isinstance(v, (int, float))]
        parts.extend(sc_lines)
    if suggestions:
        parts.append("Improvement suggestions: " + " | ".join(suggestions[:5]))

    # ── 8. Scout ranking ──────────────────────────────────────────────────────
    scout_ranking = state_vals.get("scout_ranking", [])
    if scout_ranking:
        scout_str = ", ".join(f"{name}: {score:.4f}" for name, score in scout_ranking[:5] if score > -999)
        parts.append(f"\n=== SCOUT RANKING (10% sample) ===\n{scout_str}")

    return "\n".join(parts)


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert ML engineer assistant embedded in an AutoML pipeline tool.
You have access to the full results of a machine learning analysis run. Your job is to answer
questions about this specific analysis — the data, the models, the features, the results, and
what it all means.

RULES:
1. Only answer based on the ANALYSIS CONTEXT provided. Do not make up numbers or results.
2. Be direct and specific — cite actual metrics and model names from the context.
3. If the question is a real-world analogy ("who would win?", "what would happen if..."),
   answer it in terms of the ML results, then give the analogy back to the user in plain language.
4. If asked about something not in the context (e.g., the pipeline hasn't run yet), say so clearly.
5. Keep answers concise — 2-5 sentences unless the user asks for detail.
6. If the user asks for predictions or "what would X score?", tell them to use the deployed
   API endpoint for real inference — you only have aggregate test set results.

TONE: Knowledgeable, conversational, and practical. Like a senior data scientist reviewing
results with a colleague."""


# ── Suggested questions ───────────────────────────────────────────────────────

def _suggested_questions(state_vals: dict) -> list[str]:
    """Build 4 highly specific questions from the actual pipeline run data."""
    problem_type  = state_vals.get("problem_type", "")
    target_col    = state_vals.get("target_column", "the target")
    is_regression = "regression" in problem_type
    viz           = state_vals.get("visualization_data") or {}
    bm            = viz.get("best_model") or {}
    mc            = viz.get("model_comparison") or {}
    rc            = state_vals.get("reasoning_context") or {}
    ds            = state_vals.get("dataset_summary") or {}
    lc            = viz.get("learning_curve") or {}

    best_name     = bm.get("name", "the model")
    questions: list[str] = []

    # ── Q1: Why did the best model win? (name the runner-up if available) ─────
    model_names = mc.get("model_names", [])
    if not isinstance(model_names, list):
        model_names = [model_names] if model_names else []
    losers = [m for m in model_names if m != best_name]
    if losers:
        runner_up = losers[0]
        questions.append(f"Why did {best_name} beat {runner_up} and the other models?")
    else:
        questions.append(f"Why was {best_name} chosen as the best model?")

    # ── Q2: Feature-specific — use actual top feature name ────────────────────
    fi = bm.get("feature_importance") or {}
    top_feature = (fi.get("feature_names") or [None])[0]
    corr_map    = ds.get("correlations_with_target") or {}
    top_corr_feat = max(corr_map, key=lambda k: abs(float(corr_map[k])) if isinstance(corr_map[k], (int, float)) else 0, default=None) if corr_map else None

    if top_feature:
        questions.append(f"Why is '{top_feature}' the most important feature for predicting {target_col}?")
    elif top_corr_feat:
        questions.append(f"'{top_corr_feat}' has the highest correlation with {target_col} — does that make it the most useful feature?")
    else:
        questions.append(f"Which features matter most for predicting {target_col}?")

    # ── Q3: Metric-specific — cite the actual score ───────────────────────────
    if is_regression:
        r2   = bm.get("r2")
        rmse = bm.get("rmse")
        if r2 is not None and rmse is not None:
            questions.append(f"The model has R²={r2:.3f} and RMSE={rmse:.3f} — is that good for this dataset?")
        elif r2 is not None:
            questions.append(f"R² is {r2:.3f} — is that good enough to trust for real decisions?")
        else:
            questions.append(f"What does the RMSE mean in practical terms for {target_col}?")
    else:
        imb_ratio = rc.get("imbalance_ratio", "N/A")
        imb_strat = rc.get("imbalance_strategy", "none")
        f1 = bm.get("test_f1")
        auc = bm.get("test_roc_auc")
        if imb_strat not in ("none", "") and f1 is not None:
            questions.append(f"Class imbalance ratio is {imb_ratio} — how did {imb_strat} affect the F1 score of {f1:.3f}?")
        elif auc is not None and f1 is not None:
            questions.append(f"AUC-ROC is {auc:.3f} and F1 is {f1:.3f} — which metric should I trust more here?")
        else:
            questions.append(f"Should I trust the accuracy score for this dataset?")

    # ── Q4: Bias/variance, data quality, or deployment readiness ─────────────
    if lc.get("train_scores_mean") and lc.get("val_scores_mean"):
        final_train = lc["train_scores_mean"][-1]
        final_val   = lc["val_scores_mean"][-1]
        gap = final_train - final_val
        if gap > 0.15:
            questions.append(f"The learning curve shows a gap of {gap:.2f} between training and validation — how do I fix the overfitting?")
        elif final_val < 0.6:
            questions.append(f"Both training and validation scores are low ({final_val:.2f}) — what's causing the underfitting?")
        else:
            questions.append(f"The learning curve looks balanced — would adding more data still improve the model?")
    else:
        skewed = ds.get("skewed_columns", [])
        issues = state_vals.get("data_issues", [])
        if skewed:
            questions.append(f"Why were {skewed[0]} and other skewed columns log-transformed before training?")
        elif issues:
            questions.append(f"There were {len(issues)} data issues found — how much did they affect the model?")
        else:
            n_rows = (ds.get("shape") or [0])[0]
            questions.append(f"With only {n_rows:,} rows, how reliable are these results?" if isinstance(n_rows, int) and n_rows < 1000 else "Is this model ready to deploy?")

    return questions[:4]


# ── Main render function ──────────────────────────────────────────────────────

def render_chat_panel(state_vals: dict):
    has_results = bool(
        state_vals and (
            state_vals.get("visualization_data") or
            state_vals.get("model_result") or
            state_vals.get("profile_report")
        )
    )

    # Header
    st.markdown(
        '<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.25rem">'
        '<span style="font-size:1.4rem">🤖</span>'
        '<span style="font-size:1.1rem;font-weight:700;color:var(--text-pri)">Ask the AI</span>'
        '</div>'
        '<p style="font-size:0.85rem;color:var(--text-muted);margin-bottom:1rem">'
        'Chat with an AI that has read your entire ML analysis — ask anything about '
        'the results, the models, the features, or what to try next.'
        '</p>',
        unsafe_allow_html=True,
    )

    if not has_results:
        st.markdown(
            '<div style="background:var(--surface);border:1px solid var(--border);'
            'border-radius:var(--radius-card);padding:1.5rem;text-align:center;'
            'color:var(--text-muted);font-size:0.9rem">'
            '⏳ Run the pipeline first — the AI needs analysis results to answer questions.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # Build analysis context (cached per run so we don't rebuild on every message)
    ctx_key = "chat_context"
    if ctx_key not in st.session_state:
        st.session_state[ctx_key] = build_chat_context(state_vals)
    context = st.session_state[ctx_key]

    # Rebuild context if pipeline completed a new agent since last chat
    current_agent = state_vals.get("current_agent", "")
    agent_key     = "chat_last_agent"
    if st.session_state.get(agent_key) != current_agent:
        st.session_state[ctx_key]  = build_chat_context(state_vals)
        st.session_state[agent_key] = current_agent
        context = st.session_state[ctx_key]

    # Init chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Suggested questions (only when history is empty)
    if not st.session_state["chat_history"]:
        st.markdown(
            '<p style="font-size:0.8rem;font-weight:600;color:var(--text-muted);margin-bottom:0.4rem">Suggested questions</p>',
            unsafe_allow_html=True,
        )
        suggested = _suggested_questions(state_vals)
        cols = st.columns(2)
        for i, q in enumerate(suggested[:4]):
            with cols[i % 2]:
                if st.button(q, key=f"sq_{i}", width="stretch"):
                    st.session_state["chat_history"].append({"role": "user", "content": q})
                    st.rerun()
        st.markdown('<hr style="border:none;border-top:1px solid var(--border);margin:0.75rem 0">', unsafe_allow_html=True)

    # Render existing messages
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # If last message is from user, generate assistant reply
    history = st.session_state["chat_history"]
    if history and history[-1]["role"] == "user":
        user_question = history[-1]["content"]
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    messages = [
                        SystemMessage(content=_SYSTEM_PROMPT),
                        HumanMessage(content=(
                            f"ANALYSIS CONTEXT:\n{context}\n\n"
                            f"USER QUESTION: {user_question}"
                        )),
                    ]
                    # Inject prior turns (last 6 exchanges) as context
                    if len(history) > 2:
                        prior_turns = history[:-1][-12:]
                        prior_text = "\n".join(
                            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
                            for m in prior_turns
                        )
                        messages = [
                            SystemMessage(content=_SYSTEM_PROMPT),
                            HumanMessage(content=(
                                f"ANALYSIS CONTEXT:\n{context}\n\n"
                                f"CONVERSATION SO FAR:\n{prior_text}\n\n"
                                f"USER QUESTION: {user_question}"
                            )),
                        ]
                    response, model_used = call_llm_with_fallback(messages, temperature=0.4)
                    reply = response.content
                except Exception as e:
                    reply = f"Sorry, I couldn't generate a response: {str(e)[:120]}"
            st.markdown(reply)
        st.session_state["chat_history"].append({"role": "assistant", "content": reply})

    # Chat input
    user_input = st.chat_input("Ask anything about your analysis...")
    if user_input and user_input.strip():
        st.session_state["chat_history"].append({"role": "user", "content": user_input.strip()})
        st.rerun()

    # Clear button
    if st.session_state["chat_history"]:
        st.markdown('<div style="margin-top:0.5rem">', unsafe_allow_html=True)
        if st.button("Clear conversation", key="chat_clear"):
            st.session_state["chat_history"] = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
