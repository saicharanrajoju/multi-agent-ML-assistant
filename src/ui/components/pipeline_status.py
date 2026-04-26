import streamlit as st
from typing import Callable
from src.ui.ui_components import step_tracker, banner

AGENT_STEPS = [
    {"name": "profiler",          "label": "Profiling"},
    {"name": "cleaner",           "label": "Cleaning"},
    {"name": "feature_engineer",  "label": "Features"},
    {"name": "modeler",           "label": "Modeling"},
    {"name": "critic",            "label": "Critique"},
]

AGENT_NAME_TO_STEP = {step["name"]: i for i, step in enumerate(AGENT_STEPS)}


def render_pipeline_progress(current_step: int, is_running: bool, is_complete: bool):
    """Renders the step tracker bar. Logic unchanged — visual layer only."""
    if is_running or is_complete:
        html = step_tracker(AGENT_STEPS, current_step, is_complete)
        st.markdown(html, unsafe_allow_html=True)


def render_pipeline_logs(
    logs: list,
    last_error: str,
    is_running: bool,
    on_reset: Callable,
    on_retry: Callable,
):
    """Renders pipeline activity log and error recovery. Logic unchanged."""
    if not logs and not last_error:
        return

    if logs:
        # Compact log panel — most recent 6 entries, oldest at bottom
        visible = logs[-6:]
        rows = []
        for entry in visible:
            if "❌" in entry or "Error" in entry:
                rows.append(
                    f'<div style="display:flex;align-items:baseline;gap:0.5rem;padding:0.3rem 0;'
                    f'border-bottom:1px solid var(--border)">'
                    f'<span style="color:var(--error);font-size:0.8rem">✕</span>'
                    f'<span style="font-size:0.82rem;color:var(--text-pri)">{entry}</span></div>'
                )
            elif "✅" in entry:
                rows.append(
                    f'<div style="display:flex;align-items:baseline;gap:0.5rem;padding:0.3rem 0;'
                    f'border-bottom:1px solid var(--border)">'
                    f'<span style="color:var(--success);font-size:0.8rem">✓</span>'
                    f'<span style="font-size:0.82rem;color:var(--text-sec)">{entry}</span></div>'
                )
            elif "⏸️" in entry or "Waiting" in entry:
                rows.append(
                    f'<div style="display:flex;align-items:baseline;gap:0.5rem;padding:0.3rem 0;'
                    f'border-bottom:1px solid var(--border)">'
                    f'<span style="color:var(--warn);font-size:0.8rem">⏸</span>'
                    f'<span style="font-size:0.82rem;color:var(--text-sec)">{entry}</span></div>'
                )
            else:
                rows.append(
                    f'<div style="display:flex;align-items:baseline;gap:0.5rem;padding:0.3rem 0;'
                    f'border-bottom:1px solid var(--border)">'
                    f'<span style="color:var(--accent);font-size:0.8rem">›</span>'
                    f'<span style="font-size:0.82rem;color:var(--text-sec)">{entry}</span></div>'
                )

        total = len(logs)
        hidden = max(0, total - 6)
        hidden_note = (
            f'<div style="font-size:0.75rem;color:var(--text-muted);padding:0.3rem 0">'
            f'+ {hidden} earlier entries</div>'
            if hidden > 0 else ""
        )

        st.markdown(
            f'<div style="background:var(--surface);border:1px solid var(--border);'
            f'border-radius:var(--radius-card);padding:0.75rem 1rem;margin:0.75rem 0">'
            f'<div style="font-size:0.72rem;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:0.06em;color:var(--text-muted);margin-bottom:0.5rem">Activity</div>'
            f'{hidden_note}{"".join(rows)}</div>',
            unsafe_allow_html=True,
        )

    if last_error and not is_running:
        st.markdown(
            banner(last_error, kind="error", title="Pipeline error"),
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Reset everything", key="err_reset", use_container_width=True):
                on_reset()
        with c2:
            if st.button("Retry last step", key="err_retry", use_container_width=True):
                on_retry()
