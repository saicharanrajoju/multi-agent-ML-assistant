import streamlit as st
import os
import pandas as pd
from typing import Callable


def render_sidebar(datasets_dir: str, on_reset: Callable) -> str:
    """
    Renders the sidebar for dataset selection and configuration.
    Returns the selected dataset path filename.
    Logic unchanged — visual layer only.
    """
    with st.sidebar:
        # Wordmark
        st.markdown(
            '<div style="padding:0.5rem 0 1rem">'
            '<span style="font-size:1.05rem;font-weight:700;color:var(--text-pri);letter-spacing:-0.02em">ML Agent</span>'
            '<span style="font-size:1.05rem;font-weight:400;color:var(--accent-dark)"> Assistant</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p style="font-size:0.72rem;font-weight:600;text-transform:uppercase;'
            'letter-spacing:0.08em;color:var(--text-muted);margin:0 0 0.5rem">Dataset</p>',
            unsafe_allow_html=True,
        )

        available_datasets = sorted([f for f in os.listdir(datasets_dir) if f.endswith('.csv')])

        dataset_choice = st.radio(
            "Select a dataset",
            options=["Upload my own"] + available_datasets,
            index=0,
            label_visibility="collapsed",
        )

        dataset_path = ""
        if dataset_choice == "Upload my own":
            uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], label_visibility="collapsed")
            if uploaded_file:
                save_path = os.path.join(datasets_dir, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                dataset_path = uploaded_file.name
                st.markdown(
                    f'<div style="font-size:0.8rem;color:var(--success);padding:0.3rem 0">'
                    f'&#10003; {uploaded_file.name}</div>',
                    unsafe_allow_html=True,
                )
        else:
            dataset_path = dataset_choice

        # Dataset preview
        if dataset_path:
            try:
                preview_path = os.path.join(datasets_dir, dataset_path)
                if os.path.exists(preview_path):
                    df_preview = pd.read_csv(preview_path, nrows=3)
                    total_rows = sum(1 for _ in open(preview_path, encoding="utf-8", errors="replace")) - 1
                    st.markdown(
                        f'<div style="display:flex;gap:0.5rem;margin:0.75rem 0 0.4rem;flex-wrap:wrap">'
                        f'<span style="font-size:0.72rem;background:#F3F4F6;color:var(--text-sec);'
                        f'padding:2px 8px;border-radius:999px;font-weight:500">'
                        f'{total_rows:,} rows</span>'
                        f'<span style="font-size:0.72rem;background:#F3F4F6;color:var(--text-sec);'
                        f'padding:2px 8px;border-radius:999px;font-weight:500">'
                        f'{df_preview.shape[1]} cols</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.dataframe(df_preview, height=120, width="stretch")
            except Exception:
                pass

        # Divider + destructive action at bottom
        st.markdown('<hr style="border:none;border-top:1px solid var(--border);margin:1.25rem 0 0.75rem">', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.72rem;color:var(--text-muted);margin:0 0 0.5rem">Danger zone</p>',
            unsafe_allow_html=True,
        )
        if st.button("Reset everything", key="sidebar_reset", width="stretch"):
            on_reset()

    return dataset_path
