import streamlit as st
import os

def apply_custom_styles():
    """Injects the design-system CSS from styles.css."""
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
