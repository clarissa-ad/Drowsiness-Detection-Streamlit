# components/dashboard.py
import streamlit as st

def render_sidebar():
    st.sidebar.title("Settings")
    threshold = st.sidebar.slider("Drowsiness Threshold", 0.0, 1.0, 0.5)
    return threshold