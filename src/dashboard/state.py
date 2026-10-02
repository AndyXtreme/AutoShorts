from __future__ import annotations

import streamlit as st

from dashboard.utils.process import ProcessManager


def get_process_manager() -> ProcessManager:
    if "process_manager" not in st.session_state:
        st.session_state.process_manager = ProcessManager()
    return st.session_state.process_manager
