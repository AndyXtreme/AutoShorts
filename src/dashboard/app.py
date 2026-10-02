"""AutoShorts web UI: generate Shorts, and the settings that shape them."""
from __future__ import annotations

import sys
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if parent.name == "src":
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

import streamlit as st

from dashboard.views import generate, settings

st.navigation(
    [
        st.Page(generate.render, title="Generate", icon="🎬", url_path="generate", default=True),
        st.Page(settings.render, title="Settings", icon="⚙️", url_path="settings"),
    ]
).run()
