from __future__ import annotations

import sys
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if parent.name == "src":
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

import streamlit as st

from dashboard.components.video_player import render_video
from dashboard.utils.thumbnails import list_videos


LOGO_PATH = Path("assets/logo.png")


def _render_video_grid(videos, cols_count=3, key_prefix="vid"):
    """Render videos as thumbnail grid with click-to-play."""
    if not videos:
        st.info("📁 No videos found in this folder.")
        return
    
    # Initialize selected video in session state
    selected_key = f"{key_prefix}_selected"
    if selected_key not in st.session_state:
        st.session_state[selected_key] = None
    
    # Video player at TOP (visible immediately)
    if st.session_state[selected_key]:
        selected_path = st.session_state[selected_key]
        if selected_path.exists():
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1A1D2E 0%, #1E3A5F 100%); 
                            padding: 1rem; border-radius: 10px; margin-bottom: 1rem;
                            border: 1px solid #2D3348;">
                    <h3 style="margin: 0;">▶️ {selected_path.parent.name} / {selected_path.name}</h3>
                </div>
            """, unsafe_allow_html=True)
            
            st.video(str(selected_path))
            
            col1, col2, col3 = st.columns([1, 1, 2])
            if col1.button("✕ Close Player", key=f"{key_prefix}_close", width="stretch"):
                st.session_state[selected_key] = None
                st.rerun()
            if col2.button("🗑️ Delete", key=f"{key_prefix}_delete", type="secondary", width="stretch"):
                try:
                    selected_path.unlink()
                    st.session_state[selected_key] = None
                    st.toast(f"Deleted {selected_path.name}", icon="🗑️")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete: {e}")
            
            st.divider()
        else:
            st.warning(f"Video not found: {selected_path}")
            st.session_state[selected_key] = None
    
    # Stats bar
    total_size = sum(v.size_mb for v in videos)
    total_duration = sum(v.duration for v in videos)
    st.markdown(f"**{len(videos)} videos** • {total_size:.1f} MB total • {total_duration/60:.1f} min total duration")
    
    # Thumbnail grid
    rows = [videos[i:i + cols_count] for i in range(0, len(videos), cols_count)]
    for row in rows:
        cols = st.columns(cols_count)
        for idx, info in enumerate(row):
            with cols[idx]:
                # Card container
                with st.container():
                    # Thumbnail
                    if info.thumbnail and info.thumbnail.exists():
                        st.image(str(info.thumbnail), width="stretch")
                    else:
                        st.markdown("""
                            <div style="background: #1E2030; height: 120px; display: flex; 
                                        align-items: center; justify-content: center; 
                                        border-radius: 8px; font-size: 2rem;
                                        border: 1px solid #2D3348;">
                                🎬
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Clips are grouped in one folder per source video, so the
                    # file name alone is just "scene-0" - show the source too.
                    label = f"{info.path.parent.name} / {info.path.name}"
                    st.markdown(f"**{label[:34]}{'...' if len(label) > 34 else ''}**")
                    st.caption(f"⏱️ {info.duration:.0f}s • 💾 {info.size_mb:.1f}MB")

                    # Key must include the folder: several sources produce a
                    # scene-0, and duplicate widget keys are a hard Streamlit error.
                    widget_key = f"{key_prefix}_{info.path.parent.name}_{info.path.stem}"
                    if st.button("▶️ Play", key=widget_key, width="stretch"):
                        st.session_state[selected_key] = info.path
                        st.rerun()


def render() -> None:
    st.set_page_config(page_title="Browse - AutoShorts", page_icon="📁", layout="wide")
    
    # Sidebar logo
    if LOGO_PATH.exists():
        st.logo(str(LOGO_PATH))
    
    # Import shared theme
    from dashboard.utils.theme import get_shared_css
    
    # Apply shared theme
    st.markdown(get_shared_css(), unsafe_allow_html=True)
    
    # Header
    st.markdown("""
        <div class="page-header">
            <h1>📁 Browse Files</h1>
            <p>View and manage your gameplay and generated clips</p>
        </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🎬 Generated Clips", "🎮 Gameplay Videos"])
    
    with tab1:
        # recursive: clips live in one subfolder per source video
        generated_videos = list_videos(Path("generated"), recursive=True)
        # Sort by modification time, newest first
        generated_videos = sorted(generated_videos, key=lambda v: v.path.stat().st_mtime, reverse=True)
        
        if generated_videos:
            # Filter options
            col1, col2 = st.columns([2, 1])
            with col2:
                sort_order = st.selectbox(
                    "Sort by",
                    ["Newest first", "Oldest first", "Largest first", "Name A-Z"],
                    key="gen_sort"
                )
                
                if sort_order == "Oldest first":
                    generated_videos = sorted(generated_videos, key=lambda v: v.path.stat().st_mtime)
                elif sort_order == "Largest first":
                    generated_videos = sorted(generated_videos, key=lambda v: v.size_mb, reverse=True)
                elif sort_order == "Name A-Z":
                    generated_videos = sorted(generated_videos, key=lambda v: v.path.name.lower())
        
        _render_video_grid(generated_videos, cols_count=4, key_prefix="gen")
    
    with tab2:
        gameplay_videos = list_videos(Path("gameplay"))
        _render_video_grid(gameplay_videos, cols_count=3, key_prefix="gameplay")


if __name__ == "__main__":
    render()
