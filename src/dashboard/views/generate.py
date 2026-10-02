from __future__ import annotations

import os
import sys
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if parent.name == "src":
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

import streamlit as st

from dashboard.components.log_viewer import render_logs
from dashboard.state import get_process_manager
from dashboard.utils.config import load_env_values, save_env_values
from dashboard.utils.thumbnails import list_videos

for parent in Path(__file__).resolve().parents:
    if parent.name == "src":
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break




VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


def _folder_signature(folder: Path) -> tuple:
    """Cheap fingerprint of a folder's videos: name, size and mtime.

    Deliberately avoids opening the files - this runs on a timer, and decoding
    every video on each poll would keep a GPU-sized process busy for nothing.
    Size is part of the signature so a file still being copied in registers as
    changed until the copy settles.
    """
    if not folder.exists():
        return ()
    entries = []
    for path in folder.iterdir():
        if path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append((path.name, stat.st_size, int(stat.st_mtime)))
    return tuple(sorted(entries))


@st.fragment(run_every="3s")
def _watch_gameplay_folder(folder: Path) -> None:
    """Rerun the page when the gameplay folder changes underneath us."""
    signature = _folder_signature(folder)
    previous = st.session_state.get("_gameplay_signature")
    st.session_state["_gameplay_signature"] = signature
    if previous is not None and previous != signature:
        st.rerun(scope="app")


def _build_env(overrides: dict) -> dict:
    env = os.environ.copy()
    env.update({k: str(v) for k, v in overrides.items()})
    return env


UPLOAD_CHUNK = 8 * 1024 * 1024


@st.dialog("Delete from gameplay/?")
def _confirm_delete(paths: list) -> None:
    """Delete recordings from the queue for good, after asking once.

    Removing used to move the file into gameplay/.disabled/, which kept the
    whole recording on disk and recreated that folder on every page load.
    Clips already generated from a recording are not touched.
    """
    st.markdown("\n".join(f"- {p.name}" for p in paths))
    st.caption("The files are deleted permanently. Clips already generated from them stay in generated/.")
    yes, no = st.columns(2)
    if yes.button("🗑️ Delete", type="primary", width="stretch"):
        for path in paths:
            Path(path).unlink(missing_ok=True)
        st.rerun()
    if no.button("Cancel", width="stretch"):
        st.rerun()


def _save_upload(upload, destination: Path) -> None:
    """Copy a browser upload into gameplay/, showing how far along it is.

    Streamlit dims the entire page while a rerun is in flight, so writing a
    multi-gigabyte recording in one shutil call looks exactly like a hung UI.
    Copying in chunks lets a progress bar report the one thing worth knowing.
    """
    total = int(getattr(upload, "size", 0) or 0)
    caption = st.empty()
    bar = st.progress(0.0)
    written = 0
    upload.seek(0)
    try:
        with open(destination, "wb") as target:
            while True:
                chunk = upload.read(UPLOAD_CHUNK)
                if not chunk:
                    break
                target.write(chunk)
                written += len(chunk)
                if total:
                    bar.progress(min(written / total, 1.0))
                    caption.caption(
                        f"Saving {upload.name} - {written / 1e9:.2f} / {total / 1e9:.2f} GB"
                    )
                else:
                    caption.caption(f"Saving {upload.name} - {written / 1e9:.2f} GB")
    finally:
        bar.empty()
        caption.empty()


def render() -> None:
    st.set_page_config(page_title="Generate - AutoShorts", page_icon="🎬", layout="wide")
    
    # Import shared theme
    from dashboard.utils.theme import get_shared_css
    
    # Apply shared theme
    st.markdown(get_shared_css(), unsafe_allow_html=True)
    

    # Header
    st.markdown("""
        <div class="page-header">
            <h1>🎬 Generate Shorts</h1>
            <p>Select gameplay, configure style, and generate viral clips</p>
        </div>
    """, unsafe_allow_html=True)

    values, extras = load_env_values()
    gameplay_videos = list_videos(Path("gameplay"))

    # Two-column layout
    col_left, col_right = st.columns([1, 1])
    
    with col_left:

        # Video Queue Management
        st.markdown("### 🎬 Video Queue")

        # Directories
        gameplay_dir = Path("gameplay")

        # Pick up files copied into gameplay/ from outside the UI. Streamlit only
        # redraws on interaction, so without this a file dropped into the mounted
        # folder would stay invisible until the user clicked something. The
        # fragment polls cheaply (names, sizes, mtimes - no video decoding) and
        # reruns the whole page only when the folder actually changed.
        _watch_gameplay_folder(gameplay_dir)

        st.caption(f"📂 Watching `{gameplay_dir.resolve()}` - files copied in here show up automatically.")

        gameplay_videos = list_videos(gameplay_dir)

        # Display current queue
        queue_container = st.container()
        if gameplay_videos:
            st.info(f"📁 {len(gameplay_videos)} video(s) in active queue (will be processed)")
            
            # Show thumbnails grid
            cols = st.columns(3)
            for idx, vid in enumerate(gameplay_videos):
                col = cols[idx % 3]
                with col:
                    if vid.thumbnail:
                        st.image(str(vid.thumbnail), width="stretch")
                    st.caption(f"**{vid.path.name}**")
                    st.caption(f"⏱️ {vid.duration:.0f}s | 💾 {vid.size_mb:.1f}MB")
                    
                    if st.button("🗑️ Delete", key=f"delete_{vid.path.name}",
                                 help="Delete this recording from gameplay/"):
                        _confirm_delete([vid.path])

        else:
            st.info("The queue is empty. Add a video to start.")

        # Action Buttons
        st.divider()
        col_add, col_clear, col_rescan = st.columns([2, 1, 1])
        
        with col_add:
            # Upload rather than a native file dialog: the previous version
            # shelled out to zenity/tkinter, which do not exist in the container
            # the UI is served from. Dropping files into gameplay/ on the host
            # works just as well and shows up here on the next rerun.
            upload_round = st.session_state.get("_upload_round", 0)
            uploaded = st.file_uploader(
                "📂 Add video",
                type=["mp4", "mkv", "mov", "avi", "webm", "m4v"],
                accept_multiple_files=True,
                key=f"gameplay_upload_{upload_round}",
                help="Copies into gameplay/. Browser uploads are buffered in memory, so for "
                     "very large recordings it is faster and safer to copy the file straight "
                     "into the gameplay/ folder - it is picked up automatically.",
            )
            if uploaded:
                added = 0
                gameplay_dir.mkdir(exist_ok=True)
                for upload in uploaded:
                    destination = gameplay_dir / upload.name
                    if destination.exists():
                        st.warning(f"⚠️ '{upload.name}' is already in the queue")
                        continue
                    try:
                        _save_upload(upload, destination)
                        added += 1
                    except Exception as e:
                        destination.unlink(missing_ok=True)
                        st.error(f"❌ Could not save '{upload.name}': {e}")
                if added:
                    st.success(f"✅ Added {added} video(s) to the queue")
                    # A new widget key drops Streamlit's in-memory copy of the
                    # upload, which is the whole file and stays for the session.
                    st.session_state["_upload_round"] = upload_round + 1
                    st.rerun()

        with col_clear:
            if st.button("🗑️ Clear Queue", type="secondary", width="stretch", disabled=not gameplay_videos,
                         help="Delete every recording in gameplay/"):
                _confirm_delete([vid.path for vid in gameplay_videos])

        with col_rescan:
            # The folder is polled every few seconds anyway; this is for when you
            # just finished copying a file and do not want to wait for the tick.
            if st.button("🔄 Rescan", type="secondary", width="stretch",
                         help="Re-read the gameplay folder now"):
                st.rerun()

        st.divider()
        
        # Quick settings
        st.markdown("### ⚡ Quick Settings")
        
        def save_scene_limit():
            values["SCENE_LIMIT"] = st.session_state.quick_scene_limit
            save_env_values(values, extras)

        st.number_input(
            "🎬 Clips to generate",
            value=int(values.get("SCENE_LIMIT", 6)),
            min_value=1,
            max_value=20,
            key="quick_scene_limit",
            help="Number of short clips to create from the video",
            on_change=save_scene_limit
        )
    
    with col_right:
        # PyCaps Subtitle Style selector
        with st.container():
            st.markdown("### ✨ Subtitle Style")

            PYCAPS_TEMPLATES = {
                "hype":         {"icon": "⚡", "desc": "Comic bold, yellow highlights", "color": "#FFFF00", "bg": "#1a1a2e", "shadow": "#000", "font": "Impact"},
                "vibrant":      {"icon": "🌈", "desc": "Neon chromatic aberration", "color": "#FFFFFF", "bg": "#1a1a2e", "shadow": "#FF00FF", "font": "Impact"},
                "explosive":    {"icon": "💥", "desc": "Orange glow, intense energy", "color": "#FFDD00", "bg": "#1a1a2e", "shadow": "#FF4400", "font": "Impact"},
                "word-focus":   {"icon": "🎯", "desc": "Orange word highlighting", "color": "#FFFFFF", "bg": "#f76f00", "shadow": "none", "font": "Arial Black"},
                "line-focus":   {"icon": "📌", "desc": "Blue line highlight bar", "color": "#FFFFFF", "bg": "#0055DD", "shadow": "none", "font": "Arial Black"},
                "retro-gaming": {"icon": "👾", "desc": "Pixel arcade aesthetic", "color": "#FFFF88", "bg": "#0a0a3a", "shadow": "#000", "font": "monospace"},
                "neo-minimal":  {"icon": "💻", "desc": "Code editor dark theme", "color": "#569CD6", "bg": "#1E1E1E", "shadow": "none", "font": "Consolas, monospace"},
                "fast":         {"icon": "💨", "desc": "Thick outline, bold cursive", "color": "#FFFFFF", "bg": "#1a1a2e", "shadow": "#000", "font": "cursive"},
                "classic":      {"icon": "📝", "desc": "Clean white on black shadow", "color": "#FFFFFF", "bg": "#1a1a2e", "shadow": "#000", "font": "Arial"},
                "minimalist":   {"icon": "🤍", "desc": "Subtle, semi-transparent", "color": "rgba(255,255,255,0.9)", "bg": "rgba(0,0,0,0.5)", "shadow": "none", "font": "Helvetica Neue, Arial"},
                "default":      {"icon": "⬜", "desc": "Standard white text", "color": "#FFFFFF", "bg": "#1a1a2e", "shadow": "#000", "font": "Arial Black"},
            }

            tpl_names = list(PYCAPS_TEMPLATES.keys())
            current_template = str(values.get("PYCAPS_TEMPLATE", extras.get("PYCAPS_TEMPLATE", "hype")))
            tpl_idx = tpl_names.index(current_template) if current_template in tpl_names else 0

            def save_template():
                new_tpl = st.session_state.gen_template
                if new_tpl != current_template:
                    # Persist template as an extra setting (moved out of Settings page)
                    extras["PYCAPS_TEMPLATE"] = new_tpl
                    values["PYCAPS_TEMPLATE"] = new_tpl
                    save_env_values(values, extras)

            selected = st.selectbox(
                "Subtitle template",
                options=tpl_names,
                index=tpl_idx,
                key="gen_template",
                format_func=lambda v: f"{PYCAPS_TEMPLATES[v]['icon']} {v}",
                on_change=save_template,
                label_visibility="collapsed",
            )

            # Render preview for selected template
            info = PYCAPS_TEMPLATES[selected]
            if info["shadow"] == "none":
                txt_shadow = "none"
            else:
                s = info["shadow"]
                txt_shadow = f"2px 2px 0 {s}, -2px -2px 0 {s}, 2px -2px 0 {s}, -2px 2px 0 {s}"

            st.markdown(f"""
                <div style="
                    background: {info['bg']};
                    border: 2px solid #6C63FF;
                    border-radius: 10px;
                    padding: 20px 16px;
                    text-align: center;
                    margin-top: 8px;
                    min-height: 70px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                ">
                    <span style="
                        color: {info['color']};
                        font-family: {info['font']};
                        font-size: 20px;
                        font-weight: 700;
                        text-shadow: {txt_shadow};
                        text-transform: uppercase;
                        letter-spacing: 1px;
                    ">HEADSHOT!</span>
                </div>
            """, unsafe_allow_html=True)
            st.caption(f"{info['desc']}")

    st.divider()

    # Job control section
    manager = get_process_manager()
    status = manager.status()

    st.markdown("### 🚀 Processing")
    
    control_cols = st.columns([1, 1, 2])
    
    with control_cols[0]:
        start_disabled = status.running
        if st.button(
            "▶️ Start Processing",
            type="primary",
            width="stretch",
            disabled=start_disabled
        ):
            save_env_values(values, extras)
            manager.start(env=_build_env(values))
            st.rerun()
    
    with control_cols[1]:
        stop_disabled = not status.running
        if st.button(
            "⏹️ Stop",
            type="secondary",
            width="stretch",
            disabled=stop_disabled
        ):
            manager.stop()
            st.rerun()
    
    with control_cols[2]:
        # Status indicator
        if status.running:
            st.success(f"🟢 Running (PID {status.pid})")
        elif status.exit_code is not None:
            if status.exit_code == 0:
                st.info(f"✅ Completed successfully")
            else:
                st.error(f"❌ Exited with code {status.exit_code}")
        else:
            st.warning("⏸️ Idle - Ready to process")

    # Logs section
    st.markdown("### 📋 Logs")
    show_all = st.toggle("Show full logs", value=False, key="show_all_logs")
    render_logs(status.tail, show_all=show_all)

    # Auto-refresh while running
    # Auto-refresh while running (only if not viewing full logs)
    if status.running and not show_all:
        import time
        time.sleep(2)
        st.rerun()
