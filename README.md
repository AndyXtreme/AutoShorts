# AutoShorts

> Automatically generate viral-ready vertical short clips from long-form gameplay footage using AI-powered scene analysis, GPU-accelerated rendering, and optional AI voiceovers.

AutoShorts analyzes your gameplay videos to identify the most engaging moments—action sequences, funny fails, or highlight achievements—then automatically crops, renders, and adds subtitles or AI voiceovers to create ready-to-upload short-form content.

![Python](https://img.shields.io/badge/python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.x-green)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎬 Example Output

Here are some shorts automatically generated from gameplay footage:

| sample 1 | sample 2 | sample 3 | sample 4 |
| :---: | :---: | :---: | :---: |
| ![sample 1](generated/showcase/indianajones_pt1_scene-0.gif) | ![sample 2](generated/showcase/indianajones_pt1_scene-1.gif) | ![sample 3](generated/showcase/indianajones_pt1_scene-2.gif) | ![sample 4](generated/showcase/indianajones_pt1_scene-3.gif) |

### 🎥 Showcase: Multi-Language & Style Generation

AutoShorts automatically adapts its editing style, captions, and voiceover personality based on the content and target language. Here are some examples generated entirely by the pipeline:

| Content | Style | Language | Video |
| :--- | :--- | :--- | :--- |
| **Fortnite** | Story Roast | 🇺🇸 English | [Watch Part 1](https://www.youtube.com/shorts/tTUipTAdBlk) |
| **Indiana Jones** | GenZ Slang | 🇺🇸 English | [Watch Part 1](https://www.youtube.com/shorts/VAOlR5RAX14) |
| **Battlefield 6** | Dramatic Story | 🇯🇵 Japanese | [Watch Part 1](https://www.youtube.com/shorts/DYNEr1CzTpY) |
| **Indiana Jones** | Story News | 🇨🇳 Chinese | [Watch Part 1](https://www.youtube.com/shorts/kGRrpu66fpk) |
| **Fortnite** | Story Roast | 🇪🇸 Spanish | [Watch Part 1](https://www.youtube.com/shorts/5QcelWS1oSo) |
| **Fortnite** | Story Roast | 🇷🇺 Russian | [Watch Part 1](https://www.youtube.com/shorts/A06FdnycTYo) |
| **Indiana Jones** | Auto Gameplay | 🇧🇷 Portuguese | [Watch Part 1](https://www.youtube.com/shorts/qDFsTnH9qxc) |

---

## ✨ Features

### 🎯 AI-Powered Scene Analysis

- **Multi-Provider Support**: Choose between **OpenAI** (GPT-5-mini, GPT-4o) or **Google Gemini** for scene analysis, or run in `local` mode with heuristic scoring (no API needed)
- **Gemini Deep Analysis Mode** 🧠: Upload full video to Gemini for context-aware scene detection — the AI sees the whole game, not just short clips
- **7 Semantic Types** (all analyzed automatically):
  - `action` — Combat, kills, intense gameplay, close calls
  - `funny` — Fails, glitches, unexpected humor, comedic timing
  - `clutch` — 1vX situations, comebacks, last-second wins
  - `wtf` — Unexpected events, "wait what?" moments, random chaos
  - `epic_fail` — Embarrassing deaths, tragic blunders, game-losing mistakes
  - `hype` — Celebrations, "LET'S GO" energy, peak excitement
  - `skill` — Trick shots, IQ plays, advanced mechanics, impressive techniques

### 🎙️ Subtitle Generation

- **Speech Mode**: Uses OpenAI Whisper to transcribe voice/commentary
- **AI Captions Mode**: AI-generated contextual captions for gameplay without voice
- **Caption Styles**:
  - Classic: `gaming`, `dramatic`, `funny`, `minimal`
  - **GenZ Mode** ✨: `genz` - Slang-heavy reactions ("bruh 💀", "no cap", "finna")
  - **Story Modes** ✨: Narrative-style captions
    - `story_news` - Professional esports broadcaster
    - `story_roast` - Sarcastic roasting commentary
    - `story_creepypasta` - Horror/tension narrative
    - `story_dramatic` - Epic cinematic narration
  - `auto` - Auto-match style to detected semantic type
- **PyCaps Integration**: Multiple visual templates including `hype`, `retro-gaming`, `neo-minimal`
- **AI Enhancement**: Semantic tagging and emoji suggestions (e.g., "HEADSHOT! 💀🔥")

### 🔊 AI Voiceover (Qwen3-TTS)

- **Voice Design Engine**: Powered by **Qwen3-TTS 1.7B-VoiceDesign** for creating unique voices from natural language descriptions
- **Dynamic Voice Generation**: AI automatically generates voice persona based on caption style + caption content
- **Style-Adaptive Voices**: Each caption style has a unique voice preset:
  - GenZ → Casual energetic voice with modern slang
  - Story News → Professional broadcaster
  - Story Roast → Sarcastic playful narrator
  - Story Creepypasta → Deep ominous voice with tension
  - Story Dramatic → Epic movie-trailer narrator
- **Natural Language Instructions**: Define voice characteristics via text prompts without needing reference audio
- **Ultra-Low Latency**: Local inference with FlashAttention 2 optimization
- **Multilingual Support**: Native support for 10+ languages including English, Chinese, Japanese, Korean
- **Smart Mixing**: Automatic ducking of game audio when voiceover plays

### ⚡ GPU-Accelerated Pipeline

- **Scene Detection**: Custom implementation using `decord` + PyTorch on GPU
- **Audio Analysis**: `torchaudio` on GPU for fast RMS and spectral flux calculation
- **Video Analysis**: GPU streaming via `decord` for stable motion estimation
- **Image Processing**: `cupy` (CUDA-accelerated NumPy) for blur and transforms
- **Rendering**: PyTorch + **NVENC** hardware encoder for ultra-fast rendering

### 📐 Smart Video Processing

- Scenes ranked by combined action score (audio 0.6 + video 0.4 weights)
- Configurable aspect ratio (default 9:16 for TikTok/Shorts/Reels)
- Smart cropping with optional blurred background for non-vertical footage
- Retry logic during rendering to avoid spurious failures

### 🛡️ Robust Fallback System

AutoShorts is designed to work even when optimal components fail:

| Component | Primary | Fallback |
| :--- | :--- | :--- |
| **Video Encoding** | NVENC (GPU) | libx264 (CPU) |
| **Subtitle Rendering** | PyCaps (styled) | FFmpeg burn-in (basic) |
| **AI Analysis** | OpenAI/Gemini API | Heuristic scoring (`local` mode) |
| **TTS Device** | GPU (6GB+ VRAM) | CPU Fallback (slower) |

---

## 📋 Requirements

### Hardware

- **NVIDIA GPU** with CUDA support (6GB+ VRAM recommended for Qwen3-TTS 1.7B)
- **NVIDIA Drivers** and **System RAM** (16GB+ recommended)

### Software

- Python 3.10
- FFmpeg 4.4.2 (for Decord compatibility)
- CUDA Toolkit with `nvcc` (for building Decord from source)
- System libraries: `libgl1`, `libglib2.0-0`

---

## 🚀 Installation

### Option 1: Makefile Installation (Recommended)

The Makefile handles everything automatically—environment creation, dependency installation, and building Decord with CUDA support.

```bash
git clone https://github.com/divyaprakash0426/autoshorts.git
cd autoshorts

# Run the installer (uses conda/micromamba automatically)
make install

# Setup environment variables
cp .env.example .env
# Edit .env and add your API keys (Gemini/OpenAI) 

# Activate the environment
overlay use .venv/bin/activate.nu    # For Nushell
# OR
source .venv/bin/activate            # For Bash/Zsh
```

The Makefile will:

1. Download micromamba if conda/mamba is not found
2. Create a Python 3.10 environment with FFmpeg 4.4.2
3. Install NV Codec Headers for NVENC support
4. Build Decord from source with CUDA enabled
5. Install all pip requirements

### Option 2: Docker (GPU Required)

**Prerequisite**: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) must be installed.

```bash
# Build the image
docker build -t autoshorts .

# Serve the web UI on http://localhost:8501 (default)
docker run -d --name autoshorts-ui \
    --gpus all \
    --shm-size=8g \
    -p 8501:8501 \
    -v $(pwd):/app \
    autoshorts
```

Mounting the whole project directory keeps settings (`.env`), inputs and
rendered clips on the host, so nothing is lost when the container is replaced.

To run the pipeline once over everything in `gameplay/` without the UI:

```bash
docker run --rm --gpus all --shm-size=8g \
    -e MODE=batch \
    -v $(pwd):/app \
    autoshorts
```

> **Note**: The `--gpus all` flag is essential for NVENC and CUDA acceleration.

> **Blackwell GPUs (RTX 50xx)**: these need CUDA 12.8 or newer. The Dockerfile
> pins the base image and the PyTorch wheels accordingly — an unpinned
> `pip install torch` resolves to a build without `sm_120` kernels and fails
> with `CUDA error: no kernel image is available for execution on the device`.

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

### Key Configuration Options

| Category | Variable | Description |
| :--- | :--- | :--- |
| **AI Provider** | `AI_PROVIDER` | `openai`, `gemini`, or `local` (heuristic-only, no API) |
| | `VIDEO_TYPE` | Content type preset (`gaming`, `podcasts`, `sports`, `educational`, etc.) used for universal clipping + caption style defaults |
| | `AI_ANALYSIS_ENABLED` | Enable/disable AI scene analysis |
| | `GEMINI_DEEP_ANALYSIS` | Gemini-only: upload full video for smarter scene detection (slower initial upload, better results) |
| | `OPENAI_MODEL` | Model for analysis (e.g., `gpt-5-mini`) |
| | `AI_SCORE_WEIGHT` | How much to weight AI vs heuristic (0.0-1.0) |
| **Semantic Analysis** | `SEMANTIC_TYPES` | All 7 types analyzed: `action`, `funny`, `clutch`, `wtf`, `epic_fail`, `hype`, `skill` |
| | `CANDIDATE_CLIP_COUNT` | Number of clips to analyze |
| **Subtitles** | `ENABLE_SUBTITLES` | Enable subtitle generation |
| | `SUBTITLE_MODE` | `speech` (Whisper), `ai_captions`, or `none` |
| | `CAPTION_STYLE` | Gaming styles + story/genz + universal styles like `podcast_quote`, `sports_playbyplay`, `educational_explainer`, `news_breaking`, or `auto` |
| | `PYCAPS_TEMPLATE` | Visual template for captions |
| **TTS Voiceover** | `ENABLE_TTS` | Enable Qwen3-TTS voiceover |
| | `TTS_LANGUAGE` | Language code (`en`, `zh`, `ja`, `ko`, `de`, `fr`, `ru`, `pt`, `es`, `it`) |
| | `TTS_VOICE_DESCRIPTION` | Natural language voice description (auto-generated if empty) |
| | `TTS_GAME_AUDIO_VOLUME` | Game audio volume when TTS plays (0.0-1.0, default 0.3) |
| | `TTS_VOICEOVER_VOLUME` | TTS voiceover volume (0.0-1.0, default 1.0) |
| **Video Output** | `TARGET_RATIO_W/H` | Aspect ratio (default 9:16) |
| | `SCENE_LIMIT` | Max clips per source video |
| | `MIN/MAX_SHORT_LENGTH` | Clip duration bounds (seconds) |

See `.env.example` for the complete list with detailed descriptions.

### Clip Selection

Which moments become clips. Applied *before* anything is rendered.

| Variable | UI (Settings →) | Default | Effect |
| :--- | :--- | :--- | :--- |
| `SCENE_LIMIT` | Core Settings → Scene limit | `4` | Clips per source video. Raise it for more coverage, then discard what you do not need |
| `ACTION_W_AUDIO` | Action Detection → Audio weight | `0.6` | How strongly loudness peaks (gunfire, shouting, impacts) drive selection |
| `ACTION_W_VIDEO` | Action Detection → Motion weight | `0.4` | How strongly frame-to-frame motion drives it. Only the ratio matters — raise it above the audio weight to favour visually busy moments over loud ones |

### Clip Length

| Variable | UI (Settings →) | Default | Effect |
| :--- | :--- | :--- | :--- |
| `CLIP_LENGTH_MODE` | Clip Length → Window length | `max` | `max` always takes the longest allowed window and lets dead-air removal tighten it, so runs are reproducible. `random` draws a length per clip, so reruns end at different points |
| `MAX_SHORT_LENGTH` | Clip Length → Max short length | `59` | Upper bound of the window. The detected scene may cap it earlier |
| `MIN_SHORT_LENGTH` | Clip Length → Min short length | `15` | How much source material is selected at minimum. Also filters out shorter scenes entirely |

### Dead Air Removal

Cuts stretches out of a rendered clip where nothing is said *and* nothing
happens, then stitches the rest back together. Runs *after* rendering and
*before* subtitles, so captions are transcribed from the final timeline.

| Variable | UI (Settings →) | Default | Effect |
| :--- | :--- | :--- | :--- |
| `REMOVE_SILENCE` | Dead Air → Cut out dead air | `true` | Master switch |
| `SILENCE_MIN_GAP` | Dead Air → Min gap to cut | `1.0` | Minimum pause length that gets removed. `0.5` gives the hard jump-cut style, `2.0` only strips long lulls |
| `SILENCE_MOTION_KEEP` | Dead Air → Protect motion above | `0.5` | Threshold in standard deviations above the video's average motion. **High** → only very busy moments are protected, so the clip follows the **voice**. **Low** → little motion already counts as protected, so the clip follows the **gameplay** |
| `SILENCE_PADDING` | Dead Air → Keep around speech | `0.15` | Breathing room kept around each word so cuts do not clip syllables |
| `SILENCE_MIN_RESULT` | Dead Air → Min length after cutting | `8.0` | Floor for the finished clip. Cutting stops here even if more dead air remains |

### Caption Layout

| Variable | UI (Settings →) | Default | Effect |
| :--- | :--- | :--- | :--- |
| `SUBTITLE_MAX_LINES` | Caption Layout → Max caption lines | `2` | Maximum number of lines a caption wraps to |
| `SUBTITLE_MIN_LINES` | Caption Layout → Min caption lines | `1` | Reserved lines, so captions do not jump vertically |
| `SUBTITLE_MAX_CHARS` | Caption Layout → Max chars per caption | `15` | Where text is split into the next caption. Small values give the fast word-by-word look |
| `SUBTITLE_MIN_CHARS` | Caption Layout → Min chars per caption | `10` | Avoids very short leftover captions |
| `SUBTITLE_OVERFLOW` | Caption Layout → When text does not fit | `exceed_lines` | `exceed_lines` adds another line, so the line limit is only a target. `exceed_width` keeps the line count and lets the last line run wider |
| `SUBTITLE_WIDTH_RATIO` | Caption Layout → Caption width ratio | `0.85` | How much of the frame width captions may use |
| `SUBTITLE_VERTICAL_ALIGN` | Caption Layout → Vertical position | `bottom` | `bottom`, `center` or `top` |
| `SUBTITLE_VERTICAL_OFFSET` | Caption Layout → Vertical offset | `-0.1` | Nudge away from the chosen edge |
| `PYCAPS_KEEP_SPLITTERS` | Caption Layout → Split long captions | `true` | Off shows a whole transcript block at once: exact SRT boundaries, but walls of text |

### How the settings interact

Four pairs are easy to confuse because they sound similar but act at different
stages of the pipeline.

**`ACTION_W_VIDEO` vs. `SILENCE_MOTION_KEEP`** — both weigh motion, but at
opposite ends. The action weights decide **where in the source** clips are
looked for, before rendering. The motion-keep threshold decides **what survives
inside** a clip, after rendering. Gameplay-heavy shorts want a high
`ACTION_W_VIDEO` *and* a low `SILENCE_MOTION_KEEP`; commentary-driven shorts
want the opposite.

**`MIN_SHORT_LENGTH` vs. `SILENCE_MIN_RESULT`** — the first governs how much
raw material is selected, the second how short the finished clip may end up.
They are deliberately separate: select generously, cut tightly. Lowering
`MIN_SHORT_LENGTH` to allow shorter finals would also make the pipeline accept
thinner source windows, and it filters scenes as well.

**`SUBTITLE_MAX_LINES` vs. `SUBTITLE_OVERFLOW`** — the line limit alone is only
a target. With the default `exceed_lines`, a generous `SUBTITLE_MAX_CHARS`
still spills onto extra lines. Set `SUBTITLE_OVERFLOW=exceed_width` for a hard
line limit.

**`MAX_SHORT_LENGTH` vs. the detected scene** — the maximum is an upper bound,
not a target. If scene detection finds a 27-second scene, that caps the window
regardless of a higher `MAX_SHORT_LENGTH`.

---

## 📖 Usage

1. **Place source videos** in the `gameplay/` directory
2. **Run the script**:

   ```bash
   python run.py
   ```

3. **Generated clips** are saved to `generated/<source video name>/`

### 🧭 Dashboard (Streamlit UI)

Every setting documented above is editable in the dashboard, which also manages
the input queue, starts jobs and previews the results.

```bash
streamlit run src/dashboard/About.py
```

In Docker the UI is the default entrypoint — see
[Option 2: Docker](#option-2-docker-gpu-required). Videos copied into
`gameplay/` from outside the UI appear in the queue automatically.

| About | Generate | Browse |
| :---: | :---: | :---: |
| ![About](assets/dashboard/dashboard_about.png) | ![Generate](assets/dashboard/dashboard_generate.png) | ![Browse](assets/dashboard/dashboard_browse.png) |

| Features | Settings | Roadmap |
| :---: | :---: | :---: |
| ![Features](assets/dashboard/dashboard_features.png) | ![Settings](assets/dashboard/dashboard_settings.png) | ![Coming Soon](assets/dashboard/dashboard_coming_soon.png) |

### Output Structure

One folder per source video, so clips from different recordings do not
interleave:

```text
generated/
└── video_name/
    ├── scene-0.mp4            # Rendered short clip
    ├── scene-0.words.json     # Whisper word-level timings
    ├── scene-0_sub.json       # Caption layout data
    ├── scene-0.ffmpeg.log     # Render log
    ├── scene-1.mp4
    └── ...
```

Rerunning a video clears its folder first, so you always get one consistent set
of clips rather than a mix of runs. Only files the pipeline writes (`scene-*`)
are removed; anything else you keep in that folder is left alone.

> Two source files whose names differ only by extension (`clip.mkv` and
> `clip.mp4`) share one output folder and overwrite each other.

---

## 🧪 Development

### Linting

```bash
pip install ruff
ruff check .
```

### Running Tests

```bash
pytest -q
```

> Tests mock GPU availability and can run in standard CI environments.

### Debug Variables

For faster iteration during development, you can skip expensive steps using these environment variables in your `.env`:

| Variable | Description |
| :--- | :--- |
| `DEBUG_SKIP_ANALYSIS=1` | Skip AI scene analysis (uses cached/heuristic scores) |
| `DEBUG_SKIP_RENDER=1` | Skip video rendering (useful for testing analysis only) |
| `DEBUG_RENDERED_CLIPS="path1:category,path2"` | Test with specific pre-rendered clips |

Example workflow for testing subtitles only:

```bash
# In .env
DEBUG_SKIP_ANALYSIS=1
DEBUG_SKIP_RENDER=1
DEBUG_RENDERED_CLIPS="generated/test_clip.mp4:action"
```

---

## 🔧 Troubleshooting

| Issue | Solution |
| :--- | :--- |
| **"CUDA not available"** | Ensure `--gpus all` (Docker) or CUDA toolkit is installed |
| **NVENC Error** | Falls back to `libx264` automatically; check GPU driver |
| **PyCaps fails** | Falls back to FFmpeg burn-in subtitles automatically |
| **Decord EOF hang** | Increase `DECORD_EOF_RETRY_MAX` or set `DECORD_SKIP_TAIL_FRAMES=300` |
| **API rate limits** | Switch to `gpt-5-mini` (10M free tokens/day) or use `local` provider |

---

## 🤝 Contributing & Roadmap

We love contributions! Whether you're fixing a bug, adding a feature, or improving documentation:

- Check out our **[Contributing Guide](CONTRIBUTING.md)** to get started.
- See the **[Roadmap](ROADMAP.md)** for our future plans (YOLO Auto-Zoom, Next-Gen TTS, etc.).

---

## 🙏 Acknowledgments

This project builds upon the excellent work of:

- **[artryazanov/shorts-maker-gpu](https://github.com/artryazanov/shorts-maker-gpu)** — Heuristics-based shorts maker
- **[Binary-Bytes/Auto-YouTube-Shorts-Maker](https://github.com/Binary-Bytes/Auto-YouTube-Shorts-Maker)** — Original concept and inspiration

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/divyaprakash0426)
