# AutoShorts

Turns long gameplay recordings into vertical Shorts: it finds the moments with
the most action, cuts them into 9:16 clips, removes dead air and burns in
animated captions of what you said. Everything runs locally on an NVIDIA GPU,
no cloud services involved.

Based on [AutoShorts](https://github.com/divyaprakash0426/autoshorts) by
Divyaprakash Dhurandhar (MIT licence, see `LICENSE`), since reworked heavily.

## What it does

1. **Finds the action.** Every scene is scored by an audio and motion profile
   of the recording; the highest-scoring scenes become clips. No randomness -
   the same settings always pick the same moments.
2. **Places each clip.** The `balanced` length preset builds the clip around
   the scene's peak and may start before the scene, so the attempts leading up
   to a payoff stay in. It runs longer than the target only for clearly active
   material and shorter for clearly dead air; the end moves to the nearest lull.
3. **Renders 9:16** on the GPU, with a blurred copy of the frame as background.
4. **Cuts dead air** - stretches with neither speech nor motion - but always
   keeps the seconds right before the highlight.
5. **Adds captions.** Whisper transcribes the clip, PyCaps renders animated,
   word-highlighted captions.

Output keeps the source codec end to end: an AV1 recording gives AV1 Shorts,
an HEVC one HEVC Shorts, with no generation in another codec in between.

## Requirements

- NVIDIA GPU with a current driver (RTX 50xx / Blackwell supported)
- Docker with the NVIDIA container runtime

## Running it

```bash
touch .env                                   # PowerShell: New-Item -ItemType File .env
docker build -t andyxtreme/autoshorts:latest .
docker compose up -d
```

Open <http://localhost:8501>.

- **Generate** - drop recordings into `gameplay/` (or upload them), pick the
  number of clips and a caption template, start.
- **Settings** - everything else, with an explanation on each field. Changes
  are saved to `.env` right away.

Finished clips land in `generated/<recording name>/scene-1.mp4`,
`scene-2.mp4`, ... numbered in the order they occur in the recording. The
folder holds the clips and nothing else.

To process `gameplay/` without the UI:

```bash
docker compose run --rm -e MODE=batch autoshorts
```

## Settings worth knowing

| Setting | What it does |
|---|---|
| Length preset (`CLIP_LENGTH_PRESET`) | `balanced` (default), `loose` (more often longer), `fixed` (exact target length, starts inside the scene) |
| Target clip length (`TARGET_SHORT_LENGTH`) | What the presets aim for |
| Longest dynamic clip (`DYNAMIC_MAX_LENGTH`) | Upper limit for `balanced` and `loose` |
| Snap the end to a lull (`CLIP_END_SNAP`) | How far the end may move to land on a quiet moment |
| Cut out dead air (`REMOVE_SILENCE`) | Turn dead-air removal off entirely |
| Keep before the highlight (`SILENCE_PEAK_LEAD`) | Seconds before the peak that are never cut |
| Spoken language (`WHISPER_LANGUAGE`) | Caption language; empty detects it per clip |
| Output codec (`OUTPUT_VIDEO_CODEC`) | `auto` matches the source |

`.env.example` lists every setting with its default.

## Development

Tests run inside the container:

```bash
docker compose exec autoshorts python -m pytest -q tests
```
