# whiz

A handy CLI wrapper around [whisper-cli](https://github.com/ggerganov/whisper.cpp) (whisper.cpp) that fixes the common headaches so you can just say `whiz transcribe recording.mov` and get on with your day:

1. **`failed to open 'large-v3'`** — whiz auto-discovers models across your filesystem and resolves friendly aliases (`turbo`, `large-v3`, `medium`) to the actual `.bin` file.
2. **whisper-cli choking on `.mov`/`.mp4`** — whiz extracts a 16 kHz mono WAV with ffmpeg and hands *that* to whisper-cli, then cleans up.
3. **No multi-speaker labels** — for video inputs, speaker diarization and on-screen frame capture are **on by default** (opt out with `--no-speakers` / `--no-screenshots`). whiz runs true diarization via sherpa-onnx and emits labeled `Speaker A:` / `Speaker B:` output (or real names — see [Speaker naming](#naming-speakers)). Diarization results are cached so re-tuning names/thresholds via `whiz merge` is instant.
4. **VAD model download moved** — whiz auto-downloads the current Silero VAD model from the new `ggml-org/whisper-vad` repo when VAD is enabled and no model is found.

Zero runtime dependencies — pure Python 3.11+ stdlib. Speaker diarization requires the optional `sherpa-onnx` package (see below).

## Requirements

- Python ≥ 3.11
- [`whisper-cli`](https://github.com/ggerganov/whisper.cpp) on `PATH` (e.g. `brew install whisper-cpp`)
- [`ffmpeg`](https://ffmpeg.org) on `PATH` (e.g. `brew install ffmpeg`) — only needed for video inputs
- At least one ggml Whisper model (see below)

## Install

```bash
pipx install git+https://github.com/ReidenXerx/whiz.git
```

Or from a clone:

```bash
git clone https://github.com/ReidenXerx/whiz.git
cd whiz
pipx install .
```

Then make sure a model is available — download one from the official whisper.cpp HuggingFace repo:

```bash
whiz models download turbo      # ggml-large-v3-turbo-q5_0.bin — fast & accurate
```

## Quick start

```bash
# Auto-picks the best available model, extracts audio if it's a video,
# and (for video) auto-enables speaker diarization + on-screen screenshots.
# Writes SRT + JSON + labeled speakers.srt/.txt + frames alongside the input.
whiz transcribe ~/Desktop/recording.mov

# Opt out of the video defaults if you don't need them
whiz transcribe --no-speakers --no-screenshots recording.mov

# Be explicit
whiz transcribe -m turbo -l en --outputs srt,vtt recording.mp4

# See what it would run, without running it
whiz transcribe --dry-run recording.mov

# Plain text, no timestamps (audio input — no auto screenshots/speakers)
whiz transcribe --outputs txt --no-timestamps meeting.m4a
```

> **Video inputs** auto-enable `--screenshots` and `--speakers` (auto-detect) so you get a labeled transcript plus per-segment frames for AI analysis / HTML output without extra flags. Pass `--no-screenshots` and/or `--no-speakers` to opt out. Audio inputs are unaffected.

## Commands

### `whiz transcribe <file>`

Transcribe an audio or video file.

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model` | auto-pick best | Model alias (`turbo`, `large-v3`, `medium`) or path |
| `-o, --output` | alongside input | Output base path (no extension) |
| `--outputs` | `srt,json` | Comma-separated: `txt,srt,vtt,json,json-full,csv,lrc,html` (`html` requires `--speakers`) |
| `-l, --language` | `auto` | Spoken language code or `auto` |
| `-t, --threads` | auto (`min(8, cores)`) | CPU threads |
| `--vad` / `--no-vad` | on | Enable/disable voice activity detection |
| `--vad-threshold` | `0.5` | VAD threshold |
| `--no-auto-vad-download` | off | Don't auto-download the Silero VAD model when VAD is enabled and missing |
| `--translate` | off | Translate to English instead of transcribing |
| `--no-timestamps` | off | Strip timestamps from output |
| `--print-progress` | on (TTY) | Print whisper-cli progress; default on when stderr is a TTY, off otherwise |
| `--no-progress` | off | Disable whisper-cli progress passthrough (forces `-np`) |
| `--keep-wav` | off | Keep the intermediate extracted WAV |
| `--speakers [N]` | on (video) | Enable speaker diarization; optional integer = known speaker count, omit = auto-detect. Auto-enabled for video inputs |
| `--no-speakers` | off | Disable the auto-enabled diarization for video inputs (opt out) |
| `--cluster-threshold` | `0.9` | Diarization clustering threshold when auto-detecting (larger = fewer speakers) |
| `--name-speakers` | off | After transcription, interactively prompt to name each detected speaker |
| `--speakers-names Alice,Bob,...` | off | Non-interactive speaker names assigned by total talk time (most talkative first) |
| `--screenshots` | on (video) | For video inputs, extract one on-screen frame per segment into `<stem>.frames/` + write `<stem>.frames.json` (for AI analysis / HTML output). Auto-enabled for video inputs |
| `--no-screenshots` | off | Disable the auto-enabled on-screen frame extraction for video inputs (opt out) |
| `--screenshot-width` | `1280` | Frame width in pixels (0 = native resolution) |
| `--no-voice-profiles` | off | Don't compute voice-profile embeddings or auto-match/save speaker profiles this run |
| `--resume` | off | Skip whisper-cli transcription if its JSON output already exists and go straight to diarization + merge |
| `--extra ...` | — | Extra flags passed verbatim to whisper-cli |
| `--dry-run` | off | Print the command without executing |

### `whiz merge <file>`

Re-run only diarization + the merge against an existing whisper JSON, skipping the expensive transcription. Lets you tune speaker count / threshold / names cheaply after a first run. Diarization results are cached in `<file>.wav.diar.json`, so a second `whiz merge` with the same `--speakers`/`--cluster-threshold` reuses the cache and skips the ~3 min embedding pass — only the cheap merge step runs. Changing either parameter re-runs diarization and overwrites the cache.

| Flag | Default | Description |
|------|---------|-------------|
| `--json` | auto-find | Explicit path to the whisper JSON |
| `--speakers [N]` | on (video) | Known speaker count; omit = auto-detect. Auto-enabled for video inputs |
| `--no-speakers` | off | Disable the auto-enabled diarization for video inputs (opt out) |
| `--cluster-threshold` | `0.9` | Clustering threshold when auto-detecting (larger = fewer speakers) |
| `--name-speakers` | off | Interactively prompt to name each detected speaker |
| `--speakers-names Alice,Bob,...` | off | Non-interactive speaker names assigned by total talk time (most talkative first) |
| `--screenshots` | on (video) | Re-extract on-screen frames per segment into `<stem>.frames/` + write `<stem>.frames.json`. Auto-enabled for video inputs |
| `--no-screenshots` | off | Disable the auto-enabled on-screen frame extraction for video inputs (opt out) |
| `--screenshot-width` | `1280` | Frame width in pixels (0 = native resolution) |
| `--no-voice-profiles` | off | Don't compute voice-profile embeddings or auto-match/save speaker profiles this run |
| `--outputs` | `srt,json` | Comma-separated output formats; add `html` for a self-contained transcript (requires `--speakers`) |

### `whiz models list`

Show all ggml models discovered in the search directories.

### `whiz models download <name>`

Download a model from `huggingface.co/ggerganov/whisper.cpp`. Accepts short names:
`turbo`, `large-v3`, `medium`, `small`, `base`, `tiny`, or full filenames like
`ggml-large-v3-turbo-q5_0.bin`. Default destination: `~/.cache/whisper/`.

```bash
whiz models download turbo
whiz models download ggml-large-v3.bin
whiz models download large-v3 --dest ~/models
```

### `whiz models known`

List the canonical set of whisper.cpp model filenames.

### `whiz models download-vad [version]`

Download the Silero VAD model. The VAD model moved from `ggerganov/whisper.cpp` to
a separate repo `ggml-org/whisper-vad` with versioned filenames. Default destination:
`~/.cache/whisper/`.

```bash
whiz models download-vad            # ggml-silero-v5.1.2.bin (default)
whiz models download-vad v6.2.0     # specific version
```

### `whiz models download-diarization`

Download the diarization models (~90 MB total): a pyannote segmentation model and a
3D-Speaker embedding extractor. Default destination: `~/.cache/whiz/diarization/`.

```bash
whiz models download-diarization
```

### `whiz config show`

Print current config and the model search directories.

### `whiz config edit`

Open the config file (`~/.config/whiz/config.toml`) in `$EDITOR`.

### `whiz config set KEY=VALUE`

Set a value persistently:

```bash
whiz config set model=turbo
whiz config set threads=8
whiz config set vad=false
whiz config set outputs=srt,txt
whiz config set cluster_threshold=0.95
```

## Configuration

Config lives at `~/.config/whiz/config.toml` (created on first `config edit`/`set`):

```toml
model = "turbo"
model_dirs = ["/Users/me/models"]
whisper_cli = ""
ffmpeg = ""
threads = 8
language = "auto"
vad = true
vad_model = ""
vad_threshold = 0.5
outputs = ["srt", "json"]
verbose = true
extra_args = []
# --- Speaker diarization ---
diarize = false
num_speakers = 0
cluster_threshold = 0.9
diarization_segmentation_model = ""
diarization_embedding_model = ""
# --- AI analysis (Ollama / OpenAI-compatible) ---
ai_base_url = "http://localhost:11434/v1"
ai_model = ""
ai_api_key = ""
ai_max_frames = 50
# --- Speaker voice profiles ---
speaker_match_threshold = 0.8
save_voice_profiles = true
```

If `model` is empty, whiz auto-picks the best available model by this preference:

1. `large-v3-turbo-q5_0`
2. `large-v3-turbo`
3. `large-v3-turbo-q8_0`
4. `large-v3-q5_0`
5. `large-v3`
6. `medium-q5_0` → `medium` → `small-q5_0` → `small`

## Model search directories

whiz scans these for `ggml-*.bin` files:

- `~/.cache/whisper`
- `~/Library/Application Support/com.unspoken.app/WhisperModels` (macOS)
- `~/Library/Caches/whisper`
- `/usr/local/share/whisper`, `/opt/homebrew/share/whisper`, `/usr/share/whisper`
- Any dirs you add via `model_dirs` in config

## Why?

Because `whisper-cli --model large-v3 -f recording.mov` fails twice — once because
`large-v3` isn't a file path, and again because whisper-cli can't demux a `.mov`.
whiz handles both so you can just say `whiz transcribe recording.mov` and get on
with your day.

## Speaker diarization (multi-speaker labels)

whiz can label who spoke when on mono recordings (meetings, screen recordings) via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), which combines a pyannote segmentation model with speaker-embedding clustering.

### One-time setup

```bash
# 1. Install the optional dependency into whiz's environment
pipx inject whiz sherpa-onnx

# 2. Download the diarization models (~90 MB total)
whiz models download-diarization
```

### Usage

For video inputs, diarization is already on by default — `whiz transcribe recording.mov` labels speakers automatically (auto-detect). The examples below show the explicit forms when you want a known count, tuning, or opt-out:

```bash
# Video: speakers + screenshots are auto-on already
whiz transcribe recording.mov

# Opt out of diarization for a video
whiz transcribe --no-speakers recording.mov

# Known speaker count (more accurate — threshold is ignored)
whiz transcribe --speakers 2 meeting.mp4

# Tune clustering threshold when auto-detecting (larger = fewer speakers; default 0.9)
whiz transcribe --speakers --cluster-threshold 0.95 call.m4a

# Name the speakers interactively after transcription
whiz transcribe --speakers 4 --name-speakers meeting.mov

# Name speakers non-interactively (assigned by total talk time, most talkative first)
whiz transcribe --speakers 4 --speakers-names Alice,Bob,Carol,Dave meeting.mov
```

If diarization is auto-enabled but sherpa-onnx or its models aren't installed yet, whiz skips speaker labeling with a one-line hint (and still transcribes + captures screenshots) instead of crashing — run the one-time setup above to turn it on.

This produces the normal whisper-cli outputs (SRT, JSON) plus two labeled files alongside the input:

- `*.speakers.srt` — SRT with `Speaker A: ...` (or real names) per cue
- `*.speakers.txt` — readable dialogue transcript (`Speaker A (00:01:23): text`), consecutive same-speaker lines merged
- `*.wav.diar.json` — cached diarization result (reused by later `whiz merge` runs with the same `--speakers`/`--cluster-threshold`)

When `--screenshots` is set (video inputs only), whiz also writes:

- `<stem>.frames/` — one JPEG per segment (`seg0001.jpg` ...), taken at each segment's start timestamp
- `<stem>.frames.json` — machine manifest referencing frames by path with per-segment metadata `{index, start, end, speaker, text, frame}` (no image bytes; small, re-runnable)

For video inputs `--screenshots` is on by default; pass `--no-screenshots` to skip it. The frames manifest is the join key for AI analysis (`whiz analyze --vision`) and for the self-contained HTML transcript (which inlines frames as base64).

### HTML transcript

Add `html` to `--outputs` (or pass `--outputs html` to `whiz merge`) to write a self-contained `<stem>.speakers.html` alongside the input. Each segment is rendered as a color-coded cue with a timestamp link, the speaker label, and (when `--screenshots` was set) the on-screen frame inlined as a base64 `data:` URI — so the file is fully portable with no external image dependencies.

```bash
# Transcribe + diarize + capture frames + write HTML transcript
# (for video, --speakers and --screenshots are already on by default)
whiz transcribe --outputs srt,html recording.mov

# Add HTML to an existing run via merge (reuses diarization cache)
whiz merge --outputs html recording.mov
```

The HTML file can be large (one base64-encoded JPEG per segment), but it opens in any browser with no server and no missing images. Speaker colors are assigned deterministically by a hash of the speaker label.

When diarization is on, whisper-cli VAD is disabled (sherpa-onnx handles speech segmentation). If diarization is auto-enabled for a video but unavailable, VAD stays on so you still get a clean transcription.

**Tip:** if you know the speaker count, always pass `--speakers N`. It locks clustering to exactly N speakers and ignores the threshold — this is the single biggest accuracy lever.

### Naming speakers

There are two ways to name speakers:

- **Interactive** — pass `--name-speakers` and, after transcription + diarization, whiz shows one representative quote per detected speaker (the longest utterance — most identifying) and prompts for a real name. Blank input keeps the default `Speaker A` label.
- **Non-interactive** — pass `--speakers-names Alice,Bob,Carol,Dave` to name speakers in a single command. Names are assigned to speakers ordered by total speaking time (most talkative gets the first name). Extra names beyond the detected speaker count are ignored; speakers beyond the provided names keep their `Speaker A/B/C` labels.

Both can be combined: `--speakers-names` provides defaults that are shown in the `--name-speakers` prompt, so you can confirm or override each one. Real names replace the `Speaker A/B/C` labels in both `*.speakers.srt` and `*.speakers.txt`.

### Re-tuning without re-transcribing: `whiz merge`

`whiz merge` re-runs only diarization + the merge against an existing whisper
JSON, so you can try different speaker counts / thresholds / names without
redoing the expensive transcription:

```bash
# Re-diarize with a known count, reusing the prior whisper JSON
whiz merge --speakers 4 recording.mov

# Video: diarization + screenshots are auto-on via merge too
whiz merge recording.mov

# Opt out of the video defaults at merge time
whiz merge --no-speakers --no-screenshots recording.mov

# Name speakers at merge time too
whiz merge --speakers 4 --name-speakers recording.mov

# Name speakers non-interactively (instant — diarization cache is reused)
whiz merge --speakers 4 --speakers-names Alice,Bob,Carol,Dave recording.mov
```

## Speaker voice profiles (cross-recording recognition)

When you name a speaker (with `--name-speakers` or `--speakers-names`), whiz can save a **voice profile**: a fixed-size embedding vector for that speaker's audio, computed with the same sherpa-onnx embedding extractor used for diarization. On later recordings, each detected cluster's embedding is compared (cosine similarity) to the stored profiles and a name is auto-assigned when the best match exceeds `speaker_match_threshold` (default `0.8`).

Profiles live at `~/.config/whiz/speakers/<Name>.json` (one file per name, inspectable and easy to delete). whiz saves a profile automatically whenever a speaker receives a real name — so the first time you transcribe a meeting with `--speakers-names Alice,Bob,Carol,Dave`, those four voice profiles are stored; the next recording with the same people is labeled automatically, no flags needed.

```bash
# First recording: name speakers explicitly — profiles are saved automatically
whiz transcribe --speakers 4 --speakers-names Alice,Bob,Carol,Dave meeting1.mov

# Later recording: speakers auto-matched from stored profiles, no flags needed
whiz transcribe --speakers 4 meeting2.mov

# See what's stored
whiz speakers list

# Check how a recording matches before committing (dry run)
whiz speakers match meeting2.mov --speakers 4

# Remove a profile (e.g. someone left the team)
whiz speakers forget Dave
```

Naming precedence when profiles exist: voice-profile auto-match seeds the names, `--speakers-names` overrides them, and `--name-speakers` prompts interactively with the auto-matched names shown as defaults. Pass `--no-voice-profiles` to skip both auto-matching and profile saving for a run. Auto-matched clusters that don't reach the threshold keep their `Speaker A/B/C` labels for you to name manually.

The embedding pass reuses the diarization cache, so on a cached run profile matching adds only seconds. Tune the threshold with `whiz config set speaker_match_threshold=0.85` (higher = stricter, fewer auto-assignments) or disable saving with `whiz config set save_voice_profiles=false`.

## AI analysis (summary, action items, vision)

`whiz analyze` sends a prior transcript (and optionally on-screen frames) to a chat model via an OpenAI-compatible API ([Ollama](https://ollama.com) by default). It produces a markdown analysis (`.analysis.md`) alongside the input and prints the response to stdout. Requires a prior `whiz transcribe` of a video (which auto-produces speakers + screenshots) or an audio run with `--speakers` (and `--screenshots` for `--vision`).

### Setup

```bash
# 1. Tell whiz which model to use (text-only for --summary/--actions)
whiz config set ai_model=gpt-4o-mini        # or any Ollama / OpenAI-compatible model

# For vision (--vision), use a vision-capable model
whiz config set ai_model=llava              # or qwen2.5-vl, minicpm-v, etc.

# Optional: point at a different server / set an API key for cloud providers
whiz config set ai_base_url=http://localhost:11434/v1
whiz config set ai_api_key=your-key          # Ollama ignores this
```

### Usage

```bash
# Summary + action items (default when neither --summary nor --actions is set)
whiz analyze recording.mov

# Just a summary
whiz analyze recording.mov --summary

# Just action items
whiz analyze recording.mov --actions

# Freeform question (use {transcript} where the transcript should go)
whiz analyze recording.mov --prompt "What risks did the team raise? Transcript: {transcript}"

# Vision analysis: send on-screen frames to a vision model (requires a prior
# --screenshots run; frames are spread evenly, capped at ai_max_frames=50)
whiz analyze recording.mov --vision --summary
```

`--vision` requires a vision-capable model (`llava`, `qwen2.5-vl`, `minicpm-v`, `gpt-4o`, ...). whiz detects a text-only model rejecting images and prints a clear hint. Frames are base64-encoded only at send time, so the on-disk `.frames.json` manifest stays small (paths only).

Output is written to `<stem>.analysis.md` (the prompt + the response) and the response is also printed to stdout.

### `whiz speakers list`

List stored speaker voice profiles (name, embedding dimension, creation time, file path). See [Speaker voice profiles](#speaker-voice-profiles-cross-recording-recognition).

### `whiz speakers forget <name>`

Delete a stored voice profile by name.

```bash
whiz speakers forget Alice
```

### `whiz speakers match <file>`

Run diarization on the given file and print, for each detected cluster, the cosine-similarity score against every stored profile plus the auto-assignment decision at the configured threshold. This is a dry run — it relabels nothing and saves nothing. Useful for tuning `speaker_match_threshold` or checking whether a recording's speakers are already known.

```bash
whiz speakers match recording.mov --speakers 4
```

## Testing

whiz ships a pytest suite covering the pure-Python modules (merge, models, screenshots, ai, diarize cache, profiles) — no sherpa-onnx, ffmpeg, or network required. Install the test extras and run:

```bash
pipx install --force --editable '.[test]'
pytest tests/
```

Tests isolate the filesystem (via `monkeypatch` and `tmp_path`) so host-installed models don't leak into model-discovery assertions. The suite runs in under a second.

## License

MIT © ReidenXerx
