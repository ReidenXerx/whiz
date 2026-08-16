# whiz

A handy CLI wrapper around [whisper-cli](https://github.com/ggerganov/whisper.cpp) (whisper.cpp) that fixes the common headaches so you can just say `whiz transcribe recording.mov` and get on with your day:

1. **`failed to open 'large-v3'`** — whiz auto-discovers models across your filesystem and resolves friendly aliases (`turbo`, `large-v3`, `medium`) to the actual `.bin` file.
2. **whisper-cli choking on `.mov`/`.mp4`** — whiz extracts a 16 kHz mono WAV with ffmpeg and hands *that* to whisper-cli, then cleans up.
3. **No multi-speaker labels** — `whiz transcribe --speakers` runs true diarization via sherpa-onnx and emits labeled `Speaker A:` / `Speaker B:` output (or real names — see [Speaker naming](#naming-speakers)).
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
# writes SRT + JSON alongside the input.
whiz transcribe ~/Desktop/recording.mov

# Be explicit
whiz transcribe -m turbo -l en --outputs srt,vtt recording.mp4

# See what it would run, without running it
whiz transcribe --dry-run recording.mov

# Plain text, no timestamps
whiz transcribe --outputs txt --no-timestamps meeting.m4a
```

## Commands

### `whiz transcribe <file>`

Transcribe an audio or video file.

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model` | auto-pick best | Model alias (`turbo`, `large-v3`, `medium`) or path |
| `-o, --output` | alongside input | Output base path (no extension) |
| `--outputs` | `srt,json` | Comma-separated: `txt,srt,vtt,json,json-full,csv,lrc` |
| `-l, --language` | `auto` | Spoken language code or `auto` |
| `-t, --threads` | auto (`min(8, cores)`) | CPU threads |
| `--vad` / `--no-vad` | on | Enable/disable voice activity detection |
| `--vad-threshold` | `0.5` | VAD threshold |
| `--no-auto-vad-download` | off | Don't auto-download the Silero VAD model when VAD is enabled and missing |
| `--translate` | off | Translate to English instead of transcribing |
| `--no-timestamps` | off | Strip timestamps from output |
| `--print-progress` | off | Print whisper-cli progress |
| `--keep-wav` | off | Keep the intermediate extracted WAV |
| `--speakers [N]` | off | Enable speaker diarization; optional integer = known speaker count, omit = auto-detect |
| `--cluster-threshold` | `0.9` | Diarization clustering threshold when auto-detecting (larger = fewer speakers) |
| `--name-speakers` | off | After transcription, prompt to name each detected speaker |
| `--extra ...` | — | Extra flags passed verbatim to whisper-cli |
| `--dry-run` | off | Print the command without executing |

### `whiz merge <file>`

Re-run only diarization + the merge against an existing whisper JSON, skipping the expensive transcription. Lets you tune speaker count / threshold / names cheaply after a first run.

| Flag | Default | Description |
|------|---------|-------------|
| `--json` | auto-find | Explicit path to the whisper JSON |
| `--speakers [N]` | auto-detect | Known speaker count; omit = auto-detect |
| `--cluster-threshold` | `0.9` | Clustering threshold when auto-detecting (larger = fewer speakers) |
| `--name-speakers` | off | Prompt to name each detected speaker |

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

```bash
# Auto-detect number of speakers
whiz transcribe --speakers recording.mov

# Known speaker count (more accurate — threshold is ignored)
whiz transcribe --speakers 2 meeting.mp4

# Tune clustering threshold when auto-detecting (larger = fewer speakers; default 0.9)
whiz transcribe --speakers --cluster-threshold 0.95 call.m4a

# Name the speakers interactively after transcription
whiz transcribe --speakers 4 --name-speakers meeting.mov
```

This produces the normal whisper-cli outputs (SRT, JSON) plus two labeled files alongside the input:

- `*.speakers.srt` — SRT with `Speaker A: ...` (or real names) per cue
- `*.speakers.txt` — readable dialogue transcript (`Speaker A (00:01:23): text`), consecutive same-speaker lines merged

When `--speakers` is set, whisper-cli VAD is disabled (sherpa-onnx handles speech segmentation).

**Tip:** if you know the speaker count, always pass `--speakers N`. It locks clustering to exactly N speakers and ignores the threshold — this is the single biggest accuracy lever.

### Naming speakers

Pass `--name-speakers` and, after transcription + diarization, whiz shows one
representative quote per detected speaker and prompts for a real name. The
quotes are the longest utterance per speaker (most identifying), and blank
input keeps the default `Speaker A` label. Real names then replace the
`Speaker A/B/C` labels in both `*.speakers.srt` and `*.speakers.txt`.

### Re-tuning without re-transcribing: `whiz merge`

`whiz merge` re-runs only diarization + the merge against an existing whisper
JSON, so you can try different speaker counts / thresholds / names without
redoing the expensive transcription:

```bash
# Re-diarize with a known count, reusing the prior whisper JSON
whiz merge --speakers 4 recording.mov

# Name speakers at merge time too
whiz merge --speakers 4 --name-speakers recording.mov
```

## License

MIT © ReidenXerx
