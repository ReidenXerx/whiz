# whiz

A handy CLI wrapper around [whisper-cli](https://github.com/ggerganov/whisper.cpp) (whisper.cpp) that fixes the two most common headaches:

1. **`failed to open 'large-v3'`** — whiz auto-discovers models across your filesystem and resolves friendly aliases (`turbo`, `large-v3`, `medium`) to the actual `.bin` file.
2. **whisper-cli choking on `.mov`/`.mp4`** — whiz extracts a 16 kHz mono WAV with ffmpeg and hands *that* to whisper-cli, then cleans up.
3. **No multi-speaker labels** — `whiz transcribe --speakers` runs true diarization via sherpa-onnx and emits labeled `Speaker A:` / `Speaker B:` output.

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
| `--translate` | off | Translate to English instead of transcribing |
| `--no-timestamps` | off | Strip timestamps from output |
| `--keep-wav` | off | Keep the intermediate extracted WAV |
| `--extra ...` | — | Extra flags passed verbatim to whisper-cli |
| `--dry-run` | off | Print the command without executing |

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
vad_threshold = 0.5
outputs = ["srt", "json"]
verbose = true
extra_args = []
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

# Known speaker count (more accurate)
whiz transcribe --speakers 2 meeting.mp4

# Tune clustering threshold when auto-detecting (smaller = more speakers)
whiz transcribe --speakers --cluster-threshold 0.8 call.m4a
```

This produces the normal whisper-cli outputs (SRT, JSON) plus two labeled files alongside the input:

- `*.speakers.srt` — SRT with `Speaker A: ...` per cue
- `*.speakers.txt` — readable dialogue transcript (`Speaker A (00:01:23): text`), consecutive same-speaker lines merged

When `--speakers` is set, whisper-cli VAD is disabled (sherpa-onnx handles speech segmentation).

## License

MIT © ReidenXerx
