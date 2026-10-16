# Architecture

How the pieces fit, and the rules they follow.

As of September 2026 whiz is one product: a transcription CLI. A recording goes
in — audio or video — and a labeled, named, frame-illustrated transcript plus
an optional AI analysis comes out, all on the user's machine.

Dictation used to live here too. It moved out and became
[mynah](https://github.com/ReidenXerx/mynah) — a product that runs from login
to shutdown, with a hotkey, a tray icon and its own release cycle. The things
this repo used to own for it (the segmentation engine, the tuning contract
`tuning/tuning.toml`, the golden corpus, the macOS Swift app, the vendored
whisper.cpp submodule) all moved with it, including their history. This
document describes what is left, and what the split owed the user on the way
out.

## The split, and what it owes

Dictation and transcription turned out to be two tools that happened to share
a speech engine. The handoff is deliberately gentle, because `whiz dictate` is
in people's muscle memory, their LaunchAgents and their shell history:

- `whiz dictate` (and the `d` alias) still parse and run — for a release or
  two — and print where it went (`cli.py`, `MYNAH_MOVED`), exiting nonzero.
- The `dictate_*` config keys stay readable in an existing `config.toml`:
  `save()` preserves keys it does not know, so nothing is stripped on a
  routine config write before Mynah's first run has imported them
  (`config.py`). Mynah reads them once and owns them from then on.
- `whiz upgrade` lost the extra and the LaunchAgent to look after; it now
  does the one thing it is still for — reinstall, re-inject the `diarize`
  extra when that one was already there, re-verify.

The tuning contract and the golden corpus that used to live under `tuning/`
were about dictation segmentation — utterance ends, energy gates, calibration.
Nothing in whiz segments audio at session speed anymore; transcription is a
batch pipeline whose speech segmentation is whisper-cli's VAD plus sherpa-onnx
diarization. The contract moved to mynah unchanged.

## The components

One Python package, ~6,000 lines, no compiled parts:

| Module | Role |
|---|---|
| `cli.py` | command surface: `transcribe`, `merge`, `analyze`, `models`, `config`, `speakers`, `upgrade`, the `dictate` stub; video-input defaults (speakers/screenshots/name-speakers auto-on) |
| `audio.py` | ffmpeg extraction to 16 kHz mono PCM WAV — the only container whisper-cli accepts |
| `models.py` | ggml model discovery, alias resolution (`turbo`, `large-v3`), download; the NS-15 preference order |
| `diarize.py` | sherpa-onnx diarization (pyannote segmentation + 3D-Speaker embedding), model download, the fingerprinted result cache |
| `merge.py` | maximum-temporal-overlap assignment of whisper segments to diarization speakers; labeled SRT/TXT/HTML emission |
| `profiles.py` | speaker voice profiles: cosine matching, sample-weighted merging, provenance |
| `screenshots.py` | one frame per segment into `<stem>.frames/` + the `frames.json` manifest (the join key for vision analysis and HTML) |
| `ai.py` | OpenAI-compatible chat API (Ollama by default): classifier probe, rolling-context map-reduce, retries, the Essentials section |
| `config.py` | `~/.config/whiz/config.toml`: flat-TOML read/write, tri-state consent key, foreign-key preservation |
| `ui.py` | rich terminal output; degrades to clean plain text when piped |

**The macOS app** (`macos/`, SwiftUI, since September 2026) is a front-end
for the CLI, not a second pipeline — a different app from the dictation one
that moved to mynah. It spawns `whiz transcribe` and reads its merged
stdout+stderr; every decision about audio, speakers and analysis stays in
the Python package. See *The pipe contract* below. Its other reads are
files and data the CLI already produces: the results view parses
`<stem>.frames.json` / `.speakers.srt` / `.speakers.txt` / `.analysis.md`
(the formats `tests/` pin), and Settings reads and writes `config.toml`
only through `whiz config show --json` and `whiz config set` (secrets via
`--stdin`, never argv) — the app keeps no settings copy of its own.

Everything shells out or calls optional dependencies rather than bundling:
whisper-cli and ffmpeg from PATH, sherpa-onnx as an optional extra installed
on demand (with consent), the chat model over HTTP. The only vendored artifact
left is nothing — the whisper.cpp submodule moved to mynah with the app that
needed it.

## The pipeline

```
transcribe ──► audio.py ──► whisper-cli ──► diarize.py ──► merge.py ──► outputs
   │                          (SRT/JSON)     (cacheable)    (labels)   (srt, speakers.*, html)
   │                                                                      │
   └── video input: screenshots.py ──► frames.json ──────────────┤
                                                                  ▼
analyze ──► ai.py (frames + transcript ──► map-reduce ──► .analysis.md)
```

`whiz merge` re-runs only the diarization + merge against an existing whisper
JSON, reusing the diarization cache, so tuning speaker count/threshold/names
after a first run is instant. `whiz analyze` consumes the artifacts either
path produced.

## The contracts

These are the rules a change must respect. Each is enforced by a test — a
rule without a pin is a wish.

**Model preference (NS-15).** Whisper models are used unquantized, always:
quantization corrupts transcription quality. Within each model class the
unquantized variant ranks first and every `-q*` variant is reachable only
when its own class's unquantized model is absent (preference is per-class,
never blocking a quantized model behind an unrelated class). `tiny` is
excluded from `KNOWN_MODELS`/`PREFERENCE` entirely. Pinned by
`tests/test_models.py` against `models.py:PREFERENCE` + alias resolution.
Quantized files stay runnable when named explicitly — informed use is never
blocked, it is just never the default.

**Voice-profile provenance.** A profile save knows where it came from:
`save_profile(..., auto_match=True)` (a machine match) may *create* a profile
marked `source: "auto"` but may never *merge into* an existing one — a chain
of self-confirming auto-matches would silently drift the stored centroid. A
human confirmation (`auto_match=False`) merges normally and upgrades an auto
profile to `source: "user"`. Pinned by `tests/test_profiles.py`. The same
contract covers the embedding dimension: a profile saved against a different
embedding model is discarded, not averaged.

**Diarization cache identity.** The cache (`<file>.wav.diar.json`) is keyed on
the parameters that produced it *and* a fingerprint of the input it was
produced from — a stale cache can never be reused against different audio or
a different `--speakers`/`--cluster-threshold` combination. Pinned by
`tests/test_diarize_cache.py`.

**Degraded-run contract.** When diarization cannot run (setup declined or
failed): an explicit `--speakers` degrades loudly and — when the run writes no
speaker-labeled artifacts at all — exits nonzero, so `|| alert` wrappers can
tell; an explicitly passed `--outputs html` is never silently dropped (generic
`Speaker` labels + a note line in the page); existing outputs carrying *real*
speaker labels are never clobbered by a degraded re-run, while existing
degraded files are refreshed in place; names passed via `--speakers-names` are
never silently discarded — the warning says so. Pinned by `tests/test_cli.py` /
`tests/test_merge.py`. A `whiz merge` whose only outcome is keeping existing
outputs is rc=0, not a false alarm.

**Consent on the setup path.** The one-time diarization setup (sherpa-onnx
install + model download) asks once on an interactive terminal and persists
the answer in the tri-state `auto_diarization_setup` key (`bool | None`;
unset is omitted from emitted TOML, never clobbered). Non-TTY runs proceed
without asking and without persisting; `--no-auto-diarization-setup`
short-circuits before any prompt. Pinned by `tests/test_cli.py`.

**Analysis contract.** Every `whiz analyze` run — any mode, single-call or
map-reduced — appends a dense `## Essentials` section to the same
`.analysis.md`; long inputs are chunked with a rolling-context map-reduce
(sliding window of prior partials, frames carried per-chunk so the model sees
a coherent visual timeline, not a bag of images); transient HTTP failures
(429/5xx, connection errors) retry with backoff, permanent errors surface
immediately with the server's body. Pinned by `tests/test_ai.py`.

**Extracted audio belongs to whoever made it.** A video's audio is extracted
to `<stem>.wav` beside it, and every output is named after that path
(`<stem>.wav.json`, the diarization cache, `whiz merge`'s lookup). whiz
writes its source's name into the WAV (`comment=whiz-extracted:<source>`,
RIFF INFO/ICMT) and overwrites, reuses or deletes a `<stem>.wav` only when it
carries that marker for the same source. Any other file there — the user's
own recording, a same-stem sibling's audio — is left alone: the audio goes to
a temp dir and `-of` keeps whisper's outputs named as usual. Cleanup runs on
every exit path, ^C included. Audio the diarization reader cannot take (m4a,
mp3, 44.1 kHz) is converted to a temp 16 kHz copy for diarization only.
Pinned by `tests/test_audio.py` / `tests/test_cli.py`.

**The pipe contract.** Piped (non-TTY), `ui.py` emits four machine lines on
stderr, which the macOS app parses (`CLIBackend.swift`):

    ▸ <phase label>
    ✓ <artifact label>: <absolute path>
    ? speaker-name: <label> | <quote>[ | <suggestion>]
    ~ progress: <whole percent>%

Fields substitute U+01C0 `ǀ` for ` | ` so the split stays exact; an artifact
path is always absolute, and the app accepts no other (model-written text is
never echoed into the pipe). The app sets `WHIZ_UI_PIPE=1`, and only then —
or on a TTY, or with an explicit `--name-speakers` — does an auto-enabled
prompt read answers from stdin (`-` declines); any other piped stdin belongs
to someone else. Prompts that cannot be answered over the pipe (the AI model
picker) never block: they take the recommended choice for the run and
persist nothing. The app reads the pipe to EOF before a run counts as
finished, decodes per line (a chunk can end mid-character), and gives the
child the Homebrew/pipx PATH a Finder launch lacks. Pinned on both sides:
`tests/test_ui_machine_contract.py` and `macos/Tests/WhizUITests/`.

**Config compatibility.** `config.toml` is written flat; `save()` preserves
every key it does not know — including the `dictate_*` keys Mynah has not
imported yet, and anything else a future reader puts there. A new key must
round-trip through TOML's value space (the reason `None` is omitted, not
emitted). Pinned by `tests/test_config.py`.

## Testing

Pure-Python suite, no sherpa-onnx, ffmpeg, models or network required; runs in
under a second:

```
uv run --extra test pytest tests/ -q
```

Filesystem isolation via `monkeypatch`/`tmp_path` keeps host-installed models
out of discovery assertions. The app's suite drives real `Process` plumbing
against stub scripts, no whiz install needed:

```
cd macos && swift test
```

Both run in CI (`.github/workflows/tests.yml`, Python 3.11–3.14 plus
`uv lock --check`, and the Swift suite + release build on macOS).

## Where the old contract went

The tuning contract, golden corpus, cross-implementation divergence rules
(trailing-silence policies, min-utterance gate placement, secondary VAD,
decoder thresholds) and the Rust core plan all described *dictation
segmentation across Python/Swift implementations*. That problem — keeping
independent implementations of one engine from drifting — is now mynah's; its
`tuning/tuning.toml` carried every value over byte-for-byte at the split. If
whiz ever grows a second implementation of the pipeline above, the pattern
moves with the need: constants in a pinned data file, no runtime reads, a
corpus that refuses to encode a divergence.
