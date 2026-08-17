"""whiz CLI — a handy wrapper around whisper-cli.

Subcommands:
  whiz transcribe <file>   Transcribe an audio/video file.
  whiz models list         Show discovered models.
  whiz models download N   Download a model from HuggingFace.
  whiz config show         Show current config.
  whiz config edit         Open config in $EDITOR.
  whiz config set K=V      Set a config value.

Run `whiz transcribe -h` for transcription flags.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from whiz import __version__
from whiz import audio as aud
from whiz import config as cfg
from whiz import diarize as D
from whiz import merge as MR
from whiz import models as M

# whisper-cli output-format flags.
OUTPUT_FLAGS = {
    "txt": "-otxt",
    "srt": "-osrt",
    "vtt": "-ovtt",
    "json": "-oj",
    "json-full": "-ojf",
    "csv": "-ocsv",
    "lrc": "-olrc",
}


def _find_whisper_cli(configured: str = "") -> str:
    if configured:
        return configured
    found = shutil.which("whisper-cli")
    if not found:
        found = shutil.which("whisper")
    if not found:
        raise RuntimeError(
            "whisper-cli not found on PATH — install whisper.cpp "
            "(brew install whisper-cpp) or set whisper_cli in config."
        )
    return found


def _auto_threads() -> int:
    return min(8, os.cpu_count() or 4)


def _run_whisper_streaming(cmd: list[str]) -> subprocess.Popen:
    """Run whisper-cli, streaming its stdout/stderr line-by-line to our stderr.

    Each line is prefixed with elapsed time since the process started so long
    transcriptions show progress pacing. stderr is unbuffered so progress
    updates appear immediately. Returns the Popen object after completion.
    """
    import time

    start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        elapsed = time.monotonic() - start
        prefix = f"[{_fmt_elapsed(elapsed)}] "
        sys.stderr.write(prefix + line if line.endswith("\n") else prefix + line + "\n")
        sys.stderr.flush()
    proc.wait()
    return proc


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds as M:SS or H:MM:SS."""
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


# ---------- transcribe ----------

def _build_transcribe_args(args: argparse.Namespace, config: cfg.Config) -> list[str]:
    """Assemble the whisper-cli argv."""
    diarize_enabled = args.speakers is not None
    # Resolve model.
    model_ref = args.model or config.model
    if model_ref:
        model_path = M.resolve(model_ref, config)
        if model_path is None:
            raise SystemExit(
                f"Model '{model_ref}' not found. Run `whiz models list` to see what's available, "
                f"or `whiz models download {model_ref}` to fetch it."
            )
    else:
        model_path = M.pick_best(config)
        if model_path is None:
            raise SystemExit(
                "No models found. Run `whiz models download turbo` to get a fast one."
            )

    # Resolve input file.
    in_path = Path(args.file).expanduser()
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    # Extract audio if it's a video container.
    keep_wav = args.keep_wav
    if aud.needs_extraction(in_path):
        if args.dry_run:
            wav = aud.extract_audio(in_path, aud.find_ffmpeg(config.ffmpeg), dry_run=True)
            print(f"DRY-RUN: would extract audio -> {wav}")
        else:
            print(f"Input is a video container ({in_path.suffix}); extracting audio ...", file=sys.stderr)
            wav = aud.extract_audio(in_path, aud.find_ffmpeg(config.ffmpeg))
            print(f"Extracted -> {wav}", file=sys.stderr)
    elif aud.is_audio(in_path):
        wav = in_path
    else:
        # Unknown extension — let whisper-cli try; it may still work.
        print(f"Unrecognized extension {in_path.suffix}; passing directly to whisper-cli.", file=sys.stderr)
        wav = in_path

    # Threads.
    threads = args.threads if args.threads and args.threads > 0 else (config.threads or _auto_threads())

    # Outputs.
    outputs = args.outputs if args.outputs else config.outputs
    # When diarizing we need a parseable whisper output to merge against.
    # Force JSON (in addition to any user-requested formats) so we can parse segments.
    if diarize_enabled and "json" not in outputs and "json-full" not in outputs:
        outputs = list(outputs) + ["json"]
    out_flags = []
    for o in outputs:
        flag = OUTPUT_FLAGS.get(o)
        if flag is None:
            raise SystemExit(f"Unknown output format '{o}'. Valid: {', '.join(OUTPUT_FLAGS)}")
        out_flags.append(flag)

    # Output base path.
    of_flag: list[str] = []
    of_base = Path(args.output).expanduser() if args.output else wav.with_suffix("")
    if args.output:
        of_flag = ["-of", str(of_base)]

    # Language.
    lang = args.language or config.language

    # VAD. When diarizing, sherpa-onnx handles speech segmentation, so skip whisper-cli VAD.
    vad_enabled = (args.vad if args.vad is not None else config.vad) and not diarize_enabled
    if diarize_enabled:
        print("Diarization enabled; disabling whisper-cli VAD (sherpa-onnx handles segmentation).", file=sys.stderr)
    vad_flags: list[str] = []
    if vad_enabled:
        vad_flags = ["--vad", "-vt", str(args.vad_threshold if args.vad_threshold is not None else config.vad_threshold)]
        # Resolve the Silero VAD model. Auto-download if missing and not dry-run.
        vad_model_path = M.find_vad_model(config)
        if vad_model_path is None and not args.dry_run and not args.no_auto_vad_download:
            print("VAD enabled but no Silero VAD model found; downloading ggml-silero-vad.bin ...", file=sys.stderr)
            vad_model_path = M.ensure_vad_model(config, auto_download=True)
        if vad_model_path is not None:
            vad_flags += ["--vad-model", str(vad_model_path)]
        elif not args.dry_run:
            print("Warning: VAD enabled but no VAD model available; whisper-cli may fail. "
                  "Run `whiz models download-vad` or disable with --no-vad.", file=sys.stderr)
        elif args.dry_run and vad_model_path is None:
            print("DRY-RUN: no VAD model found; would download ggml-silero-vad.bin at run time.", file=sys.stderr)
            vad_flags += ["--vad-model", "<PATH-TO-VAD-MODEL>"]

    cmd = [
        _find_whisper_cli(config.whisper_cli),
        "-m", str(model_path),
        "-f", str(wav),
        "-t", str(threads),
        "-l", lang,
    ]
    cmd += out_flags
    cmd += of_flag
    cmd += vad_flags
    if args.translate:
        cmd.append("-tr")
    if args.no_timestamps:
        cmd.append("-nt")
    # Progress: whisper-cli uses -pp (print progress) and -np (no progress).
    # They are mutually exclusive. When stderr is a TTY we default to -pp so
    # the user sees live progress; otherwise -np keeps logs clean. --no-progress
    # forces -np even on a TTY; --print-progress forces -pp even off a TTY.
    progress_enabled = args.print_progress or (
        sys.stderr.isatty() and not args.no_progress
    )
    if progress_enabled:
        cmd.append("-pp")
    elif not config.verbose and not args.verbose:
        cmd.append("-np")
    if config.verbose or args.verbose:
        # verbose => let whisper-cli print everything; don't suppress.
        pass
    if config.extra_args:
        cmd += config.extra_args
    if args.extra:
        cmd += args.extra

    return cmd, model_path, wav, in_path, keep_wav, of_base


def _output_base_path(args, wav: Path) -> Path:
    """Determine the whisper-cli output base path (no extension)."""
    if args.output:
        return Path(args.output).expanduser()
    return wav.with_suffix("")


def _find_whisper_json(of_base: Path, wav: Path, of_passed: bool) -> Path | None:
    """Locate the whisper-cli JSON output.

    whisper-cli names outputs after the *input file stem*. When ``-of`` is NOT
    passed the input is the ``.wav`` file, so the JSON is ``<wav>.json`` (the
    ``.wav`` suffix is part of the stem, e.g. ``foo.wav.json``). When ``-of`` IS
    passed the JSON is ``<of_base>.json``.
    """
    candidates: list[Path] = []
    if of_passed:
        candidates.append(of_base.with_suffix(".json"))
        candidates.append(of_base.with_suffix(".json.json"))
    else:
        # whisper-cli appends the format extension to the full input path.
        candidates.append(Path(str(wav) + ".json"))
        candidates.append(of_base.with_suffix(".json"))
        candidates.append(of_base.with_suffix(".json.json"))
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _apply_speaker_names_list(
    merged: list[tuple[MR.WhisperSeg, str]],
    names: list[str],
) -> tuple[list[tuple[MR.WhisperSeg, str]], dict[str, str]]:
    """Assign names to speakers by total talk time (most talkative first).

    Returns the relabeled merged list and the {label: name} map used. Speakers
    beyond the provided names keep their default ``Speaker X`` label.
    ``names`` may be a single comma-separated token (``["Enric,Vadim"]``) or
    multiple tokens; both are flattened into a flat name list.
    """
    flat: list[str] = []
    for token in names:
        flat.extend(part.strip() for part in str(token).split(",") if part.strip())
    order = MR.speakers_by_talk_time(merged)
    name_map: dict[str, str] = {}
    for i, label in enumerate(order):
        if i < len(flat):
            name_map[label] = flat[i]
    return MR.relabel(merged, name_map), name_map


def _prompt_speaker_names(
    merged: list[tuple[MR.WhisperSeg, str]],
    default_names: dict[str, str] | None = None,
) -> dict[str, str]:
    """Interactively ask the user to name each detected speaker.

    Shows one representative quote per speaker (the longest utterance) and
    prompts for a real name. Returns a {"Speaker A": "Enric", ...} map.
    Blank input keeps the default label. When ``default_names`` is supplied
    (from ``--speakers-names``), the suggested name is shown in the prompt
    and used as the value if the user presses Enter.
    """
    speakers = MR.speakers_in_order(merged)
    quotes = MR.representative_quotes(merged)
    name_map: dict[str, str] = {}
    print("\n" + "=" * 60, file=sys.stderr)
    print("Name the speakers", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print("A representative quote is shown for each. Enter a real name", file=sys.stderr)
    print("(or press Enter to keep the default).", file=sys.stderr)
    for label in speakers:
        quote = quotes.get(label, "(no quote)")
        suggestion = (default_names or {}).get(label)
        print("\n" + "-" * 60, file=sys.stderr)
        print(f"{label} said:", file=sys.stderr)
        print(f'  "{quote}"', file=sys.stderr)
        prompt_text = f"Name for {label}"
        if suggestion:
            prompt_text = f"Name for {label} [{suggestion}]"
        try:
            name = input(prompt_text + ": ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            break
        if name:
            name_map[label] = name
        elif suggestion:
            name_map[label] = suggestion
    print("\n" + "-" * 60, file=sys.stderr)
    return name_map


def _write_labeled_outputs(
    merged: list[tuple[MR.WhisperSeg, str]],
    of_base: Path,
    name_speakers: bool = False,
    speakers_names: list[str] | None = None,
) -> tuple[Path, Path, dict[str, str]]:
    """Optionally relabel speakers, then write .speakers.srt and .speakers.txt.

    Returns the (srt_path, txt_path, name_map) used. Naming precedence:
    ``--speakers-names`` supplies a non-interactive list (assigned by total
    talk time); ``--name-speakers`` then prompts interactively, with the list
    names shown as defaults.
    """
    name_map: dict[str, str] = {}
    # Non-interactive names from --speakers-names are applied first.
    if speakers_names and merged:
        merged, list_map = _apply_speaker_names_list(merged, speakers_names)
        name_map.update(list_map)
    # Interactive prompt overrides/augments the list when both are given.
    if name_speakers and merged:
        interactive_map = _prompt_speaker_names(merged, default_names=name_map or None)
        if interactive_map:
            name_map.update(interactive_map)
            merged = MR.relabel(merged, name_map)
    labeled_srt = MR.format_labeled_srt(merged)
    dialogue = MR.format_dialogue_txt(merged)
    # Append (not Path.with_suffix) so dots in the stem like "...16.03.40"
    # aren't treated as a replaceable suffix.
    srt_out = Path(str(of_base) + ".speakers.srt")
    txt_out = Path(str(of_base) + ".speakers.txt")
    srt_out.write_text(labeled_srt + "\n", encoding="utf-8")
    txt_out.write_text(dialogue + "\n", encoding="utf-8")
    return srt_out, txt_out, name_map


def cmd_transcribe(args: argparse.Namespace) -> int:
    config = cfg.load()
    cmd, model_path, wav, in_path, keep_wav, of_base = _build_transcribe_args(args, config)
    diarize_enabled = args.speakers is not None

    print(f"Model:  {model_path}", file=sys.stderr)
    print(f"Input:  {in_path}", file=sys.stderr)
    if wav != in_path:
        print(f"Audio:  {wav}", file=sys.stderr)
    print(f"Run:    {' '.join(cmd)}", file=sys.stderr)
    print("-" * 60, file=sys.stderr)

    if args.dry_run:
        if diarize_enabled:
            num_sp = args.speakers if args.speakers else 0
            thr = args.cluster_threshold if args.cluster_threshold is not None else config.cluster_threshold
            D.run_diarization(wav, config, num_speakers=num_sp, threshold=thr, dry_run=True)
        print("\nDRY-RUN: not executing whisper-cli.", file=sys.stderr)
        return 0

    # --- Diarization path ---
    if diarize_enabled:
        num_sp = args.speakers if args.speakers else 0
        thr = args.cluster_threshold if args.cluster_threshold is not None else config.cluster_threshold
        diar_segments = D.run_diarization(wav, config, num_speakers=num_sp, threshold=thr)
        if not diar_segments:
            print("Warning: diarization produced no segments; falling back to unlabeled output.", file=sys.stderr)

    proc = _run_whisper_streaming(cmd)
    rc = proc.returncode

    # --- Merge diarization with whisper output ---
    if diarize_enabled and rc == 0:
        json_path = _find_whisper_json(of_base, wav, of_passed=bool(args.output))
        if not json_path.exists():
            print(f"Warning: expected whisper JSON output at {json_path} but it's missing; skipping merge.", file=sys.stderr)
        else:
            try:
                whisper_segs = MR.parse_whisper_json(json_path)
            except Exception as e:  # noqa: BLE001
                print(f"Warning: failed to parse {json_path}: {e}", file=sys.stderr)
                whisper_segs = []
            if whisper_segs and diar_segments:
                merged = MR.assign_speakers(whisper_segs, diar_segments)
                srt_out, txt_out, name_map = _write_labeled_outputs(
                    merged, of_base,
                    name_speakers=args.name_speakers,
                    speakers_names=args.speakers_names,
                )
                print(f"Wrote labeled SRT:  {srt_out}", file=sys.stderr)
                print(f"Wrote dialogue TXT: {txt_out}", file=sys.stderr)

    # Clean up the intermediate WAV unless asked to keep it.
    if wav != in_path and not keep_wav and wav.exists():
        try:
            wav.unlink()
            print(f"Removed intermediate {wav}", file=sys.stderr)
        except OSError:
            pass

    return rc


# ---------- models ----------

def cmd_models_list(args: argparse.Namespace) -> int:
    config = cfg.load()
    found = M.discover(config)
    if not found:
        print("No models found in:")
        for d in cfg.model_search_dirs(config):
            print(f"  {d}")
        print("\nDownload one with: whiz models download turbo")
        return 0
    print(f"{'ALIAS':<32} {'SIZE':>8}  PATH")
    for m in found:
        print(f"{m.alias:<32} {m.size_mb:>7.1f}M  {m.path}")
    return 0


def cmd_models_download(args: argparse.Namespace) -> int:
    config = cfg.load()
    dest = Path(args.dest).expanduser() if args.dest else None
    try:
        path = M.download(args.model, config, dest_dir=dest)
        print(f"\nDone. Use it with: whiz transcribe -m {path} <file>")
        return 0
    except FileExistsError as e:
        print(e)
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"Download failed: {e}", file=sys.stderr)
        return 2


def cmd_models_known(args: argparse.Namespace) -> int:
    for name in M.list_known():
        print(name)
    return 0


def cmd_models_download_vad(args: argparse.Namespace) -> int:
    config = cfg.load()
    dest = Path(args.dest).expanduser() if args.dest else None
    version = getattr(args, "version", "") or ""
    try:
        path = M.download_vad(config, dest_dir=dest, version=version)
        print(f"\nDone. VAD model at: {path}")
        return 0
    except FileExistsError as e:
        print(e)
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"Download failed: {e}", file=sys.stderr)
        return 2


def cmd_models_download_diarization(args: argparse.Namespace) -> int:
    dest = Path(args.dest).expanduser() if args.dest else None
    try:
        seg, emb = D.download_diarization_models(dest_dir=dest)
        print("\nDone. Enable with: whiz transcribe --speakers <file>")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"Download failed: {e}", file=sys.stderr)
        return 2


def cmd_merge(args: argparse.Namespace) -> int:
    """Re-run only diarization + merge against an existing whisper JSON.

    Lets you tune --speakers / --cluster-threshold without redoing the
    expensive whisper-cli transcription. The whisper JSON (produced by a
    prior `whiz transcribe --speakers` or `--outputs json`) is reused.
    """
    config = cfg.load()
    in_path = Path(args.file).expanduser()
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    # Resolve the audio (WAV) to diarize. Reuse an existing sibling WAV if the
    # transcribe run kept it; otherwise re-extract from the video.
    if aud.is_audio(in_path):
        wav = in_path
    elif aud.needs_extraction(in_path):
        wav = in_path.with_suffix(".wav")
        if not wav.exists():
            print(f"Extracting audio from {in_path} ...", file=sys.stderr)
            wav = aud.extract_audio(in_path, aud.find_ffmpeg(config.ffmpeg))
    else:
        wav = in_path

    # Locate the whisper JSON produced by a prior transcribe run.
    json_path = Path(args.json).expanduser() if args.json else _find_whisper_json(wav.with_suffix(""), wav, of_passed=False)
    if not json_path.exists():
        raise SystemExit(
            f"No whisper JSON found (looked for {json_path}).\n"
            "Run `whiz transcribe --speakers <file>` first to produce one, or pass --json <path>."
        )
    print(f"Whisper JSON:  {json_path}", file=sys.stderr)

    try:
        whisper_segs = MR.parse_whisper_json(json_path)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Failed to parse {json_path}: {e}")
    if not whisper_segs:
        raise SystemExit(f"No segments parsed from {json_path}.")
    print(f"Whisper segments: {len(whisper_segs)}", file=sys.stderr)

    # Diarization params.
    num_sp = args.speakers if args.speakers else 0
    thr = args.cluster_threshold if args.cluster_threshold is not None else config.cluster_threshold
    print(f"Diarize: num_speakers={num_sp or 'auto'} cluster_threshold={thr}", file=sys.stderr)

    diar_segments = D.run_diarization(wav, config, num_speakers=num_sp, threshold=thr)
    if not diar_segments:
        raise SystemExit("Diarization produced no segments; cannot merge.")

    merged = MR.assign_speakers(whisper_segs, diar_segments)

    of_base = json_path.with_suffix("")  # e.g. ...16.03.40.wav -> ...16.03.40
    # For the wav.json case, of_base should be the input stem without .json.
    if json_path.name.endswith(".wav.json"):
        of_base = json_path.with_name(json_path.name[: -len(".json")])  # ...16.03.40.wav
        of_base = of_base.with_suffix("")  # ...16.03.40

    # Speaker tally to stderr for quick tuning feedback (before relabeling).
    from collections import Counter
    tally = Counter(label for _, label in merged)
    print(f"Detected speakers: {len(tally)}", file=sys.stderr)
    for label, n in tally.most_common():
        print(f"  {label}: {n} segments", file=sys.stderr)

    srt_out, txt_out, name_map = _write_labeled_outputs(
        merged, of_base,
        name_speakers=args.name_speakers,
        speakers_names=args.speakers_names,
    )
    print(f"Wrote labeled SRT:  {srt_out}", file=sys.stderr)
    print(f"Wrote dialogue TXT: {txt_out}", file=sys.stderr)
    return 0


# ---------- config ----------

def cmd_config_show(args: argparse.Namespace) -> int:
    config = cfg.load()
    print(f"# {cfg.CONFIG_PATH}")
    print()
    for k, v in config.to_dict().items():
        if isinstance(v, list):
            print(f"{k} = {v}")
        elif isinstance(v, str) and v == "":
            print(f"{k} = \"\"")
        else:
            print(f"{k} = {v!r}")
    print()
    print("Model search dirs:")
    for d in cfg.model_search_dirs(config):
        marker = "+" if d.exists() else "-"
        print(f"  {marker} {d}")
    return 0


def cmd_config_edit(args: argparse.Namespace) -> int:
    config = cfg.load()
    path = cfg.save(config)
    editor = os.environ.get("EDITOR", "vi")
    subprocess.run([editor, str(path)])
    return 0


def _coerce(value: str, field_type: type):
    if field_type is bool:
        return value.lower() in {"1", "true", "yes", "on"}
    if field_type is int:
        return int(value)
    if field_type is float:
        return float(value)
    if field_type is list:
        return [v.strip() for v in value.split(",") if v.strip()]
    return value


def cmd_config_set(args: argparse.Namespace) -> int:
    config = cfg.load()
    assignment = args.assignment
    if "=" not in assignment:
        raise SystemExit("Expected KEY=VALUE (e.g. whiz config set threads=8)")
    key, _, value = assignment.partition("=")
    key = key.strip()
    if key not in cfg.Config.__dataclass_fields__:
        raise SystemExit(f"Unknown config key '{key}'. Valid: {', '.join(cfg.Config.__dataclass_fields__)}")
    field_type = cfg.Config.__dataclass_fields__[key].type
    coerced = _coerce(value.strip(), field_type)
    setattr(config, key, coerced)
    path = cfg.save(config)
    print(f"Set {key} = {coerced!r}")
    print(f"Saved to {path}")
    return 0


# ---------- argparse ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="whiz",
        description="A handy CLI wrapper around whisper-cli (whisper.cpp). "
                    "Auto-finds models, extracts audio from video, sensible defaults.",
    )
    p.add_argument("-V", "--version", action="version", version=f"whiz {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    # transcribe
    t = sub.add_parser("transcribe", aliases=["t"], help="Transcribe an audio/video file")
    t.add_argument("file", help="Input audio or video file")
    t.add_argument("-m", "--model", default="", help="Model alias/path (default: auto-pick best)")
    t.add_argument("-o", "--output", default="", help="Output base path (no extension); default: alongside input")
    t.add_argument("--outputs", default=None, help=f"Comma-separated output formats: {','.join(OUTPUT_FLAGS)}")
    t.add_argument("-l", "--language", default="", help="Spoken language code or 'auto' (default: config)")
    t.add_argument("-t", "--threads", type=int, default=0, help="CPU threads (default: auto)")
    t.add_argument("--vad", dest="vad", action="store_true", default=None, help="Force VAD on")
    t.add_argument("--no-vad", dest="vad", action="store_false", help="Disable VAD")
    t.add_argument("--vad-threshold", type=float, default=None, help="VAD threshold (default: 0.5)")
    t.add_argument("--translate", action="store_true", help="Translate to English instead of transcribing")
    t.add_argument("--no-timestamps", action="store_true", help="Suppress timestamps in output")
    t.add_argument("--print-progress", action="store_true", help="Print progress (force on; default on when stderr is a TTY)")
    t.add_argument("--no-progress", dest="no_progress", action="store_true", help="Disable whisper-cli progress passthrough (forces -np)")
    t.add_argument("--keep-wav", action="store_true", help="Keep the intermediate extracted WAV (default: deleted after)")
    t.add_argument("--no-auto-vad-download", action="store_true", help="Don't auto-download the Silero VAD model when VAD is enabled and missing")
    t.add_argument("--speakers", type=int, default=None, nargs="?", const=0, help="Enable speaker diarization via sherpa-onnx. Optional integer = known speaker count; omit = auto-detect")
    t.add_argument("--cluster-threshold", type=float, default=None, help="Diarization clustering threshold when auto-detecting (larger = fewer speakers; default 0.9)")
    t.add_argument("--name-speakers", action="store_true", help="After transcription, prompt to name each detected speaker (replaces Speaker A/B/C with real names)")
    t.add_argument("--speakers-names", dest="speakers_names", nargs="+", default=None, help="Non-interactive speaker names assigned by total talk time (most talkative first), e.g. --speakers-names Enric,Vadim,Thomas,Dziyana")
    t.add_argument("--verbose", action="store_true", help="Verbose whisper-cli output")
    t.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="Extra flags passed verbatim to whisper-cli")
    t.add_argument("--dry-run", action="store_true", help="Print the command without running it")
    t.set_defaults(func=cmd_transcribe)

    # merge
    mg = sub.add_parser("merge", help="Re-run diarization + merge against an existing whisper JSON (skip transcription)")
    mg.add_argument("file", help="Input audio/video file (used to find the whisper JSON and re-extract WAV if needed)")
    mg.add_argument("--json", default="", help="Explicit path to the whisper JSON (default: auto-find next to input)")
    mg.add_argument("--speakers", type=int, default=None, nargs="?", const=0, help="Known speaker count; omit = auto-detect")
    mg.add_argument("--cluster-threshold", type=float, default=None, help="Clustering threshold when auto-detecting (larger = fewer speakers; default 0.9)")
    mg.add_argument("--name-speakers", action="store_true", help="Prompt to name each detected speaker (replaces Speaker A/B/C with real names)")
    mg.add_argument("--speakers-names", dest="speakers_names", nargs="+", default=None, help="Non-interactive speaker names assigned by total talk time (most talkative first), e.g. --speakers-names Enric,Vadim,Thomas,Dziyana")
    mg.set_defaults(func=cmd_merge)

    # models
    mp = sub.add_parser("models", aliases=["m"], help="Manage whisper models")
    msub = mp.add_subparsers(dest="models_command", required=True)
    msub.add_parser("list", aliases=["ls"]).set_defaults(func=cmd_models_list)
    md = msub.add_parser("download", aliases=["dl"], help="Download a model from HuggingFace")
    md.add_argument("model", help="Model name, e.g. 'turbo', 'large-v3', or full 'ggml-large-v3-turbo-q5_0.bin'")
    md.add_argument("--dest", default="", help="Destination directory (default: ~/.cache/whisper)")
    md.set_defaults(func=cmd_models_download)
    msub.add_parser("known", help="List canonical known model names").set_defaults(func=cmd_models_known)
    mvd = msub.add_parser("download-vad", aliases=["vad"], help="Download the Silero VAD model (default: ggml-silero-v5.1.2.bin)")
    mvd.add_argument("version", nargs="?", default="", help="VAD version, e.g. 'v5.1.2', 'v6.2.0', or full filename (default: v5.1.2)")
    mvd.add_argument("--dest", default="", help="Destination directory (default: ~/.cache/whisper)")
    mvd.set_defaults(func=cmd_models_download_vad)
    mdiar = msub.add_parser("download-diarization", aliases=["diar"], help="Download diarization models (sherpa-onnx segmentation + embedding)")
    mdiar.add_argument("--dest", default="", help="Destination directory (default: ~/.cache/whiz/diarization)")
    mdiar.set_defaults(func=cmd_models_download_diarization)

    # config
    cp = sub.add_parser("config", aliases=["c"], help="View or edit configuration")
    csub = cp.add_subparsers(dest="config_command", required=True)
    csub.add_parser("show", aliases=["cat"]).set_defaults(func=cmd_config_show)
    csub.add_parser("edit", help="Open config in $EDITOR").set_defaults(func=cmd_config_edit)
    cs = csub.add_parser("set", help="Set a value: KEY=VALUE")
    cs.add_argument("assignment", help="KEY=VALUE, e.g. threads=8 or model=turbo")
    cs.set_defaults(func=cmd_config_set)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        rc = args.func(args)
    except SystemExit:
        raise
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    sys.exit(rc or 0)