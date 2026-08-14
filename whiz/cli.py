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


# ---------- transcribe ----------

def _build_transcribe_args(args: argparse.Namespace, config: cfg.Config) -> list[str]:
    """Assemble the whisper-cli argv."""
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
    out_flags = []
    for o in outputs:
        flag = OUTPUT_FLAGS.get(o)
        if flag is None:
            raise SystemExit(f"Unknown output format '{o}'. Valid: {', '.join(OUTPUT_FLAGS)}")
        out_flags.append(flag)

    # Output base path.
    of_flag: list[str] = []
    if args.output:
        of_flag = ["-of", str(Path(args.output).expanduser())]

    # Language.
    lang = args.language or config.language

    # VAD.
    vad_enabled = args.vad if args.vad is not None else config.vad
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
    if args.print_progress:
        cmd.append("-pp")
    if not config.verbose and not args.verbose:
        cmd.append("-np")
    if config.extra_args:
        cmd += config.extra_args
    if args.extra:
        cmd += args.extra

    return cmd, model_path, wav, in_path, keep_wav


def cmd_transcribe(args: argparse.Namespace) -> int:
    config = cfg.load()
    cmd, model_path, wav, in_path, keep_wav = _build_transcribe_args(args, config)

    print(f"Model:  {model_path}", file=sys.stderr)
    print(f"Input:  {in_path}", file=sys.stderr)
    if wav != in_path:
        print(f"Audio:  {wav}", file=sys.stderr)
    print(f"Run:    {' '.join(cmd)}", file=sys.stderr)
    print("-" * 60, file=sys.stderr)

    if args.dry_run:
        print("\nDRY-RUN: not executing whisper-cli.", file=sys.stderr)
        return 0

    proc = subprocess.run(cmd)
    rc = proc.returncode

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
    t.add_argument("--print-progress", action="store_true", help="Print progress")
    t.add_argument("--keep-wav", action="store_true", help="Keep the intermediate extracted WAV (default: deleted after)")
    t.add_argument("--no-auto-vad-download", action="store_true", help="Don't auto-download the Silero VAD model when VAD is enabled and missing")
    t.add_argument("--verbose", action="store_true", help="Verbose whisper-cli output")
    t.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="Extra flags passed verbatim to whisper-cli")
    t.add_argument("--dry-run", action="store_true", help="Print the command without running it")
    t.set_defaults(func=cmd_transcribe)

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