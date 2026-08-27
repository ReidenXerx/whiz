"""whiz CLI — transcription subcommands.

Subcommands:
  whiz transcribe <file>   Transcribe an audio/video file.
    --analyze              Chain into AI analysis after transcription.
  whiz merge <file>        Re-run diarization + merge against an existing JSON.
  whiz models list         Show discovered models.
  whiz models download N   Download a model from HuggingFace.
  whiz speakers list       List stored voice profiles.
  whiz analyze <file>      AI-analyze a prior transcript (+ frames).
  whiz dictate             System-wide voice dictation (toggle hotkey + floating indicator).
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
from whiz import screenshots as SC
from whiz import ai as AI
from whiz import profiles as P
from whiz import ui

# whisper-cli output-format flags. "html" is whiz-only (post-merge, not a
# whisper-cli flag) — handled in _write_labeled_outputs via merge.format_speakers_html.
OUTPUT_FLAGS = {
    "txt": "-otxt",
    "srt": "-osrt",
    "vtt": "-ovtt",
    "json": "-oj",
    "json-full": "-ojf",
    "csv": "-ocsv",
    "lrc": "-olrc",
    "html": "__whiz_html__",  # sentinel; filtered out before whisper-cli
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


def _outputs_include(args: argparse.Namespace, config: cfg.Config, fmt: str) -> bool:
    """True if ``fmt`` is in the requested/configured outputs (comma-split)."""
    raw = args.outputs if args.outputs else ",".join(config.outputs)
    return fmt in [o.strip() for o in raw.split(",") if o.strip()]


def _run_whisper_streaming(cmd: list[str]) -> subprocess.Popen:
    """Run whisper-cli, streaming its stdout/stderr line-by-line to our stderr.

    Each line is prefixed with elapsed time since the process started so long
    transcriptions show progress pacing. Output is styled via the ui module
    (dimmed timestamp prefix + muted content) and degrades to plain text when
    piped. Returns the Popen object after completion.
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
    with ui.streaming_progress(cmd) as write:
        for line in proc.stdout:
            elapsed = time.monotonic() - start
            write(line, elapsed)
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

def _video_auto_flags(args: argparse.Namespace, in_path: Path) -> tuple[bool, bool]:
    """Resolve effective (screenshots, speakers) for a video input.

    For video inputs whiz auto-enables screenshots and diarization so the user
    doesn't have to pass ``--screenshots`` / ``--speakers`` every time. The
    opt-out flags ``--no-screenshots`` / ``--no-speakers`` disable either.
    Explicit ``--speakers`` / ``--screenshots`` (the on-switches) still work
    and imply intent; this helper only adds defaults the user omitted.

    Returns (screenshots, speakers_auto) where ``speakers_auto`` is True when
    diarization should run via auto-detect (the caller still needs to honor an
    explicit ``args.speakers`` count). For non-video inputs both stay as-is.
    """
    is_video = aud.needs_extraction(in_path)
    screenshots = args.screenshots or (is_video and not getattr(args, "no_screenshots", False))
    # Diarization auto-enable: video + not explicitly disabled. An explicit
    # --speakers (args.speakers is not None) already enables it with a count.
    speakers_auto = is_video and not getattr(args, "no_speakers", False)
    return screenshots, speakers_auto


def _diarization_available(config: cfg.Config) -> bool:
    """True if sherpa-onnx + diarization models are ready (no heavy import).

    The segmentation/embedding model files are checked via the diarize module's
    finders (filesystem only); sherpa_onnx itself is imported lazily just to
    confirm the package is present. Used to gracefully skip auto-enabled
    diarization on machines that haven't run the one-time setup.
    """
    if D.find_segmentation_model(config) is None or D.find_embedding_model(config) is None:
        return False
    try:
        import sherpa_onnx  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def _name_speakers_enabled(args: argparse.Namespace, diarize_enabled: bool) -> bool:
    """Resolve whether the interactive speaker-naming prompt should run.

    Naming only makes sense when diarization actually ran. It's auto-enabled
    in that case (so you get prompted to label speakers without passing
    ``--name-speakers``); ``--no-name-speakers`` opts out. An explicit
    ``--name-speakers`` keeps working regardless.
    """
    if not diarize_enabled:
        return False
    if getattr(args, "no_name_speakers", False):
        return False
    return True


def _build_transcribe_args(args: argparse.Namespace, config: cfg.Config) -> list[str]:
    """Assemble the whisper-cli argv."""
    # Resolve input file first so video auto-enable can inform diarize_enabled.
    in_path = Path(args.file).expanduser()
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")
    screenshots, speakers_auto = _video_auto_flags(args, in_path)
    # diarize_enabled is True when the user passed --speakers (with or without
    # a count) OR when it's auto-enabled for a video input.
    diarize_enabled = args.speakers is not None or speakers_auto
    # Graceful fallback: if diarization was only auto-enabled (not explicitly
    # requested) but sherpa-onnx/models aren't available, skip it silently with
    # a hint instead of crashing. VAD then stays on and screenshots still run.
    if speakers_auto and args.speakers is None and not _diarization_available(config):
        ui.status("Speakers: diarization not available (sherpa-onnx or models missing); skipping speaker labels for this run.",
                  kind="hint",
                  detail="Enable with: pipx inject whiz sherpa-onnx && whiz models download-diarization")
        ui.muted("  Or silence this with: --no-speakers")
        diarize_enabled = False
        speakers_auto = False
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

    # Extract audio if it's a video container.
    keep_wav = args.keep_wav
    if aud.needs_extraction(in_path):
        if args.dry_run:
            wav = aud.extract_audio(in_path, aud.find_ffmpeg(config.ffmpeg), dry_run=True)
            print(f"DRY-RUN: would extract audio ->> {wav}")
        else:
            ui.phase("extracting audio")
            ui.kv("Video", in_path.name)
            wav = aud.extract_audio(in_path, aud.find_ffmpeg(config.ffmpeg))
            ui.kv("Audio", str(wav))
    elif aud.is_audio(in_path):
        wav = in_path
    else:
        # Unknown extension — let whisper-cli try; it may still work.
        ui.info(f"Unrecognized extension {in_path.suffix}; passing directly to whisper-cli.")
        wav = in_path

    # Threads.
    threads = args.threads if args.threads and args.threads > 0 else (config.threads or _auto_threads())

    # Outputs. Normalize to a list (the flag/config may be a comma string).
    raw_outputs = args.outputs if args.outputs else ",".join(config.outputs)
    outputs = [o.strip() for o in raw_outputs.split(",") if o.strip()]
    # We need a parseable whisper JSON to merge diarization against AND to
    # drive the per-segment screenshots path (even without diarization). Force
    # JSON (in addition to any user-requested formats) so we can parse segments.
    if (diarize_enabled or screenshots) and "json" not in outputs and "json-full" not in outputs:
        outputs = outputs + ["json"]
    out_flags = []
    for o in outputs:
        flag = OUTPUT_FLAGS.get(o)
        if flag is None:
            raise SystemExit(f"Unknown output format '{o}'. Valid: {', '.join(OUTPUT_FLAGS)}")
        if flag == "__whiz_html__":
            continue  # html is a whiz post-merge output, not a whisper-cli flag
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
        ui.info("Diarization enabled; disabling whisper-cli VAD (sherpa-onnx handles segmentation).")
    vad_flags: list[str] = []
    if vad_enabled:
        vad_flags = ["--vad", "-vt", str(args.vad_threshold if args.vad_threshold is not None else config.vad_threshold)]
        # Resolve the Silero VAD model. Auto-download if missing and not dry-run.
        vad_model_path = M.find_vad_model(config)
        if vad_model_path is None and not args.dry_run and not args.no_auto_vad_download:
            ui.info("VAD enabled but no Silero VAD model found; downloading ggml-silero-vad.bin ...")
            vad_model_path = M.ensure_vad_model(config, auto_download=True)
        if vad_model_path is not None:
            vad_flags += ["--vad-model", str(vad_model_path)]
        elif not args.dry_run:
            ui.status("Warning: VAD enabled but no VAD model available; whisper-cli may fail.",
                      kind="warn",
                      detail="Run `whiz models download-vad` or disable with --no-vad.")
        elif args.dry_run and vad_model_path is None:
            ui.muted("DRY-RUN: no VAD model found; would download ggml-silero-vad.bin at run time.")
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

    return cmd, model_path, wav, in_path, keep_wav, of_base, diarize_enabled, screenshots


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
    ``names`` may be a single comma-separated token (``["Alice,Bob"]``) or
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
    prompts for a real name. Returns a {"Speaker A": "Alice", ...} map.
    Blank input keeps the default label. When ``default_names`` is supplied
    (from ``--speakers-names``), the suggested name is shown in the prompt
    and used as the value if the user presses Enter.
    """
    speakers = MR.speakers_in_order(merged)
    quotes = MR.representative_quotes(merged)
    name_map: dict[str, str] = {}
    ui.header("whiz", "name the speakers")
    ui.muted("A representative quote is shown for each. Enter a real name")
    ui.muted("(or press Enter to keep the default).")
    for label in speakers:
        quote = quotes.get(label, "(no quote)")
        suggestion = (default_names or {}).get(label)
        ui.note("")
        ui.speaker_label_line(label)
        ui.muted(f'  "{quote}"')
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
    ui.note("")
    return name_map


def _extract_and_manifest_screenshots(
    video: Path,
    merged: list[tuple[MR.WhisperSeg, str]],
    of_base: Path,
    ffmpeg: str,
    width: int,
    dry_run: bool = False,
) -> tuple[Path, Path] | None:
    """Extract one frame per segment and write the .frames.json manifest.

    Frames go into ``<of_base>.frames/``; the manifest at ``<of_base>.frames.json``
    references frames by path only (never bytes) so it stays small and
    re-runnable. Returns (frames_dir, manifest_path) or None if there are no
    segments. Only valid for video inputs (the caller checks).
    """
    if not merged:
        return None
    frames_dir = SC.frames_dir_for(of_base)
    manifest_path = SC.frames_manifest_path(of_base)
    entries = SC.extract_segment_frames(
        video, merged, frames_dir,
        ffmpeg=ffmpeg,
        width=width,
        dry_run=dry_run,
    )
    SC.write_manifest(entries, frames_dir, manifest_path)
    ok = sum(1 for e in entries if e.frame)
    ui.muted(f"Extracted {ok}/{len(entries)} frames -> {frames_dir}")
    return frames_dir, manifest_path


def _save_named_profiles(
    name_map: dict[str, str],
    cluster_embeddings: dict[int, list[float]],
) -> None:
    """Save (or merge) a voice profile for each speaker that received a real name.

    ``name_map`` is keyed by ``Speaker A/B/...`` labels; we map those back to
    cluster ids via the merge module's letter ordering and persist the
    corresponding embedding under the chosen name. If a profile already exists
    for that name, ``P.save_profile`` merges the new embedding with the stored
    one via a sample-weighted running mean, so re-confirming a speaker across
    recordings makes their profile more accurate over time.
    """
    from whiz.merge import _SPEAKER_LETTERS

    label_to_cid: dict[str, int] = {
        f"Speaker {letter}": i for i, letter in enumerate(_SPEAKER_LETTERS)
    }
    saved = 0
    merged_count = 0
    for label, name in name_map.items():
        cid = label_to_cid.get(label)
        if cid is None or cid not in cluster_embeddings:
            continue
        # Don't save a profile whose "name" is just the default Speaker label.
        if not name or name.startswith("Speaker "):
            continue
        try:
            existed = P._profile_path(name).exists()
            path = P.save_profile(name, cluster_embeddings[cid], samples=1)
            saved += 1
            if existed:
                merged_count += 1
                ui.status(f"Merged voice profile: {name}", kind="ok", detail=str(path))
            else:
                ui.status(f"Saved voice profile: {name}", kind="ok", detail=str(path))
        except Exception as e:  # noqa: BLE001
            ui.status(f"Warning: could not save voice profile for {name}: {e}", kind="warn")
    if saved:
        note = f"Saved {saved} voice profile(s)"
        if merged_count:
            note += f" ({merged_count} merged with existing)"
        ui.muted(note + f" to {P.profiles_dir()}")


def _write_labeled_outputs(
    merged: list[tuple[MR.WhisperSeg, str]],
    of_base: Path,
    name_speakers: bool = False,
    speakers_names: list[str] | None = None,
    html: bool = False,
    frames_dir: Path | None = None,
    title: str = "whiz transcript",
    profile_names: dict[str, str] | None = None,
    cluster_embeddings: dict[int, list[float]] | None = None,
    save_profiles: bool = False,
) -> tuple[Path, Path, dict[str, str]]:
    """Optionally relabel speakers, then write .speakers.srt and .speakers.txt.

    Returns the (srt_path, txt_path, name_map) used. When ``html`` is True also
    writes a self-contained ``.speakers.html`` (frames inlined as base64 if
    ``frames_dir`` is given). Naming precedence:

    1. Voice-profile auto-match (``profile_names``) seeds defaults.
    2. ``--speakers-names`` supplies a non-interactive list (assigned by total
       talk time) that overrides profile matches.
    3. ``--name-speakers`` then prompts interactively, with the combined names
       shown as defaults.

    When ``save_profiles`` is True and ``cluster_embeddings`` is provided, a
    voice profile is saved for each speaker that ended up with a real name
    (i.e. not ``Speaker X``), so later recordings can auto-match them.
    """
    name_map: dict[str, str] = {}
    # 1. Voice-profile auto-match seeds the defaults.
    if profile_names and merged:
        name_map.update(profile_names)
        ui.info(f"Auto-matched {len(profile_names)} speaker(s) from voice profiles.")
        for lbl, nm in profile_names.items():
            ui.muted(f"  {lbl} -> {nm}")
    # 2. Non-interactive --speakers-names override profile matches.
    if speakers_names and merged:
        merged, list_map = _apply_speaker_names_list(merged, speakers_names)
        name_map.update(list_map)
    # 3. Interactive prompt overrides/augments when both are given.
    if name_speakers and merged:
        interactive_map = _prompt_speaker_names(merged, default_names=name_map or None)
        if interactive_map:
            name_map.update(interactive_map)
    # Apply the combined names to the merged list so labels reflect every
    # source (profile matches alone wouldn't relabel otherwise).
    if name_map and merged:
        merged = MR.relabel(merged, name_map)
    labeled_srt = MR.format_labeled_srt(merged)
    dialogue = MR.format_dialogue_txt(merged)
    # Append (not Path.with_suffix) so dots in the stem like "...16.03.40"
    # aren't treated as a replaceable suffix.
    srt_out = Path(str(of_base) + ".speakers.srt")
    txt_out = Path(str(of_base) + ".speakers.txt")
    srt_out.write_text(labeled_srt + "\n", encoding="utf-8")
    txt_out.write_text(dialogue + "\n", encoding="utf-8")
    if html:
        html_out = Path(str(of_base) + ".speakers.html")
        html_out.write_text(
            MR.format_speakers_html(merged, frames_dir=frames_dir, title=title),
            encoding="utf-8",
        )
    # Save voice profiles for speakers that received a real name.
    if save_profiles and cluster_embeddings and name_map:
        _save_named_profiles(name_map, cluster_embeddings)
    return srt_out, txt_out, name_map


def _run_diarize_or_fallback(wav: Path, config: cfg.Config, args: argparse.Namespace) -> list[D.DiarSegment]:
    """Run diarization, returning [] and a hint if sherpa-onnx/models are missing.

    An explicitly-requested diarization (``--speakers``) that fails because the
    runtime/models aren't installed surfaces a clear hint but still returns []
    so the caller can fall back to the unlabeled (or screenshots-only) path
    instead of crashing. A truly transient failure is re-raised.
    """
    num_sp = args.speakers if args.speakers else 0
    thr = args.cluster_threshold if args.cluster_threshold is not None else config.cluster_threshold
    try:
        diar_segments = D.run_diarization(wav, config, num_speakers=num_sp, threshold=thr)
    except RuntimeError as e:
        msg = str(e)
        if "sherpa_onnx" in msg or "models not found" in msg or "download-diarization" in msg:
            ui.status(f"Speakers: diarization unavailable — {msg.splitlines()[0]}",
                      kind="hint",
                      detail="Skipping speaker labels for this run. Enable with: pipx inject whiz sherpa-onnx && whiz models download-diarization")
            return []
        raise
    if not diar_segments:
        ui.status("Warning: diarization produced no segments; falling back to unlabeled output.", kind="warn")
    return diar_segments


def cmd_transcribe(args: argparse.Namespace) -> int:
    config = cfg.load()
    cmd, model_path, wav, in_path, keep_wav, of_base, diarize_enabled, screenshots = _build_transcribe_args(args, config)

    ui.header("whiz", f"transcription · v{__version__}")
    ui.kv("Model", model_path)
    ui.kv("Input", in_path)
    if wav != in_path:
        ui.kv("Audio", str(wav))
    if aud.needs_extraction(in_path):
        flags = []
        if screenshots:
            flags.append("screenshots=on" if not args.screenshots else "screenshots=on (explicit)")
        if diarize_enabled:
            flags.append("speakers=on" if args.speakers is None else f"speakers={args.speakers or 'auto'} (explicit)")
        if diarize_enabled and not getattr(args, "no_name_speakers", False):
            flags.append("name-speakers=on" + (" (explicit)" if args.name_speakers else ""))
        if flags:
            ui.info(f"Video input — auto-enabled: {', '.join(flags)}")
    if config.verbose or args.verbose:
        ui.muted(f"Run:    {' '.join(cmd)}")

    if args.dry_run:
        if diarize_enabled:
            num_sp = args.speakers if args.speakers else 0
            thr = args.cluster_threshold if args.cluster_threshold is not None else config.cluster_threshold
            D.run_diarization(wav, config, num_speakers=num_sp, threshold=thr, dry_run=True)
        ui.muted("\nDRY-RUN: not executing whisper-cli.")
        return 0

    # --- Resumability: skip transcription if a whisper JSON already exists ---
    # --resume lets you re-run `whiz transcribe` to redo diarization + merge
    # (e.g. with a different --speakers count) without re-running whisper-cli.
    # It's an ergonomic alias for `whiz merge` triggered from transcribe.
    json_path = _find_whisper_json(of_base, wav, of_passed=bool(args.output))
    resuming = bool(getattr(args, "resume", False) and json_path.exists())
    diar_segments: list[D.DiarSegment] = []
    if resuming:
        ui.info(f"--resume: found existing whisper JSON {json_path}; skipping transcription.")
        rc = 0
        # Diarization still runs so a new --speakers count / threshold takes
        # effect against the existing transcription.
        if diarize_enabled:
            diar_segments = _run_diarize_or_fallback(wav, config, args)
    else:
        # --- Diarization path ---
        if diarize_enabled:
            diar_segments = _run_diarize_or_fallback(wav, config, args)

        ui.phase("transcribing")
        proc = _run_whisper_streaming(cmd)
        rc = proc.returncode

    # --- Merge diarization with whisper output ---
    written: list[str] = []
    if diarize_enabled and rc == 0:
        json_path = _find_whisper_json(of_base, wav, of_passed=bool(args.output))
        if not json_path.exists():
            ui.status(f"Warning: expected whisper JSON output at {json_path} but it's missing; skipping merge.",
                      kind="warn")
        else:
            try:
                whisper_segs = MR.parse_whisper_json(json_path)
            except Exception as e:  # noqa: BLE001
                ui.status(f"Warning: failed to parse {json_path}: {e}", kind="warn")
                whisper_segs = []
            if whisper_segs and diar_segments:
                ui.phase("merging speakers")
                merged = MR.assign_speakers(whisper_segs, diar_segments)
                want_html = _outputs_include(args, config, "html")
                want_frames = screenshots and aud.needs_extraction(in_path)
                # Voice profiles: compute per-cluster embeddings and auto-match
                # against any stored profiles. The match seeds speaker names
                # (used as defaults); --speakers-names/--name-speakers can override.
                profile_names: dict[str, str] = {}
                cluster_embeddings: dict[int, list[float]] = {}
                if not args.no_voice_profiles:
                    try:
                        cluster_embeddings = P.compute_speaker_embeddings(wav, diar_segments, config)
                        if cluster_embeddings:
                            profile_names, matches = P.auto_assign_names(
                                cluster_embeddings, threshold=config.speaker_match_threshold,
                            )
                            profile_names = {k: v for k, v in profile_names.items() if v}
                    except Exception as e:  # noqa: BLE001
                        ui.status(f"Warning: voice-profile matching skipped: {e}", kind="warn")
                # Frames must be extracted before writing HTML so they can be
                # inlined; for the diarized path we extract after the labeled
                # outputs but before HTML if both are requested.
                srt_out, txt_out, name_map = _write_labeled_outputs(
                    merged, of_base,
                    name_speakers=_name_speakers_enabled(args, diarize_enabled),
                    speakers_names=args.speakers_names,
                    html=want_html and not want_frames,
                    title=in_path.name,
                    profile_names=profile_names or None,
                    cluster_embeddings=cluster_embeddings or None,
                    save_profiles=config.save_voice_profiles and not args.no_voice_profiles,
                )
                ui.wrote("Wrote labeled SRT", srt_out)
                ui.wrote("Wrote dialogue TXT", txt_out)
                written.append(str(srt_out))
                written.append(str(txt_out))
                # Apply the resolved names to the caller's merged list so the
                # screenshots manifest and the HTML pass carry real names too
                # (_write_labeled_outputs relabels a local copy only).
                if name_map:
                    merged = MR.relabel(merged, name_map)
                # Video screenshots: one frame per segment, using the relabeled
                # merged list so the manifest carries final speaker names.
                frames_dir = None
                if want_frames:
                    ui.phase("capturing frames")
                    width = args.screenshot_width if args.screenshot_width is not None else 1280
                    result = _extract_and_manifest_screenshots(
                        in_path, merged, of_base,
                        ffmpeg=aud.find_ffmpeg(config.ffmpeg),
                        width=width,
                        dry_run=args.dry_run,
                    )
                    if result is not None:
                        frames_dir = result[0]
                        ui.wrote("Wrote frames manifest", result[1])
                        written.append(str(result[1]))
                # Write HTML after frames exist so they can be inlined.
                if want_html and want_frames and frames_dir is not None:
                    ui.phase("writing HTML transcript")
                    _write_labeled_outputs(
                        merged, of_base,
                        name_speakers=False,
                        speakers_names=None,
                        html=True,
                        frames_dir=frames_dir,
                        title=in_path.name,
                    )
                    html_path = Path(str(of_base) + ".speakers.html")
                    ui.wrote("Wrote HTML transcript", html_path)
                    written.append(str(html_path))

    # --- Screenshots without diarization ---
    # Video input (or explicit --screenshots) without usable diarization: one
    # frame per whisper segment, labeled with a single generic speaker.
    if (
        screenshots
        and not diarize_enabled
        and rc == 0
        and aud.needs_extraction(in_path)
    ):
        json_path = _find_whisper_json(of_base, wav, of_passed=bool(args.output))
        if json_path.exists():
            try:
                whisper_segs = MR.parse_whisper_json(json_path)
            except Exception as e:  # noqa: BLE001
                ui.status(f"Warning: failed to parse {json_path}: {e}", kind="warn")
                whisper_segs = []
            if whisper_segs:
                ui.phase("capturing frames")
                unlabeled = [(seg, "Speaker") for seg in whisper_segs]
                width = args.screenshot_width if args.screenshot_width is not None else 1280
                result = _extract_and_manifest_screenshots(
                    in_path, unlabeled, of_base,
                    ffmpeg=aud.find_ffmpeg(config.ffmpeg),
                    width=width,
                    dry_run=args.dry_run,
                )
        if result is not None:
            ui.wrote("Wrote frames manifest", result[1])
            written.append(str(result[1]))

    # Clean up the intermediate WAV unless asked to keep it.
    if wav != in_path and not keep_wav and wav.exists():
        try:
            wav.unlink()
            ui.muted(f"Removed intermediate {wav}")
        except OSError:
            pass

    ui.summary(written)

    # Optional: chain into AI analysis after a successful transcription.
    # Runs the same auto-detect path as `whiz analyze <file>` so the user gets
    # summary+actions or an implementation plan without a second command.
    if getattr(args, "analyze", False) and rc == 0:
        from types import SimpleNamespace
        analyze_args = SimpleNamespace(
            file=str(in_path),
            model="",
            base_url="",
            api_key=None,
            max_frames=None,
            summary=False,
            actions=False,
            plan=False,
            prompt="",
            vision=getattr(args, "vision", False) or False,
            no_vision=getattr(args, "no_vision", False) or False,
        )
        ui.phase("analyzing (chained)")
        try:
            cmd_analyze(analyze_args)
        except SystemExit:
            # cmd_analyze raises SystemExit on missing transcript/model issues;
            # the transcription itself already succeeded, so don't surface that
            # as a hard failure — the hint was already printed.
            pass

    return rc


# ---------- models ----------

def cmd_models_list(args: argparse.Namespace) -> int:
    config = cfg.load()
    found = M.discover(config)
    if not found:
        ui.status("No models found in:", kind="warn")
        for d in cfg.model_search_dirs(config):
            ui.muted(f"  {d}")
        ui.info("Download one with: whiz models download turbo")
        return 0
    ui.table(
        "Discovered models",
        [("Alias", "left"), ("Size", "right"), ("Path", "left")],
        [[m.alias, f"{m.size_mb:.1f}M", str(m.path)] for m in found],
    )
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


# ---------- analyze ----------

def _analysis_output_path(of_base: Path) -> Path:
    return Path(str(of_base) + ".analysis.md")


def _recommend_model(models: list[str], prefer_vision: bool) -> int:
    """Pick the index of the recommended default model from ``models``.

    Heuristic: when ``prefer_vision`` is set, prefer a name suggesting a vision
    model (llava, vl, vision, minicpm-v, qwen2.5-vl, etc.). Otherwise prefer a
    name suggesting a strong text/coder model (gpt, qwen, llama, mistral, glm,
    deepseek, devstral, ...). Anything not tagged ':cloud' wins over cloud-tagged
    (local models respond faster and cost nothing). Falls back to 0.
    """
    if not models:
        return 0
    want_tokens = _VISION_TOKENS if prefer_vision else _TEXT_TOKENS
    best_idx = 0
    best_score = -1
    for i, name in enumerate(models):
        low = name.lower()
        is_cloud = ":cloud" in low
        score = 0
        if not is_cloud:
            score += 2
        if any(tok in low for tok in want_tokens):
            score += 5
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


# Substrings (lowercased) in a model name that signal vision capability. Used
# both by the model-recommend heuristic and the analyze-time vision gate so a
# single source of truth decides whether sending images is safe.
_VISION_TOKENS = ("vl", "vision", "llava", "minicpm-v", "qwen2.5-vl", "qwen-vl",
                  "qwen3-vl", "qwen3.5", "multimodal", "gpt-4o", "gpt-4-vision",
                  "llama-3.2-vision", "pixtral", "cogvlm", "internvl",
                  "phi-3.5-vision", "phi-3-vision", "gemma3", "gemma4",
                  "mistral-large-3", "minimax-m3", "kimi-k2.5", "kimi-k2.6",
                  "kimi-k2.7")
# Substrings that signal a strong text/coder model (non-vision). Kept narrow
# so cloud vision-capable models (qwen3.5, kimi-k2.6, gemma4, ...) are NOT
# misclassified as text-only.
_TEXT_TOKENS = ("deepseek", "devstral", "codestral", "coder", "gpt-oss")


def _looks_vision_capable(model: str) -> bool:
    """True if ``model``'s name suggests it can accept image inputs.

    This is a name heuristic only (no probing) so it's fast and offline. It errs
    on the side of "not vision" for ambiguous names so we never send images to a
    model that will reject them. The HTTP layer prints a clear hint if a text
    model still gets image content.
    """
    low = (model or "").lower()
    return any(tok in low for tok in _VISION_TOKENS)


def _resolve_vision(*, explicit_vision: bool, no_vision: bool, has_frames: bool, model: str) -> tuple[bool, str, str]:
    """Decide whether analysis should use vision (feed frames to the model).

    Returns ``(use_vision, kind, message)`` where ``kind`` is a ui status kind
    ("", "info", "warn", "hint") and ``message`` is a one-line explanation to show
    the user (empty when there's nothing worth surfacing).

    Priority:
      * ``--no-vision`` always disables (opt out), even if ``--vision`` was set.
      * ``--vision`` explicitly requests it and frames exist: always enable
        (a true user override — we don't second-guess the model name; the HTTP
        layer prints a clear rejection hint if the model actually rejects the
        images). If no frames manifest exists, fall back to text-only.
      * Otherwise: when frames exist AND the configured model looks vision-
        capable, auto-enable (info). When frames exist but the model looks
        text-only, stay text-only with a hint to switch models (we never auto-
        send images to a model that might reject them).
      * No frames: text-only, silently.
    """
    if no_vision:
        return False, "", ""
    if explicit_vision:
        # Explicit --vision is a user override: always send frames if they
        # exist. We don't second-guess the model name here (the HTTP layer's
        # _post_chat already prints a clear rejection hint if a text-only model
        # actually rejects the images). Only fall back to text-only when there
        # are no frames to send.
        if not has_frames:
            return False, "warn", "--vision requested but no frames manifest found; falling back to text-only."
        return True, "", ""
    if not has_frames:
        return False, "", ""
    if _looks_vision_capable(model):
        return True, "info", ("Frames found and '{m}' is vision-capable; auto-enabling "
                              "vision (use --no-vision to opt out).").format(m=model)
    return False, "hint", ("Frames found but '{m}' doesn't look vision-capable; staying "
                           "text-only. Run `whiz config set ai_model=llava` (or another "
                           "vision model) and re-analyze to use the frames.").format(m=model)


def _pick_model_interactive(config: cfg.Config, *, prefer_vision: bool) -> str | None:
    """List available Ollama models and let the user choose; persist to config.

    Returns the chosen model name (and saves it to ``config.ai_model`` via
    ``cfg.save``), or None if no models were reachable — in which case a hint is
    printed and the caller should exit cleanly.

    Each listed model is probed with a trivial chat completion because Ollama's
    ``/api/tags`` can list models that are retired server-side (the retirement
    only surfaces as an HTTP 410 at call time). Dead models are marked
    ``(unavailable)`` in the table; the recommended default is the highest-ranked
    model that actually responds, so we never silently save a dead model.
    """
    base_url = config.ai_base_url
    api_key = config.ai_api_key
    ui.phase("choosing AI model")
    ui.muted(f"querying {base_url} for available models ...")
    models = AI.list_ollama_models(base_url)
    if not models:
        ui.status("No AI model configured and no models found at the server.", kind="warn",
                  detail="Set one with:  whiz config set ai_model=llava\nOr start Ollama:  ollama serve")
        return None
    # Probe each model once. Cloud-tagged/retired models fail here; we mark them
    # and prefer a live one for the default.
    ui.muted(f"probing {len(models)} model(s) for availability ...")
    live: list[tuple[int, str]] = []  # (original_index, name)
    status: list[str] = []
    for i, name in enumerate(models):
        ok, _err = AI.probe_model(base_url, name, api_key)
        if ok:
            live.append((i, name))
            status.append("")
        else:
            status.append("(unavailable)")
    if not live:
        ui.status("None of the listed models responded to a probe.", kind="warn",
                  detail="Ollama listed models but every one failed a trivial chat call.\n"
                          "Cloud models may be retired server-side; pull a local one with `ollama pull llama3.1`.\n"
                          "Or set a model explicitly:  whiz config set ai_model=...")
        return None
    # Recommend the best live model (heuristic over the live subset).
    live_names = [n for _, n in live]
    rec_in_live = _recommend_model(live_names, prefer_vision=prefer_vision)
    rec_idx = live[rec_in_live][0]  # map back to the full-list index for display
    ui.header("whiz", "models")
    rows: list[list[object]] = []
    for i, name in enumerate(models):
        mark = "\u2190 recommended" if i == rec_idx else ""
        rows.append([i + 1, name, status[i] or mark])
    ui.table(
        f"{len(models)} model(s) listed, {len(live)} available",
        [("#", "right"), ("Model", "left"), ("", "left")],
        rows,
    )
    default_name = models[rec_idx]
    # Loop until the user picks a live model (or accepts the live default).
    while True:
        try:
            choice = input(f"Choose a model [1-{len(models)}] (default {rec_idx + 1} = {default_name}): ").strip()
        except EOFError:
            choice = ""
        if not choice:
            chosen = default_name
            chosen_idx = rec_idx
        else:
            chosen_idx = int(choice) - 1 if choice.isdigit() else -1
            if 0 <= chosen_idx < len(models):
                chosen = models[chosen_idx]
            else:
                # Accept a typed model name verbatim too.
                chosen = choice if choice in models else default_name
                chosen_idx = models.index(chosen) if chosen in models else rec_idx
                if chosen != default_name and choice not in models:
                    ui.status(f"'{choice}' isn't in the list; using default {default_name}.", kind="warn")
                    chosen = default_name
                    chosen_idx = rec_idx
        if status[chosen_idx]:
            # Marked unavailable (e.g. retired server-side).
            ui.status(f"{chosen} is unavailable: {status[chosen_idx].strip('()')}. Pick another.", kind="warn")
            continue
        break
    config.ai_model = chosen
    cfg.save(config)
    ui.status(f"Saved ai_model = {chosen}", kind="ok")
    return chosen


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze a prior transcript (and optionally frames) with an AI model.

    Loads the frames manifest if present (<stem>.frames.json) for both the
    transcript text and (with --vision) the frame images; otherwise loads the
    <stem>.speakers.txt transcript. Writes the prompt + response to
    <stem>.analysis.md and prints the response to stdout.

    Vision is **auto-enabled** when a frames manifest exists and the configured
    model looks vision-capable, so a video run followed by ``whiz analyze`` uses
    the frames without needing ``--vision``. ``--no-vision`` opts out, and a
    text-only model stays text-only with a hint (we never send images to a model
    that will reject them).
    """
    config = cfg.load()

    in_path = Path(args.file).expanduser()
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    of_base = in_path.with_suffix("")
    # For video inputs the manifest/transcript sit alongside, named after the
    # video stem (not the .wav). We just use the video stem directly.
    manifest_path = SC.frames_manifest_path(of_base)
    txt_path = Path(str(of_base) + ".speakers.txt")

    entries = SC.load_manifest(manifest_path)
    if entries:
        transcript = AI.transcript_text(entries)
        ui.info(f"Loaded frames manifest: {manifest_path} ({len(entries)} segments)")
    elif txt_path.exists():
        transcript = txt_path.read_text(encoding="utf-8")
        ui.info(f"Loaded transcript: {txt_path}")
    else:
        raise SystemExit(
            f"No transcript found. Looked for:\n  {manifest_path}\n  {txt_path}\n"
            "Run `whiz transcribe --speakers [--screenshots] <file>` first."
        )
    has_frames = entries is not None

    # Model picking. prefer_vision mirrors the effective vision intent: if the
    # user explicitly asked for --vision, or frames exist and they haven't opted
    # out with --no-vision, steer the interactive picker toward a vision model.
    explicit_vision = bool(getattr(args, "vision", False))
    no_vision = bool(getattr(args, "no_vision", False))
    prefer_vision = explicit_vision or (has_frames and not no_vision)
    if not config.ai_model and not args.model:
        chosen = _pick_model_interactive(config, prefer_vision=prefer_vision)
        if not chosen:
            return 1
    model = args.model or config.ai_model
    base_url = args.base_url or config.ai_base_url
    api_key = args.api_key if args.api_key is not None else config.ai_api_key
    max_frames = args.max_frames if args.max_frames is not None else config.ai_max_frames

    # Probe the configured model before doing real work. Ollama's /api/tags can
    # list models that are retired server-side (HTTP 410 at call time); a stored
    # ai_model can silently go dead. When that happens, fall back to the
    # interactive picker so the user picks a live one instead of crashing.
    if not args.model and model:
        ok, err = AI.probe_model(base_url, model, api_key)
        if not ok:
            ui.status(f"Configured model '{model}' is unavailable.", kind="warn", detail=err)
            chosen = _pick_model_interactive(config, prefer_vision=prefer_vision)
            if not chosen:
                return 1
            model = chosen
            base_url = args.base_url or config.ai_base_url
            api_key = args.api_key if args.api_key is not None else config.ai_api_key

    # Resolve the prompt. Explicit flags (--prompt/--plan/--summary/--actions)
    # skip the classifier; the default path auto-detects via the model.
    explicit_modes = AI._explicit_mode_set(args)
    detected_mode = ""
    if explicit_modes:
        prompt_template = AI.resolve_prompt(args)
        mode_label = next(iter(explicit_modes))
        detected_mode = mode_label
    else:
        prompt_template, detected_mode = AI.resolve_prompt_auto(
            transcript, base_url=base_url, model=model, api_key=api_key,
        )
        if detected_mode.endswith("(fallback)"):
            ui.status(f"Classifier failed; falling back to summary + actions.", kind="warn",
                      detail=AI._last_classifier_error[0] or "")
        else:
            ui.status(f"Auto-detected: {detected_mode}", kind="info")

    # Decide whether to feed frames to the model. See _resolve_vision for the
    # full precedence (no-vision > explicit --vision > auto-enable by model type).
    use_vision, vkind, vmsg = _resolve_vision(
        explicit_vision=explicit_vision, no_vision=no_vision,
        has_frames=has_frames, model=model,
    )
    if vmsg:
        ui.status(vmsg, kind=vkind or "info")

    ui.kv("Model", model)
    ui.muted(f"base_url: {base_url}  vision: {use_vision}  mode: {detected_mode}")
    frames_dir = SC.frames_dir_for(of_base) if use_vision else None
    with ui.spinner("analyzing") as spin:
        response = AI.analyze(
            prompt_template, transcript,
            base_url=base_url, model=model, api_key=api_key,
            entries=entries, frames_dir=frames_dir,
            use_vision=use_vision, max_frames=max_frames,
            on_progress=spin,
        )

    # Write the .analysis.md (prompt + response) and print response to stdout.
    md = f"# whiz analysis — {in_path.name}\n\n"
    md += f"**Model:** {model}  **Vision:** {use_vision}  **Mode:** {detected_mode}\n\n"
    md += "## Prompt\n\n```\n" + prompt_template.replace("{transcript}", "<transcript omitted>") + "\n```\n\n"
    md += "## Response\n\n" + response + "\n"
    out_path = _analysis_output_path(of_base)
    out_path.write_text(md, encoding="utf-8")
    ui.wrote("Wrote analysis", out_path)
    print(response)
    return 0


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

    # Video inputs auto-enable screenshots + diarization here too (opt out with
    # --no-screenshots / --no-speakers), matching `whiz transcribe`.
    screenshots, speakers_auto = _video_auto_flags(args, in_path)
    speakers_requested = args.speakers is not None or speakers_auto

    # Resolve the audio (WAV) to diarize. Reuse an existing sibling WAV if the
    # transcribe run kept it; otherwise re-extract from the video.
    if aud.is_audio(in_path):
        wav = in_path
    elif aud.needs_extraction(in_path):
        wav = in_path.with_suffix(".wav")
        if not wav.exists():
            ui.phase("extracting audio")
            ui.kv("Video", in_path.name)
            wav = aud.extract_audio(in_path, aud.find_ffmpeg(config.ffmpeg))
            ui.kv("Audio", str(wav))
    else:
        wav = in_path

    # Locate the whisper JSON produced by a prior transcribe run.
    json_path = Path(args.json).expanduser() if args.json else _find_whisper_json(wav.with_suffix(""), wav, of_passed=False)
    if not json_path.exists():
        raise SystemExit(
            f"No whisper JSON found (looked for {json_path}).\n"
            "Run `whiz transcribe <file>` first to produce one, or pass --json <path>."
        )
    ui.header("whiz", f"merge · v{__version__}")
    ui.kv("JSON", json_path)

    try:
        whisper_segs = MR.parse_whisper_json(json_path)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Failed to parse {json_path}: {e}")
    if not whisper_segs:
        raise SystemExit(f"No segments parsed from {json_path}.")
    ui.kv("Segs", f"{len(whisper_segs)} whisper segments")

    of_base = json_path.with_suffix("")  # e.g. ...16.03.40.wav -> ...16.03.40
    # For the wav.json case, of_base should be the input stem without .json.
    if json_path.name.endswith(".wav.json"):
        of_base = json_path.with_name(json_path.name[: -len(".json")])  # ...16.03.40.wav
        of_base = of_base.with_suffix("")  # ...16.03.40

    # Diarization params.
    num_sp = args.speakers if args.speakers else 0
    thr = args.cluster_threshold if args.cluster_threshold is not None else config.cluster_threshold
    ui.muted(f"Diarize: num_speakers={num_sp or 'auto'} cluster_threshold={thr}")

    try:
        ui.phase("diarizing")
        diar_segments = D.run_diarization(wav, config, num_speakers=num_sp, threshold=thr)
    except RuntimeError as e:
        msg = str(e)
        if "sherpa_onnx" in msg or "models not found" in msg or "download-diarization" in msg:
            if speakers_auto and args.speakers is None:
                # Auto-enabled only: fall back to screenshots-only, don't crash.
                ui.status(f"Speakers: diarization unavailable — {msg.splitlines()[0]}",
                          kind="hint",
                          detail="Skipping speaker labels. Enable with: pipx inject whiz sherpa-onnx && whiz models download-diarization")
                diar_segments = []
            else:
                raise SystemExit(
                    f"{msg}\nEnable diarization with: pipx inject whiz sherpa-onnx && "
                    f"whiz models download-diarization"
                )
        else:
            raise
    if not diar_segments:
        if speakers_requested:
            ui.status("Diarization produced no segments; writing screenshots only.", kind="warn")
        else:
            raise SystemExit("Diarization produced no segments; cannot merge.")

    merged = MR.assign_speakers(whisper_segs, diar_segments) if diar_segments else []

    # Speaker tally to stderr for quick tuning feedback (before relabeling).
    if merged:
        from collections import Counter
        counts = Counter(label for _, label in merged)
        ui.tally(counts.most_common())

    # Voice profiles: compute per-cluster embeddings and auto-match against any
    # stored profiles. Matched names seed the speaker labels; --speakers-names
    # / --name-speakers can override. Embeddings are reused for profile saving.
    profile_names: dict[str, str] = {}
    cluster_embeddings: dict[int, list[float]] = {}
    if merged and not args.no_voice_profiles:
        try:
            cluster_embeddings = P.compute_speaker_embeddings(wav, diar_segments, config)
            if cluster_embeddings:
                profile_names, _matches = P.auto_assign_names(
                    cluster_embeddings, threshold=config.speaker_match_threshold,
                )
                profile_names = {k: v for k, v in profile_names.items() if v}
        except Exception as e:  # noqa: BLE001
            ui.status(f"Warning: voice-profile matching skipped: {e}", kind="warn")

    written: list[str] = []
    if merged:
        ui.phase("merging speakers")
        srt_out, txt_out, name_map = _write_labeled_outputs(
            merged, of_base,
            name_speakers=_name_speakers_enabled(args, diarize_enabled=True),
            speakers_names=args.speakers_names,
            html=_outputs_include(args, config, "html") and not (screenshots and aud.needs_extraction(in_path)),
            title=in_path.name,
            profile_names=profile_names or None,
            cluster_embeddings=cluster_embeddings or None,
            save_profiles=config.save_voice_profiles and not args.no_voice_profiles,
        )
        ui.wrote("Wrote labeled SRT", srt_out)
        ui.wrote("Wrote dialogue TXT", txt_out)
        written.append(str(srt_out))
        written.append(str(txt_out))
        # Apply the resolved names to the caller's merged list so the
        # screenshots manifest and the HTML pass carry real names too.
        if name_map:
            merged = MR.relabel(merged, name_map)

    # Video screenshots: re-extract frames against the existing merged list.
    # Frame extraction is cheap (~seconds), so merge --screenshots re-runs it.
    frames_dir = None
    if screenshots and aud.needs_extraction(in_path):
        ui.phase("capturing frames")
        width = args.screenshot_width if args.screenshot_width is not None else 1280
        shot_list = merged if merged else [(seg, "Speaker") for seg in whisper_segs]
        result = _extract_and_manifest_screenshots(
            in_path, shot_list, of_base,
            ffmpeg=aud.find_ffmpeg(config.ffmpeg),
            width=width,
            dry_run=False,
        )
        if result is not None:
            frames_dir = result[0]
            ui.wrote("Wrote frames manifest", result[1])
            written.append(str(result[1]))
    # Write HTML after frames exist so they can be inlined.
    if _outputs_include(args, config, "html") and frames_dir is not None and merged:
        ui.phase("writing HTML transcript")
        _write_labeled_outputs(
            merged, of_base,
            name_speakers=False,
            speakers_names=None,
            html=True,
            frames_dir=frames_dir,
            title=in_path.name,
        )
        html_path = Path(str(of_base) + ".speakers.html")
        ui.wrote("Wrote HTML transcript", html_path)
        written.append(str(html_path))
    ui.summary(written)
    return 0


# ---------- dictate ----------

def cmd_dictate(args: argparse.Namespace) -> int:
    """System-wide voice dictation via mlx-whisper with a toggle hotkey.

    Listens for a global hotkey (default Ctrl+Space). Press to start/stop a
    dictation session: mic audio is transcribed and typed into whatever app
    has keyboard focus. A floating indicator shows live mic level. Requires
    macOS Accessibility + Microphone permissions and the ``dictate`` extra
    (``pipx inject whiz 'whiz[dictate]'``).
    """
    config = cfg.load()

    # --list-providers: print available providers and exit (no deps needed).
    if getattr(args, "list_providers", False):
        from whiz.dictate.providers import list_providers

        provs = list_providers()
        ui.header("whiz", "dictate providers")
        for kind in ("stt", "injector", "indicator"):
            rows = [
                [name, supports, "yes" if current else "no"]
                for name, supports, current in provs[kind]
            ]
            ui.table(
                f"{kind} providers",
                [("Name", "left"), ("Platform", "left"), ("Current", "right")],
                rows,
            )
        return 0

    # Check for the optional extra before importing the engine (which pulls in
    # sounddevice/pynput). A missing dep gives a clear install hint instead of
    # an ImportError traceback.
    try:
        import sounddevice  # noqa: F401
        import pynput  # noqa: F401
    except ImportError:
        raise SystemExit(
            "The 'dictate' extra is not installed. Install it with:\n"
            "  pipx inject whiz 'whiz[dictate]'\n\n"
            "Then grant Accessibility + Microphone permissions in System "
            "Settings → Privacy & Security."
        )

    from whiz.dictate import run_dictate

    overrides: dict[str, object] = {
        "model": args.model or "",
        "language": args.language or "",
        "prompt": args.prompt if args.prompt is not None else "",
        "hotkey": args.hotkey or "",
    }
    if args.trigger:
        overrides["trigger"] = args.trigger
    if args.idle_timeout is not None:
        overrides["idle_timeout"] = args.idle_timeout
    if args.auto_stop_silence is not None:
        overrides["auto_stop_silence"] = args.auto_stop_silence
    if args.no_indicator:
        overrides["show_indicator"] = False

    return run_dictate(config, **overrides)


def cmd_dictate_service(args: argparse.Namespace) -> int:
    """Manage the whiz dictate login LaunchAgent.

    Subcommands: install | uninstall | status.
    install writes the plist to ~/Library/LaunchAgents and loads it so
    dictation starts at login and stays running (KeepAlive). uninstall
    unloads and removes it. status reports whether it's loaded.
    """
    from whiz.dictate import service

    action = getattr(args, "service_action", "") or ""
    if action == "install":
        # Refuse to install the LaunchAgent if the 'dictate' extra isn't
        # installed: the agent would exit non-zero on startup and, with
        # KeepAlive=true, launchd would restart it in a crash loop that
        # spams the log file forever. Make the user install the extra first,
        # then run `service install`.
        try:
            import sounddevice  # noqa: F401
            import pynput  # noqa: F401
        except ImportError:
            print(
                "The 'dictate' extra is not installed. The LaunchAgent would "
                "crash-loop on startup. Install the extra first, then retry:\n"
                "  pipx inject whiz 'whiz[dictate]'\n"
                "  whiz dictate service install",
                file=sys.stderr,
            )
            return 1
        return service.install()
    if action == "uninstall":
        return service.uninstall()
    if action == "status":
        return service.status()
    raise SystemExit(f"Unknown service action '{action}'. Use install|uninstall|status.")


def cmd_dictate_setup(args: argparse.Namespace) -> int:
    """Guided first-time setup / doctor for whiz dictate.

    Checks the dictate extra, macOS Accessibility + Microphone permissions,
    prints a ✓/✗ report with next-step hints, and points at the login
    service install once prerequisites pass. No engine import — the checks
    run without starting dictation, so it's safe to run before permissions
    are granted.
    """
    from whiz.dictate import setup as setup_mod

    return setup_mod.setup()


# Friendly key names → config field names for `whiz dictate set`.
# Lets users say `whiz dictate set hotkey=<f8>` instead of the verbose
# `whiz config set dictate_hotkey=<f8>`.
_DICTATE_FRIENDLY_KEYS: dict[str, str] = {
    "model": "dictate_model",
    "language": "dictate_language",
    "lang": "dictate_language",
    "prompt": "dictate_prompt",
    "idle_timeout": "dictate_idle_timeout",
    "idle": "dictate_idle_timeout",
    "timeout": "dictate_idle_timeout",
    "hotkey": "dictate_hotkey",
    "key": "dictate_hotkey",
    "trigger": "dictate_trigger",
    "mode": "dictate_trigger",
    "vad": "dictate_vad",
    "auto_stop_silence": "dictate_auto_stop_silence",
    "silence": "dictate_auto_stop_silence",
    "show_indicator": "dictate_show_indicator",
    "indicator": "dictate_show_indicator",
    "idle_visible": "dictate_idle_visible",
    "idle_badge": "dictate_idle_visible",
    "stt_provider": "dictate_stt_provider",
    "injector": "dictate_injector",
    "indicator_provider": "dictate_indicator",
}

# All dictate_* config fields, in display order, with a short label for the
# `whiz dictate config` (show) table.
_DICTATE_CONFIG_FIELDS: list[tuple[str, str, str]] = [
    # (config_key, label, description)
    ("dictate_hotkey", "Hotkey", "Global hotkey (pynput syntax, e.g. <ctrl>+<space>)"),
    ("dictate_trigger", "Trigger", "toggle (press to start/stop) or ptt (hold to talk)"),
    ("dictate_language", "Language", "Spoken language code (default: ru)"),
    ("dictate_model", "Model", "mlx-whisper model repo/path (empty = default whisper-large-v3-turbo)"),
    ("dictate_prompt", "Prompt", "Whisper initial_prompt (empty = built-in Russian jargon)"),
    ("dictate_idle_timeout", "Idle timeout", "Seconds before model unloads after session (0 = never)"),
    ("dictate_auto_stop_silence", "Auto-stop silence", "Seconds of silence to auto-stop (0 = off)"),
    ("dictate_vad", "VAD", "WebRTC VAD for utterance segmentation"),
    ("dictate_show_indicator", "Indicator", "Floating dictation overlay"),
    ("dictate_idle_visible", "Idle badge", "Keep the indicator dimmed-visible while idle (not just during a session)"),
    ("dictate_stt_provider", "STT provider", "Force STT provider (empty = auto)"),
    ("dictate_injector", "Injector", "Force text injector (empty = auto)"),
    ("dictate_indicator", "Indicator provider", "Force indicator provider (empty = auto)"),
]


def cmd_dictate_config(args: argparse.Namespace) -> int:
    """Show current dictation settings in a readable table."""
    config = cfg.load()
    ui.header("whiz", "dictate settings")
    rows: list[list[str]] = []
    for key, label, desc in _DICTATE_CONFIG_FIELDS:
        value = getattr(config, key)
        rows.append([label, _format_dictate_value(value), desc])
    ui.table(
        f"Config: {cfg.CONFIG_PATH}",
        [("Setting", "left"), ("Value", "left"), ("Description", "left")],
        rows,
    )
    ui.muted("Change with:  whiz dictate set <key>=<value>  (e.g. whiz dictate set hotkey=<f8>)")
    return 0


def _format_dictate_value(value: object) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, str) and value == "":
        return "(default)"
    return str(value)


def cmd_dictate_set(args: argparse.Namespace) -> int:
    """Set a dictation setting using friendly key names.

    ``whiz dictate set hotkey=<f8>`` is equivalent to
    ``whiz config set dictate_hotkey=<f8>`` but easier to type and discover.
    Accepts friendly aliases (lang, idle, silence, indicator, ...) mapped via
    _DICTATE_FRIENDLY_KEYS. Unknown keys are rejected with the valid list.
    """
    config = cfg.load()
    assignment = args.assignment
    if "=" not in assignment:
        raise SystemExit("Expected KEY=VALUE (e.g. whiz dictate set hotkey=<f8>)")
    friendly_key, _, value = assignment.partition("=")
    friendly_key = friendly_key.strip().lower()
    # Resolve friendly name → config field.
    if friendly_key not in _DICTATE_FRIENDLY_KEYS:
        valid = ", ".join(sorted(_DICTATE_FRIENDLY_KEYS.keys()))
        raise SystemExit(
            f"Unknown dictate setting '{friendly_key}'. Valid: {valid}"
        )
    config_key = _DICTATE_FRIENDLY_KEYS[friendly_key]
    field_type = cfg.Config.__dataclass_fields__[config_key].type
    coerced = _coerce(value.strip(), field_type)
    # Validate enum-like fields (e.g. dictate_trigger must be toggle/ptt).
    _validate_config_value(config_key, coerced)
    setattr(config, config_key, coerced)
    path = cfg.save(config)
    ui.status(f"Set {friendly_key} = {coerced!r}  →  {config_key}", kind="ok")
    ui.muted(f"Saved to {path}")
    return 0


# ---------- speakers (voice profiles) ----------

def cmd_speakers_list(args: argparse.Namespace) -> int:
    """List stored speaker voice profiles."""
    profiles = P.load_profiles()
    if not profiles:
        ui.info(f"No voice profiles found in {P.profiles_dir()}")
        ui.muted("Profiles are saved automatically when you name speakers with")
        ui.muted("--name-speakers or --speakers-names (unless --no-voice-profiles).")
        return 0
    rows = []
    for prof in profiles:
        path = P._profile_path(prof.name)
        rows.append([prof.name, str(prof.dim), str(prof.samples), prof.created, str(path)])
    ui.table(
        f"Voice profiles ({len(profiles)})",
        [("Name", "left"), ("Dim", "right"), ("Samples", "right"), ("Created", "left"), ("Path", "left")],
        rows,
    )
    ui.muted(f"Match threshold: {cfg.load().speaker_match_threshold} (whiz config set speaker_match_threshold=...)")
    return 0


def cmd_speakers_forget(args: argparse.Namespace) -> int:
    """Delete a stored speaker voice profile by name."""
    name = args.name
    removed = P.forget_profile(name)
    if removed:
        ui.status(f"Forgot voice profile: {name}", kind="ok")
        return 0
    ui.status(f"No voice profile named {name!r} in {P.profiles_dir()}", kind="warn")
    return 1


def cmd_speakers_match(args: argparse.Namespace) -> int:
    """Show how a recording's clusters match against stored profiles (dry run).

    Runs diarization on the given file and prints the cosine-similarity scores
    of each cluster against every stored profile, plus the auto-assignment
    decision at the configured threshold. Does not relabel or save anything.
    """
    config = cfg.load()
    in_path = Path(args.file).expanduser()
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")
    if aud.is_audio(in_path):
        wav = in_path
    elif aud.needs_extraction(in_path):
        wav = in_path.with_suffix(".wav")
        if not wav.exists():
            ui.phase("extracting audio")
            ui.kv("Video", in_path.name)
            wav = aud.extract_audio(in_path, aud.find_ffmpeg(config.ffmpeg))
            ui.kv("Audio", str(wav))
    else:
        wav = in_path

    num_sp = args.speakers if args.speakers else 0
    thr = args.cluster_threshold if args.cluster_threshold is not None else config.cluster_threshold
    ui.phase("diarizing")
    diar_segments = D.run_diarization(wav, config, num_speakers=num_sp, threshold=thr)
    if not diar_segments:
        raise SystemExit("Diarization produced no segments.")

    profiles = P.load_profiles()
    if not profiles:
        ui.info(f"No stored voice profiles in {P.profiles_dir()}; nothing to match against.")
        return 0

    cluster_embeddings = P.compute_speaker_embeddings(wav, diar_segments, config)
    from whiz.merge import speaker_label
    matches = P.match_speakers(cluster_embeddings, profiles, threshold=config.speaker_match_threshold)
    rows = []
    for cid, emb in sorted(cluster_embeddings.items()):
        scores = sorted(
            ((P.cosine_similarity(emb, prof.embedding), prof.name) for prof in profiles),
            reverse=True,
        )
        all_str = ", ".join(f"{nm}={s:.3f}" for s, nm in scores)
        m = matches.get(cid)
        best = f"{m[0]}" if m else "(no match)"
        best_score = f"{m[1]:.3f}" if m else f"{scores[0][0]:.3f}"
        rows.append([speaker_label(cid), best, best_score, all_str])
    ui.table(
        "Speaker match (dry run)",
        [("Cluster", "left"), ("Best name", "left"), ("Best score", "right"), ("All scores", "left")],
        rows,
    )
    ui.muted(f"Threshold: {config.speaker_match_threshold}")
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
    # With `from __future__ import annotations`, dataclass field types are
    # strings (e.g. "bool") not the type objects. Normalize to a string name.
    ft = field_type if isinstance(field_type, str) else getattr(field_type, "__name__", str(field_type))
    if ft == "bool":
        return value.lower() in {"1", "true", "yes", "on"}
    if ft == "int":
        return int(value)
    if ft == "float":
        return float(value)
    if ft == "list":
        return [v.strip() for v in value.split(",") if v.strip()]
    return value


# Enum-like config fields with a fixed set of allowed values. Shared by both
# `whiz config set` and `whiz dictate set` so the two entry points enforce the
# same constraints — a typo via either path can't silently degrade.
_CONFIG_ENUM_VALUES: dict[str, set[str]] = {
    "dictate_trigger": {"toggle", "ptt"},
}


def _validate_config_value(key: str, value: object) -> None:
    """Reject out-of-range values for enum-like config fields."""
    allowed = _CONFIG_ENUM_VALUES.get(key)
    if allowed is not None and value not in allowed:
        raise SystemExit(
            f"Invalid {key}={value!r}. Must be one of: {', '.join(sorted(allowed))}"
        )


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
    _validate_config_value(key, coerced)
    setattr(config, key, coerced)
    path = cfg.save(config)
    print(f"Set {key} = {coerced!r}")
    print(f"Saved to {path}")
    return 0


# ---------- argparse ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="whiz",
        description="whiz — transcription CLI. Transcribe, diarize, name speakers, "
                    "capture frames, build HTML transcripts, and run AI analysis. "
                    "Powered by whisper.cpp.",
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
    t.add_argument("--speakers", type=int, default=None, nargs="?", const=0, help="Enable speaker diarization via sherpa-onnx. Optional integer = known speaker count; omit = auto-detect. Auto-enabled for video inputs (see --no-speakers)")
    t.add_argument("--no-speakers", dest="no_speakers", action="store_true", help="Disable the auto-enabled speaker diarization for video inputs (opt out)")
    t.add_argument("--cluster-threshold", type=float, default=None, help="Diarization clustering threshold when auto-detecting (larger = fewer speakers; default 0.9)")
    t.add_argument("--name-speakers", action="store_true", help="Interactively prompt to name each detected speaker. Auto-enabled when diarization runs (see --no-name-speakers)")
    t.add_argument("--no-name-speakers", dest="no_name_speakers", action="store_true", help="Disable the auto-enabled interactive speaker-naming prompt (opt out)")
    t.add_argument("--speakers-names", dest="speakers_names", nargs="+", default=None, help="Non-interactive speaker names assigned by total talk time (most talkative first), e.g. --speakers-names Alice,Bob,Carol,Dave")
    t.add_argument("--screenshots", action="store_true", help="For video inputs, extract one on-screen frame per transcribed segment into <stem>.frames/ and write <stem>.frames.json (for AI analysis / HTML output). Auto-enabled for video inputs (see --no-screenshots)")
    t.add_argument("--no-screenshots", dest="no_screenshots", action="store_true", help="Disable the auto-enabled on-screen frame extraction for video inputs (opt out)")
    t.add_argument("--screenshot-width", type=int, default=None, help="Frame width in pixels (default 1280; 0 = native resolution)")
    t.add_argument("--no-voice-profiles", dest="no_voice_profiles", action="store_true", help="Don't compute voice-profile embeddings or auto-match/save speaker profiles this run")
    t.add_argument("--resume", action="store_true", help="Skip whisper-cli transcription if its JSON output already exists and go straight to diarization + merge (ergonomic alias for `whiz merge`)"),
    t.add_argument("--verbose", action="store_true", help="Verbose whisper-cli output")
    t.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="Extra flags passed verbatim to whisper-cli")
    t.add_argument("--dry-run", action="store_true", help="Print the command without running it")
    t.add_argument("--analyze", action="store_true", help="After transcription, run AI analysis (auto-detect: summary+actions or implementation plan). Equivalent to a follow-up `whiz analyze <file>`. For video inputs this auto-enables vision when the AI model is vision-capable.")
    t.add_argument("--vision", action="store_true", help="With --analyze, force sending on-screen frames to a vision model (auto-enabled for video when the model is vision-capable; this flag forces it on for audio/non-video runs)")
    t.add_argument("--no-vision", dest="no_vision", action="store_true", help="With --analyze, opt out of the auto-enabled vision analysis (stay text-only even for a video with frames)")
    t.set_defaults(func=cmd_transcribe)

    # merge
    mg = sub.add_parser("merge", help="Re-run diarization + merge against an existing whisper JSON (skip transcription)")
    mg.add_argument("file", help="Input audio/video file (used to find the whisper JSON and re-extract WAV if needed)")
    mg.add_argument("--json", default="", help="Explicit path to the whisper JSON (default: auto-find next to input)")
    mg.add_argument("--outputs", default=None, help="Comma-separated whiz post-merge output formats: html (others are whisper-cli formats, ignored here)")
    mg.add_argument("--speakers", type=int, default=None, nargs="?", const=0, help="Known speaker count; omit = auto-detect. Auto-enabled for video inputs (see --no-speakers)")
    mg.add_argument("--no-speakers", dest="no_speakers", action="store_true", help="Disable the auto-enabled speaker diarization for video inputs (opt out)")
    mg.add_argument("--cluster-threshold", type=float, default=None, help="Clustering threshold when auto-detecting (larger = fewer speakers; default 0.9)")
    mg.add_argument("--name-speakers", action="store_true", help="Interactively prompt to name each detected speaker. Auto-enabled when diarization runs (see --no-name-speakers)")
    mg.add_argument("--no-name-speakers", dest="no_name_speakers", action="store_true", help="Disable the auto-enabled interactive speaker-naming prompt (opt out)")
    mg.add_argument("--speakers-names", dest="speakers_names", nargs="+", default=None, help="Non-interactive speaker names assigned by total talk time (most talkative first), e.g. --speakers-names Alice,Bob,Carol,Dave")
    mg.add_argument("--screenshots", action="store_true", help="Re-extract on-screen frames per segment into <stem>.frames/ and write <stem>.frames.json. Auto-enabled for video inputs (see --no-screenshots)")
    mg.add_argument("--no-screenshots", dest="no_screenshots", action="store_true", help="Disable the auto-enabled on-screen frame extraction for video inputs (opt out)")
    mg.add_argument("--screenshot-width", type=int, default=None, help="Frame width in pixels (default 1280; 0 = native resolution)")
    mg.add_argument("--no-voice-profiles", dest="no_voice_profiles", action="store_true", help="Don't compute voice-profile embeddings or auto-match/save speaker profiles this run")
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

    # analyze
    an = sub.add_parser("analyze", aliases=["a"], help="AI-analyze a prior transcript (+ frames): auto-detects meeting vs implementation-plan, or use --summary/--actions/--plan/--prompt. Every analysis also appends a dense ## Essentials section (concentrated points for later context)")
    an.add_argument("file", help="Input file (used to find the .frames.json manifest or .speakers.txt alongside it)")
    an.add_argument("--model", default="", help="AI model name (default: config ai_model, e.g. llava, qwen2.5-vl, gpt-4o-mini)")
    an.add_argument("--base-url", dest="base_url", default="", help="Chat API base URL (default: config ai_base_url, http://localhost:11434/v1)")
    an.add_argument("--api-key", dest="api_key", default=None, help="API key (default: config ai_api_key; Ollama ignores it)")
    an.add_argument("--max-frames", dest="max_frames", type=int, default=None, help="Max frames sent to a vision model, spread evenly (default: config ai_max_frames, 50)")
    an.add_argument("--summary", action="store_true", help="Use the built-in summary prompt")
    an.add_argument("--actions", action="store_true", help="Use the built-in action-items prompt")
    an.add_argument("--plan", action="store_true", help="Use the built-in implementation-plan prompt (Overview → Goal → Proposed approach → Steps with owner/effort → Risks → Open questions → Acceptance criteria)")
    an.add_argument("--prompt", default="", help="Freeform prompt (overrides --summary/--actions/--plan; use {transcript} placeholder for the transcript)")
    an.add_argument("--vision", action="store_true", help="Send on-screen frames as images to a vision model (requires a prior --screenshots run). Auto-enabled when a frames manifest exists and the model is vision-capable; --no-vision opts out")
    an.add_argument("--no-vision", dest="no_vision", action="store_true", help="Opt out of the auto-enabled vision analysis (stay text-only even when frames exist)")
    an.set_defaults(func=cmd_analyze)

    # speakers (voice profiles)
    sp = sub.add_parser("speakers", aliases=["sp"], help="Manage speaker voice profiles (cross-recording recognition)")
    spsub = sp.add_subparsers(dest="speakers_command", required=True)
    spsub.add_parser("list", aliases=["ls"]).set_defaults(func=cmd_speakers_list)
    sf = spsub.add_parser("forget", aliases=["rm"], help="Delete a stored speaker voice profile by name")
    sf.add_argument("name", help="Speaker name to forget")
    sf.set_defaults(func=cmd_speakers_forget)
    sm = spsub.add_parser("match", help="Show how a recording's clusters match stored profiles (dry run)")
    sm.add_argument("file", help="Input audio/video file")
    sm.add_argument("--speakers", type=int, default=None, nargs="?", const=0, help="Known speaker count; omit = auto-detect")
    sm.add_argument("--cluster-threshold", type=float, default=None, help="Clustering threshold when auto-detecting (default 0.9)")
    sm.set_defaults(func=cmd_speakers_match)

    # dictate
    dt = sub.add_parser("dictate", aliases=["d"], help="System-wide voice dictation via mlx-whisper. Toggle or push-to-talk with a global hotkey; transcribed text is typed into the focused app. Requires the 'dictate' extra: pipx inject whiz 'whiz[dictate]'")
    dt.add_argument("--model", default="", help="mlx-whisper model repo/path (default: mlx-community/whisper-large-v3-turbo)")
    dt.add_argument("-l", "--language", default="", help="Spoken language code (default: ru)")
    dt.add_argument("--prompt", default=None, help="Whisper initial_prompt to bias recognition (default: built-in Russian jargon/obscenity prompt)")
    dt.add_argument("--idle-timeout", dest="idle_timeout", type=float, default=None, help="Seconds to keep the model loaded after a session before unloading (default: 45; 0 = never unload)")
    dt.add_argument("--hotkey", default="", help="Global hotkey in pynput syntax (default: <ctrl>+<space>)")
    dt.add_argument("--trigger", default="", choices=["", "toggle", "ptt"], help="Trigger mode: toggle (press to start/stop) or ptt (hold to talk; release to stop). Default: config dictate_trigger")
    dt.add_argument("--auto-stop-silence", dest="auto_stop_silence", type=float, default=None, help="Seconds of silence before a session auto-stops (default: 10; 0 = off)")
    dt.add_argument("--no-indicator", dest="no_indicator", action="store_true", help="Hide the floating dictation indicator overlay")
    dt.add_argument("--list-providers", dest="list_providers", action="store_true", help="List available STT/injector/indicator providers for this platform and exit")
    # Optional subcommands: `whiz dictate config` and `whiz dictate set`.
    # When no subcommand is given, bare `whiz dictate` runs dictation (via the
    # default func=cmd_dictate set below). Subcommands override the default.
    dtsub = dt.add_subparsers(dest="dictate_command", required=False)
    dtsub.add_parser("config", aliases=["cfg"], help="Show current dictation settings").set_defaults(func=cmd_dictate_config)
    dts = dtsub.add_parser("set", aliases=["s"], help="Set a dictation setting with a friendly key name (e.g. whiz dictate set hotkey=<f8>)")
    dts.add_argument("assignment", help="KEY=VALUE, e.g. hotkey=<f8> or trigger=ptt or language=en")
    dts.set_defaults(func=cmd_dictate_set)
    dsvc = dtsub.add_parser("service", aliases=["svc"], help="Manage the whiz dictate login LaunchAgent (install | uninstall | status)")
    dsvc_sub = dsvc.add_subparsers(dest="dictate_service_action", required=True)
    dsvc_sub.add_parser("install", help="Install and load the LaunchAgent so dictation starts at login").set_defaults(func=cmd_dictate_service, service_action="install")
    dsvc_sub.add_parser("uninstall", aliases=["remove"], help="Unload and remove the LaunchAgent").set_defaults(func=cmd_dictate_service, service_action="uninstall")
    dsvc_sub.add_parser("status", aliases=["st"], help="Show whether the service is loaded").set_defaults(func=cmd_dictate_service, service_action="status")
    dtsub.add_parser("setup", aliases=["doctor"], help="Guided first-time setup: check the dictate extra, Accessibility + Microphone permissions, and point at the login service").set_defaults(func=cmd_dictate_setup)
    dt.set_defaults(func=cmd_dictate)

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