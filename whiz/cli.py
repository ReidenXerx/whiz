"""whiz CLI — a handy wrapper around whisper-cli.

Subcommands:
  whiz transcribe <file>   Transcribe an audio/video file.
  whiz models list         Show discovered models.
  whiz models download N   Download a model from HuggingFace.
  whiz speakers list       List stored voice profiles.
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
        print("Speakers: diarization not available (sherpa-onnx or models missing); "
              "skipping speaker labels for this run.", file=sys.stderr)
        print("  Enable with:  pipx inject whiz sherpa-onnx && whiz models download-diarization",
              file=sys.stderr)
        print("  Or silence this with: --no-speakers", file=sys.stderr)
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
    # We need a parseable whisper JSON to merge diarization against AND to
    # drive the per-segment screenshots path (even without diarization). Force
    # JSON (in addition to any user-requested formats) so we can parse segments.
    if (diarize_enabled or screenshots) and "json" not in outputs and "json-full" not in outputs:
        outputs = list(outputs) + ["json"]
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
    print(f"Extracted {ok}/{len(entries)} frames -> {frames_dir}", file=sys.stderr)
    print(f"Wrote frames manifest: {manifest_path}", file=sys.stderr)
    return frames_dir, manifest_path


def _save_named_profiles(
    name_map: dict[str, str],
    cluster_embeddings: dict[int, list[float]],
) -> None:
    """Save a voice profile for each speaker that received a real name.

    ``name_map`` is keyed by ``Speaker A/B/...`` labels; we map those back to
    cluster ids via the merge module's letter ordering and persist the
    corresponding embedding under the chosen name.
    """
    from whiz.merge import _SPEAKER_LETTERS

    label_to_cid: dict[str, int] = {
        f"Speaker {letter}": i for i, letter in enumerate(_SPEAKER_LETTERS)
    }
    saved = 0
    for label, name in name_map.items():
        cid = label_to_cid.get(label)
        if cid is None or cid not in cluster_embeddings:
            continue
        # Don't save a profile whose "name" is just the default Speaker label.
        if not name or name.startswith("Speaker "):
            continue
        try:
            path = P.save_profile(name, cluster_embeddings[cid])
            saved += 1
            print(f"Saved voice profile: {name} -> {path}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"Warning: could not save voice profile for {name}: {e}", file=sys.stderr)
    if saved:
        print(f"Saved {saved} voice profile(s) to {P.profiles_dir()}", file=sys.stderr)


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
        print(f"Auto-matched {len(profile_names)} speaker(s) from voice profiles.", file=sys.stderr)
        for lbl, nm in profile_names.items():
            print(f"  {lbl} -> {nm}", file=sys.stderr)
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
        print(f"Wrote HTML transcript: {html_out}", file=sys.stderr)
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
            print(f"Speakers: diarization unavailable — {msg.splitlines()[0]}", file=sys.stderr)
            print("  Skipping speaker labels for this run. Enable with:", file=sys.stderr)
            print("    pipx inject whiz sherpa-onnx && whiz models download-diarization", file=sys.stderr)
            return []
        raise
    if not diar_segments:
        print("Warning: diarization produced no segments; falling back to unlabeled output.", file=sys.stderr)
    return diar_segments


def cmd_transcribe(args: argparse.Namespace) -> int:
    config = cfg.load()
    cmd, model_path, wav, in_path, keep_wav, of_base, diarize_enabled, screenshots = _build_transcribe_args(args, config)

    print(f"Model:  {model_path}", file=sys.stderr)
    print(f"Input:  {in_path}", file=sys.stderr)
    if wav != in_path:
        print(f"Audio:  {wav}", file=sys.stderr)
    if aud.needs_extraction(in_path):
        flags = []
        if screenshots:
            flags.append("screenshots=on" if not args.screenshots else "screenshots=on (explicit)")
        if diarize_enabled:
            flags.append("speakers=on" if args.speakers is None else f"speakers={args.speakers or 'auto'} (explicit)")
        if flags:
            print(f"Video input — auto-enabled: {', '.join(flags)}", file=sys.stderr)
    print(f"Run:    {' '.join(cmd)}", file=sys.stderr)
    print("-" * 60, file=sys.stderr)

    if args.dry_run:
        if diarize_enabled:
            num_sp = args.speakers if args.speakers else 0
            thr = args.cluster_threshold if args.cluster_threshold is not None else config.cluster_threshold
            D.run_diarization(wav, config, num_speakers=num_sp, threshold=thr, dry_run=True)
        print("\nDRY-RUN: not executing whisper-cli.", file=sys.stderr)
        return 0

    # --- Resumability: skip transcription if a whisper JSON already exists ---
    # --resume lets you re-run `whiz transcribe` to redo diarization + merge
    # (e.g. with a different --speakers count) without re-running whisper-cli.
    # It's an ergonomic alias for `whiz merge` triggered from transcribe.
    json_path = _find_whisper_json(of_base, wav, of_passed=bool(args.output))
    resuming = bool(getattr(args, "resume", False) and json_path.exists())
    diar_segments: list[D.DiarSegment] = []
    if resuming:
        print(f"--resume: found existing whisper JSON {json_path}; skipping transcription.", file=sys.stderr)
        rc = 0
        # Diarization still runs so a new --speakers count / threshold takes
        # effect against the existing transcription.
        if diarize_enabled:
            diar_segments = _run_diarize_or_fallback(wav, config, args)
    else:
        # --- Diarization path ---
        if diarize_enabled:
            diar_segments = _run_diarize_or_fallback(wav, config, args)

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
                        print(f"Warning: voice-profile matching skipped: {e}", file=sys.stderr)
                # Frames must be extracted before writing HTML so they can be
                # inlined; for the diarized path we extract after the labeled
                # outputs but before HTML if both are requested.
                srt_out, txt_out, name_map = _write_labeled_outputs(
                    merged, of_base,
                    name_speakers=args.name_speakers,
                    speakers_names=args.speakers_names,
                    html=want_html and not want_frames,
                    title=in_path.name,
                    profile_names=profile_names or None,
                    cluster_embeddings=cluster_embeddings or None,
                    save_profiles=config.save_voice_profiles and not args.no_voice_profiles,
                )
                print(f"Wrote labeled SRT:  {srt_out}", file=sys.stderr)
                print(f"Wrote dialogue TXT: {txt_out}", file=sys.stderr)
                # Video screenshots: one frame per segment, using the relabeled
                # merged list so the manifest carries final speaker names.
                frames_dir = None
                if want_frames:
                    width = args.screenshot_width if args.screenshot_width is not None else 1280
                    result = _extract_and_manifest_screenshots(
                        in_path, merged, of_base,
                        ffmpeg=aud.find_ffmpeg(config.ffmpeg),
                        width=width,
                        dry_run=args.dry_run,
                    )
                    if result is not None:
                        frames_dir = result[0]
                # Write HTML after frames exist so they can be inlined.
                if want_html and want_frames and frames_dir is not None:
                    _write_labeled_outputs(
                        merged, of_base,
                        name_speakers=False,
                        speakers_names=None,
                        html=True,
                        frames_dir=frames_dir,
                        title=in_path.name,
                    )

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
                print(f"Warning: failed to parse {json_path}: {e}", file=sys.stderr)
                whisper_segs = []
            if whisper_segs:
                unlabeled = [(seg, "Speaker") for seg in whisper_segs]
                width = args.screenshot_width if args.screenshot_width is not None else 1280
                _extract_and_manifest_screenshots(
                    in_path, unlabeled, of_base,
                    ffmpeg=aud.find_ffmpeg(config.ffmpeg),
                    width=width,
                    dry_run=args.dry_run,
                )

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


# ---------- analyze ----------

def _analysis_output_path(of_base: Path) -> Path:
    return Path(str(of_base) + ".analysis.md")


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze a prior transcript (and optionally frames) with an AI model.

    Loads the frames manifest if present (<stem>.frames.json) for both the
    transcript text and (with --vision) the frame images; otherwise loads the
    <stem>.speakers.txt transcript. Writes the prompt + response to
    <stem>.analysis.md and prints the response to stdout.
    """
    config = cfg.load()
    if not config.ai_model and not args.model:
        raise SystemExit(
            "No AI model configured. Set one with:\n"
            "  whiz config set ai_model=llava\n"
            "or pass --model on the command line."
        )
    model = args.model or config.ai_model
    base_url = args.base_url or config.ai_base_url
    api_key = args.api_key if args.api_key is not None else config.ai_api_key
    max_frames = args.max_frames if args.max_frames is not None else config.ai_max_frames

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
        print(f"Loaded frames manifest: {manifest_path} ({len(entries)} segments)", file=sys.stderr)
    elif txt_path.exists():
        transcript = txt_path.read_text(encoding="utf-8")
        print(f"Loaded transcript: {txt_path}", file=sys.stderr)
    else:
        raise SystemExit(
            f"No transcript found. Looked for:\n  {manifest_path}\n  {txt_path}\n"
            "Run `whiz transcribe --speakers [--screenshots] <file>` first."
        )

    prompt_template = AI.resolve_prompt(args)
    use_vision = args.vision and entries is not None
    if args.vision and entries is None:
        print("--vision requested but no frames manifest found; falling back to text-only.", file=sys.stderr)
        use_vision = False

    print(f"Model: {model}  base_url: {base_url}  vision: {use_vision}", file=sys.stderr)
    if use_vision:
        frames_dir = SC.frames_dir_for(of_base)
        frame_paths = [frames_dir / e.frame for e in (entries or []) if e.frame]
        print(f"Sending {len(frame_paths)} frames (cap {max_frames}) ...", file=sys.stderr)
        response = AI.chat_vision(
            prompt_template, transcript, frame_paths,
            base_url=base_url, model=model, api_key=api_key, max_frames=max_frames,
        )
    else:
        response = AI.chat_text(
            prompt_template, transcript,
            base_url=base_url, model=model, api_key=api_key,
        )

    # Write the .analysis.md (prompt + response) and print response to stdout.
    out_path = _analysis_output_path(of_base)
    md = f"# whiz analysis — {in_path.name}\n\n"
    md += f"**Model:** {model}  **Vision:** {use_vision}\n\n"
    md += "## Prompt\n\n```\n" + prompt_template.replace("{transcript}", "<transcript omitted>") + "\n```\n\n"
    md += "## Response\n\n" + response + "\n"
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote analysis: {out_path}", file=sys.stderr)
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
            print(f"Extracting audio from {in_path} ...", file=sys.stderr)
            wav = aud.extract_audio(in_path, aud.find_ffmpeg(config.ffmpeg))
    else:
        wav = in_path

    # Locate the whisper JSON produced by a prior transcribe run.
    json_path = Path(args.json).expanduser() if args.json else _find_whisper_json(wav.with_suffix(""), wav, of_passed=False)
    if not json_path.exists():
        raise SystemExit(
            f"No whisper JSON found (looked for {json_path}).\n"
            "Run `whiz transcribe <file>` first to produce one, or pass --json <path>."
        )
    print(f"Whisper JSON:  {json_path}", file=sys.stderr)

    try:
        whisper_segs = MR.parse_whisper_json(json_path)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Failed to parse {json_path}: {e}")
    if not whisper_segs:
        raise SystemExit(f"No segments parsed from {json_path}.")
    print(f"Whisper segments: {len(whisper_segs)}", file=sys.stderr)

    of_base = json_path.with_suffix("")  # e.g. ...16.03.40.wav -> ...16.03.40
    # For the wav.json case, of_base should be the input stem without .json.
    if json_path.name.endswith(".wav.json"):
        of_base = json_path.with_name(json_path.name[: -len(".json")])  # ...16.03.40.wav
        of_base = of_base.with_suffix("")  # ...16.03.40

    # Diarization params.
    num_sp = args.speakers if args.speakers else 0
    thr = args.cluster_threshold if args.cluster_threshold is not None else config.cluster_threshold
    print(f"Diarize: num_speakers={num_sp or 'auto'} cluster_threshold={thr}", file=sys.stderr)

    try:
        diar_segments = D.run_diarization(wav, config, num_speakers=num_sp, threshold=thr)
    except RuntimeError as e:
        msg = str(e)
        if "sherpa_onnx" in msg or "models not found" in msg or "download-diarization" in msg:
            if speakers_auto and args.speakers is None:
                # Auto-enabled only: fall back to screenshots-only, don't crash.
                print(f"Speakers: diarization unavailable — {msg.splitlines()[0]}", file=sys.stderr)
                print("  Skipping speaker labels. Enable with:", file=sys.stderr)
                print("    pipx inject whiz sherpa-onnx && whiz models download-diarization", file=sys.stderr)
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
            print("Diarization produced no segments; writing screenshots only.", file=sys.stderr)
        else:
            raise SystemExit("Diarization produced no segments; cannot merge.")

    merged = MR.assign_speakers(whisper_segs, diar_segments) if diar_segments else []

    # Speaker tally to stderr for quick tuning feedback (before relabeling).
    if merged:
        from collections import Counter
        tally = Counter(label for _, label in merged)
        print(f"Detected speakers: {len(tally)}", file=sys.stderr)
        for label, n in tally.most_common():
            print(f"  {label}: {n} segments", file=sys.stderr)

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
            print(f"Warning: voice-profile matching skipped: {e}", file=sys.stderr)

    if merged:
        srt_out, txt_out, name_map = _write_labeled_outputs(
            merged, of_base,
            name_speakers=args.name_speakers,
            speakers_names=args.speakers_names,
            html=_outputs_include(args, config, "html") and not (screenshots and aud.needs_extraction(in_path)),
            title=in_path.name,
            profile_names=profile_names or None,
            cluster_embeddings=cluster_embeddings or None,
            save_profiles=config.save_voice_profiles and not args.no_voice_profiles,
        )
        print(f"Wrote labeled SRT:  {srt_out}", file=sys.stderr)
        print(f"Wrote dialogue TXT: {txt_out}", file=sys.stderr)

    # Video screenshots: re-extract frames against the existing merged list.
    # Frame extraction is cheap (~seconds), so merge --screenshots re-runs it.
    frames_dir = None
    if screenshots and aud.needs_extraction(in_path):
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
    # Write HTML after frames exist so they can be inlined.
    if _outputs_include(args, config, "html") and frames_dir is not None and merged:
        _write_labeled_outputs(
            merged, of_base,
            name_speakers=False,
            speakers_names=None,
            html=True,
            frames_dir=frames_dir,
            title=in_path.name,
        )
    return 0


# ---------- speakers (voice profiles) ----------

def cmd_speakers_list(args: argparse.Namespace) -> int:
    """List stored speaker voice profiles."""
    profiles = P.load_profiles()
    if not profiles:
        print(f"No voice profiles found in {P.profiles_dir()}")
        print("Profiles are saved automatically when you name speakers with")
        print("--name-speakers or --speakers-names (unless --no-voice-profiles).")
        return 0
    print(f"{'NAME':<24} {'DIM':>5}  {'CREATED':<22}  PATH")
    for prof in profiles:
        path = P._profile_path(prof.name)
        print(f"{prof.name:<24} {prof.dim:>5}  {prof.created:<22}  {path}")
    print(f"\n{len(profiles)} profile(s) in {P.profiles_dir()}")
    print(f"Match threshold: {cfg.load().speaker_match_threshold} (whiz config set speaker_match_threshold=...)")
    return 0


def cmd_speakers_forget(args: argparse.Namespace) -> int:
    """Delete a stored speaker voice profile by name."""
    name = args.name
    removed = P.forget_profile(name)
    if removed:
        print(f"Forgot voice profile: {name}")
        return 0
    print(f"No voice profile named {name!r} in {P.profiles_dir()}")
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
            print(f"Extracting audio from {in_path} ...", file=sys.stderr)
            wav = aud.extract_audio(in_path, aud.find_ffmpeg(config.ffmpeg))
    else:
        wav = in_path

    num_sp = args.speakers if args.speakers else 0
    thr = args.cluster_threshold if args.cluster_threshold is not None else config.cluster_threshold
    diar_segments = D.run_diarization(wav, config, num_speakers=num_sp, threshold=thr)
    if not diar_segments:
        raise SystemExit("Diarization produced no segments.")

    profiles = P.load_profiles()
    if not profiles:
        print(f"No stored voice profiles in {P.profiles_dir()}; nothing to match against.")
        return 0

    cluster_embeddings = P.compute_speaker_embeddings(wav, diar_segments, config)
    from whiz.merge import speaker_label
    sep = ", "
    print(f"\n{'CLUSTER':<12} {'BEST NAME':<20} {'BEST SCORE':>10}  ALL SCORES")
    matches = P.match_speakers(cluster_embeddings, profiles, threshold=config.speaker_match_threshold)
    for cid, emb in sorted(cluster_embeddings.items()):
        scores = sorted(
            ((P.cosine_similarity(emb, prof.embedding), prof.name) for prof in profiles),
            reverse=True,
        )
        all_str = sep.join(f"{nm}={s:.3f}" for s, nm in scores)
        m = matches.get(cid)
        best = f"{m[0]}" if m else "(no match)"
        best_score = f"{m[1]:.3f}" if m else f"{scores[0][0]:.3f}"
        print(f"{speaker_label(cid):<12} {best:<20} {best_score:>10}  {all_str}")
    print(f"\nThreshold: {config.speaker_match_threshold}")
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
    t.add_argument("--speakers", type=int, default=None, nargs="?", const=0, help="Enable speaker diarization via sherpa-onnx. Optional integer = known speaker count; omit = auto-detect. Auto-enabled for video inputs (see --no-speakers)")
    t.add_argument("--no-speakers", dest="no_speakers", action="store_true", help="Disable the auto-enabled speaker diarization for video inputs (opt out)")
    t.add_argument("--cluster-threshold", type=float, default=None, help="Diarization clustering threshold when auto-detecting (larger = fewer speakers; default 0.9)")
    t.add_argument("--name-speakers", action="store_true", help="After transcription, interactively prompt to name each detected speaker (replaces Speaker A/B/C with real names)")
    t.add_argument("--speakers-names", dest="speakers_names", nargs="+", default=None, help="Non-interactive speaker names assigned by total talk time (most talkative first), e.g. --speakers-names Alice,Bob,Carol,Dave")
    t.add_argument("--screenshots", action="store_true", help="For video inputs, extract one on-screen frame per transcribed segment into <stem>.frames/ and write <stem>.frames.json (for AI analysis / HTML output). Auto-enabled for video inputs (see --no-screenshots)")
    t.add_argument("--no-screenshots", dest="no_screenshots", action="store_true", help="Disable the auto-enabled on-screen frame extraction for video inputs (opt out)")
    t.add_argument("--screenshot-width", type=int, default=None, help="Frame width in pixels (default 1280; 0 = native resolution)")
    t.add_argument("--no-voice-profiles", dest="no_voice_profiles", action="store_true", help="Don't compute voice-profile embeddings or auto-match/save speaker profiles this run")
    t.add_argument("--resume", action="store_true", help="Skip whisper-cli transcription if its JSON output already exists and go straight to diarization + merge (ergonomic alias for `whiz merge`)"),
    t.add_argument("--verbose", action="store_true", help="Verbose whisper-cli output")
    t.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="Extra flags passed verbatim to whisper-cli")
    t.add_argument("--dry-run", action="store_true", help="Print the command without running it")
    t.set_defaults(func=cmd_transcribe)

    # merge
    mg = sub.add_parser("merge", help="Re-run diarization + merge against an existing whisper JSON (skip transcription)")
    mg.add_argument("file", help="Input audio/video file (used to find the whisper JSON and re-extract WAV if needed)")
    mg.add_argument("--json", default="", help="Explicit path to the whisper JSON (default: auto-find next to input)")
    mg.add_argument("--outputs", default=None, help="Comma-separated whiz post-merge output formats: html (others are whisper-cli formats, ignored here)")
    mg.add_argument("--speakers", type=int, default=None, nargs="?", const=0, help="Known speaker count; omit = auto-detect. Auto-enabled for video inputs (see --no-speakers)")
    mg.add_argument("--no-speakers", dest="no_speakers", action="store_true", help="Disable the auto-enabled speaker diarization for video inputs (opt out)")
    mg.add_argument("--cluster-threshold", type=float, default=None, help="Clustering threshold when auto-detecting (larger = fewer speakers; default 0.9)")
    mg.add_argument("--name-speakers", action="store_true", help="Interactively prompt to name each detected speaker (replaces Speaker A/B/C with real names)")
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
    an = sub.add_parser("analyze", aliases=["a"], help="Analyze a prior transcript (and optional frames) with an AI model via Ollama/OpenAI-compatible API")
    an.add_argument("file", help="Input file (used to find the .frames.json manifest or .speakers.txt alongside it)")
    an.add_argument("--model", default="", help="AI model name (default: config ai_model, e.g. llava, qwen2.5-vl, gpt-4o-mini)")
    an.add_argument("--base-url", dest="base_url", default="", help="Chat API base URL (default: config ai_base_url, http://localhost:11434/v1)")
    an.add_argument("--api-key", dest="api_key", default=None, help="API key (default: config ai_api_key; Ollama ignores it)")
    an.add_argument("--max-frames", dest="max_frames", type=int, default=None, help="Max frames sent to a vision model, spread evenly (default: config ai_max_frames, 50)")
    an.add_argument("--summary", action="store_true", help="Use the built-in summary prompt")
    an.add_argument("--actions", action="store_true", help="Use the built-in action-items prompt")
    an.add_argument("--prompt", default="", help="Freeform prompt (overrides --summary/--actions; use {transcript} placeholder for the transcript)")
    an.add_argument("--vision", action="store_true", help="Send on-screen frames as images to a vision model (requires a prior --screenshots run)")
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