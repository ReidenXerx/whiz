"""whiz CLI — transcription subcommands.

Subcommands:
  whiz transcribe <file>   Transcribe an audio/video file.
    --analyze              Chain into AI analysis after transcription.
    --ai-model NAME        AI model for the chained analysis (default: config).
  whiz merge <file>        Re-run diarization + merge against an existing JSON.
  whiz models list         Show discovered models.
  whiz models download N   Download a model from HuggingFace.
  whiz speakers list       List stored voice profiles.
  whiz analyze <file>      AI-analyze a prior transcript (+ frames).
  whiz dictate             Moved to Mynah: github.com/ReidenXerx/mynah
  whiz config show         Show current config.
  whiz config edit         Open config in $EDITOR.
  whiz config set K=V      Set a config value.

Run `whiz transcribe -h` for transcription flags.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
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
from whiz import library as L
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

# The diarize extra's requirement string. The auto-setup installs the SAME
# spec pyproject declares — a bare `pip install sherpa-onnx` could land a
# version older than the project's declared minimum, making the auto path and
# the documented manual path (`pipx inject whiz 'whiz[diarize]'`) install
# different things. Keep in sync with [project.optional-dependencies] diarize
# in pyproject.toml.
_DIARIZE_REQUIREMENT = "sherpa-onnx>=1.10"


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


def _outputs_explicitly_include(args: argparse.Namespace, fmt: str) -> bool:
    """True if ``fmt`` was passed via ``--outputs`` on THIS invocation.

    Distinct from ``_outputs_include``, which also counts config-supplied
    outputs. The unlabeled (degraded) fallbacks are gated on this: silently
    dropping an ``--outputs html`` the user just typed is the bug, while a
    config default describes the success path — when diarization produces
    nothing, a config-html run keeps master's quieter skip instead of
    surprising the user with degraded artifacts they never asked for by
    flag.
    """
    raw = getattr(args, "outputs", None)
    if not raw:
        return False
    return fmt in [o.strip() for o in raw.split(",") if o.strip()]


def _will_write_generic_labels(args: argparse.Namespace) -> bool:
    """True when the unlabeled fallback will actually write generic-label
    artifacts for this invocation: an explicit ``--outputs html``, or a
    video run whose screenshots are (auto-)enabled — the frames manifest
    and degraded transcript carry generic 'Speaker' labels. Audio runs
    without an explicit html write nothing speaker-related at all, so
    their fallback message must say "skipping", not "falling back to
    generic labels" (the message has to describe what happens next).
    """
    if _outputs_explicitly_include(args, "html"):
        return True
    file_arg = getattr(args, "file", None)
    if not file_arg:
        return False
    return aud.needs_extraction(Path(file_arg)) and not getattr(args, "no_screenshots", False)


_WHISPER_PROGRESS_RE = re.compile(
    r"whisper_print_progress_callback: progress\s*=\s*(\d+)\s*%"
)


def _run_whisper_streaming(cmd: list[str]) -> subprocess.Popen:
    """Run whisper-cli, streaming its stdout/stderr line-by-line to our stderr.

    Each line is prefixed with elapsed time since the process started so long
    transcriptions show progress pacing. Output is styled via the ui module
    (dimmed timestamp prefix + muted content) and degrades to plain text when
    piped. Returns the Popen object after completion.

    whisper-cli's own progress line
    (``whisper_print_progress_callback: progress =  42%``) is swallowed and
    re-emitted as ``ui.progress`` — over a pipe that becomes the
    ``~ progress: N%`` machine line the macOS UI's status chip reads; on a TTY
    it stays a dim ``progress: N%`` line. Swallowing it also keeps the raw
    callback noise out of logs on BOTH branches — the raw line is whisper's
    internal trace, not run output the user needs.
    """
    import time

    start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        # Explicit, not the locale's: a GUI-launched child can have no LANG,
        # and one undecodable byte must not end the run.
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    try:
        with ui.streaming_progress(cmd) as write:
            for line in proc.stdout:
                if (m := _WHISPER_PROGRESS_RE.search(line)) and ":" in line:
                    ui.progress(int(m.group(1)))
                    continue
                elapsed = time.monotonic() - start
                write(line, elapsed)
    except BaseException:
        # ^C or a rendering error: don't leave whisper-cli running unreaped
        # with nobody reading its pipe.
        proc.kill()
        proc.wait()
        raise
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


def _bundled() -> bool:
    """True when this whiz is the engine inside Whiz.app.

    The app's launcher (macos/build-engine.sh) sets ``WHIZ_BUNDLED=1``.
    That Python lives in a signed bundle: nothing may be installed into it,
    and updating it means updating the app.
    """
    return os.environ.get("WHIZ_BUNDLED") == "1"


def _install_sherpa_onnx() -> bool:
    """Install the ``diarize`` extra (sherpa-onnx) into the running venv.

    Runs ``{sys.executable} -m pip install {_DIARIZE_REQUIREMENT}`` streaming
    output live, then refreshes Python's import caches so the very next
    ``import sherpa_onnx`` in THIS process sees the fresh wheel without a
    restart (pip puts it in site-packages; importlib.invalidate_caches + a
    find_spec probe is enough — sherpa_onnx is a normal top-level module,
    not a lazy stub). Returns True on success.

    ``sys.executable`` (not a pipx binary) installs into whatever venv is
    running whiz — a dev ``uv run`` venv, a pipx venv, anything — so this
    also covers upgrade-reinstalled environments. Under pipx, ``pipx inject
    whiz 'whiz[diarize]'`` is the equivalent manual command (see README).
    """
    import importlib

    if _bundled():
        # Whiz.app ships sherpa-onnx; missing means a damaged bundle, and a
        # pip install would break its signature rather than fix it.
        ui.status("Warning: speaker detection is missing from this copy of Whiz.app.", kind="warn",
                  detail="Reinstall Whiz.app (drag a fresh copy to Applications).")
        return False

    ui.status("Speakers: sherpa-onnx missing — installing the diarize extra now", kind="info")
    ui.muted("One-time setup (a ~90 MB wheel + model download on first run). Opt out with: --no-auto-diarization-setup")
    rc = subprocess.run(
        [sys.executable, "-m", "pip", "install", _DIARIZE_REQUIREMENT],
        check=False,
    ).returncode
    if rc != 0:
        ui.status(f"Warning: pip install sherpa-onnx failed (exit {rc}).", kind="warn",
                  detail="Run manually: pipx inject whiz 'whiz[diarize]'")
        return False
    importlib.invalidate_caches()
    try:
        import importlib.util

        if importlib.util.find_spec("sherpa_onnx") is None:
            # L (wave-1 audit): pip exited 0 but the module is still not
            # importable (partial install, wrong venv, shadowed module).
            # Silently returning False made the NEXT diarization attempt
            # look like a fresh install loop with nothing explaining why.
            ui.status(
                "Warning: pip reported success but sherpa_onnx is still not importable.",
                kind="warn",
                detail="Install manually: pipx inject whiz 'whiz[diarize]' "
                       "&& whiz models download-diarization",
            )
            return False
    except Exception as e:  # noqa: BLE001
        ui.status(f"Warning: could not verify the sherpa-onnx install: {e}", kind="warn",
                  detail="Install manually: pipx inject whiz 'whiz[diarize]'")
        return False
    ui.status("sherpa-onnx installed.", kind="ok")
    return True


def _auto_setup_consent(config: cfg.Config) -> bool:
    """Resolve user consent for the one-time diarization auto-setup.

    Order of authority (review decision, 2026-09-07):

    1. ``auto_diarization_setup`` in config (bool) answers permanently —
       written by this prompt's y/N, or set by hand with
       ``whiz config set auto_diarization_setup=false``.
    2. Interactive terminal: ask ONCE (y/N, default No — writing into
       site-packages deserves a prompt, unlike the VAD model's cache-file
       download), then persist the answer so the question never recurs.
       KeyboardInterrupt at the prompt is a plain decline that persists
       nothing; EOF (piped stdin under a tty stderr) declines and persists.
    3. Non-interactive (piped stdin or stderr: scripts, cron, launchd):
       allow — prompt-out would break scripted fresh machines; the
       ``--no-auto-diarization-setup`` flag remains the per-run opt-out.
    """
    answered = getattr(config, "auto_diarization_setup", None)
    if isinstance(answered, bool):
        # L (wave-1 audit): only a REAL bool answers permanently. A
        # hand-edited `auto_diarization_setup = "false"` string is truthy
        # in Python and counted as consent — non-bool values fall through
        # to the prompt / non-interactive rules instead.
        return answered
    if not (sys.stdin.isatty() and sys.stderr.isatty()):
        return True
    ui.status(
        "Speakers: diarization needs a one-time setup — install "
        f"'{_DIARIZE_REQUIREMENT}' into this Python environment and "
        "download the diarization models (~90 MB).",
        kind="info",
    )
    # input() writes its prompt to stdout without a trailing newline — the
    # answer lands on the same line. Matches _prompt_speaker_names' idiom.
    try:
        answer = input("Proceed? [y/N] ").strip().lower()
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return False
    except EOFError:
        answer = ""
    allowed = answer in {"y", "yes"}
    if not allowed:
        ui.status(
            "Skipping the one-time diarization setup (declined). Install "
            "manually with: pipx inject whiz 'whiz[diarize]' && whiz models "
            "download-diarization — or allow it later with: whiz config set "
            "auto_diarization_setup=true",
            kind="hint",
        )
    # Remember the answer either way — the question is once-ever. A failed
    # save must not turn a user's choice into a crash.
    try:
        config.auto_diarization_setup = allowed
        path = cfg.save(config)
        ui.muted(f"Remembered this choice (auto_diarization_setup={str(allowed).lower()}) in {path}")
    except OSError as e:
        ui.status(f"Warning: could not persist the choice to config: {e}", kind="warn")
    return allowed


def _ensure_diarization_ready(config: cfg.Config, *, dry_run: bool = False, setup_allowed: bool = True) -> bool:
    """Make diarization possible before a run: install package + download models.

    Proactive-first policy (user decision, 2026-09-05): when diarization is
    about to run — auto-enabled for video or explicitly requested — and the
    one-time setup is missing, whiz performs it on the spot instead of
    degrading: ``_install_sherpa_onnx`` (pip install of the diarize extra's
    declared spec into the running venv), then ``D.download_diarization_models``
    (~90 MB one-time download). The degraded fallbacks stay as the safety net
    for when setup fails (offline, disk full, ...), is opted out via
    ``--no-auto-diarization-setup``, or is declined at the consent prompt.

    Review decision (2026-09-07): unlike the VAD model download (a cache
    file), this writes into site-packages, so an interactive terminal is
    ASKED first (y/N, once — the answer persists in the
    ``auto_diarization_setup`` config key). Non-interactive sessions
    (piped stdin/stderr: scripts, cron, launchd) proceed automatically so a
    fresh machine still just works; ``auto_diarization_setup`` answers
    permanently either way.

    Returns True when diarization is ready (either it already was, or setup
    succeeded). DRY-RUN reports what would be installed/downloaded and never
    performs the setup. A False return must leave callers on their existing
    degraded/skip path — setup failure is never a crash here.
    """
    if _diarization_available(config):
        return True
    if dry_run:
        ui.muted("DRY-RUN: diarization setup would run — pip install "
                 f"'{_DIARIZE_REQUIREMENT}' + download diarization models "
                 "(~90 MB one-time).")
        return False
    if not setup_allowed:
        return False
    if not _auto_setup_consent(config):
        return False
    # Package first: the model finders are filesystem-only, but installing
    # models without the package that runs them would leave half a setup.
    try:
        import sherpa_onnx  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        if not _install_sherpa_onnx():
            return False
    if D.find_segmentation_model(config) is None or D.find_embedding_model(config) is None:
        try:
            D.download_diarization_models()
        except Exception as e:  # noqa: BLE001
            ui.status(f"Warning: diarization model download failed: {e}", kind="warn",
                      detail="Run manually: whiz models download-diarization")
            return False
        ui.status("Diarization models downloaded.", kind="ok")
    # Re-check rather than assume: a partial download (e.g. one model file)
    # must still land on the caller's degraded path, not a broken run.
    return _diarization_available(config)


def _print_zero_segments_hints() -> None:
    """Actionable tips after diarization ran but produced no segments.

    Review follow-up: the old message stated the outcome but not what to DO
    about it. Segmentation finding no speech is an audio-content property —
    no install or setting repairs it — so the honest fix is guidance: lock
    the speaker count, loosen clustering, or accept that silence produces no
    segments (no auto-retry; it would re-run the multi-minute pipeline to
    the same verdict).
    """
    for hint in (
        "If you know the speaker count, pass it: --speakers N (locks clustering; the biggest accuracy lever)",
        "Auto-detect found nothing: try a lower --cluster-threshold (e.g. 0.85; smaller = more speakers)",
        "Diarization segments speech only — silence or non-speech audio produces no segments",
    ):
        ui.muted(f"  {hint}")


# Set by the macOS UI for the child it spawns: the caller is a front-end that
# reads the `? ...` machine lines and answers them over stdin. Without it, a
# non-TTY stdin is never read for answers to prompts the user didn't ask for.
UI_PIPE_ENV = "WHIZ_UI_PIPE"


def _name_speakers_enabled(args: argparse.Namespace, diarize_enabled: bool) -> bool:
    """Resolve whether the interactive speaker-naming prompt should run.

    Naming only makes sense when diarization actually ran. It's auto-enabled
    in that case (so you get prompted to label speakers without passing
    ``--name-speakers``); ``--no-name-speakers`` opts out. An explicit
    ``--name-speakers`` keeps working regardless — including over a piped
    stdin, where a script deliberately feeds the names.

    The AUTO-enabled prompt needs someone to answer it: a terminal, or the
    macOS UI, which sets ``WHIZ_UI_PIPE=1`` and answers ``? speaker-name:``
    lines over stdin. Any other non-TTY stdin is somebody else's data —
    ``ls *.mov | while read f; do whiz transcribe "$f"; done`` used to feed
    the remaining filenames in as speaker names and save them as
    user-confirmed voice profiles.
    """
    if not diarize_enabled:
        return False
    if getattr(args, "no_name_speakers", False):
        return False
    if getattr(args, "name_speakers", False):
        return True
    if sys.stdin.isatty() or os.environ.get(UI_PIPE_ENV) == "1":
        return True
    ui.muted("Speaker naming skipped: stdin is not a terminal "
             "(pass --name-speakers to read names from stdin).")
    return False


def _discarded_naming_flags(args: argparse.Namespace) -> list[str]:
    """Explicit speaker-naming flags that cannot take effect without labels.

    Used by the unlabeled fallbacks: ``--speakers-names`` / ``--name-speakers``
    are silently dropped when diarization produces nothing, so the fallback
    warning must say so instead of letting the user believe names were applied.
    """
    flags: list[str] = []
    if getattr(args, "speakers_names", None):
        flags.append("--speakers-names")
    if getattr(args, "name_speakers", False):
        flags.append("--name-speakers")
    return flags


def _discarded_naming_detail(args: argparse.Namespace) -> str | None:
    """Warning detail naming the speaker-naming flags a fallback discarded."""
    flags = _discarded_naming_flags(args)
    if not flags:
        return None
    return (f"{', '.join(flags)} had no effect (no speaker labels were "
            "produced); the names were not applied.")


def _build_transcribe_args(args: argparse.Namespace, config: cfg.Config) -> list[str]:
    """Assemble the whisper-cli argv (name kept for history).

    NOTE: this helper has outgrown its docstring's original scope — beyond
    argv assembly it performs run preparation with side effects:
    video auto-flag resolution, the proactive diarization one-time setup
    (``_ensure_diarization_ready`` — with consent this can pip-install the
    diarize extra and download models), VAD model resolution with its own
    auto-download, and output/flag assembly. The name stays for history
    (and the many tests that stub it).

    TEST HAZARD (review): calling the REAL helper with diarization enabled
    and no stub can run a REAL ``pip install`` on the test machine. Stub
    ``cli._ensure_diarization_ready`` (see ``_setup_transcribe``) or pass
    ``no_auto_diarization_setup=True`` in every test that reaches it.
    """
    # Resolve input file first so video auto-enable can inform diarize_enabled.
    in_path = Path(args.file).expanduser()
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")
    screenshots, speakers_auto = _video_auto_flags(args, in_path)
    # diarize_enabled is True when the user passed --speakers (with or without
    # a count) OR when it's auto-enabled for a video input.
    diarize_enabled = args.speakers is not None or speakers_auto
    # Proactive-first (user decision, 2026-09-05): diarization about to run —
    # even merely auto-enabled for video — gets its one-time setup performed
    # on the spot (pip install sherpa-onnx + ~90 MB model download) instead
    # of being skipped. Only when the setup fails (or --no-auto-diarization-setup)
    # does the quiet skip-with-hint below remain, mirroring the VAD model's
    # auto-download; VAD stays on and screenshots still run either way.
    if (speakers_auto or args.speakers is not None) and not _ensure_diarization_ready(
        config,
        dry_run=getattr(args, "dry_run", False),
        setup_allowed=not getattr(args, "no_auto_diarization_setup", False),
    ) and speakers_auto and args.speakers is None:
        ui.status("Speakers: diarization not available (setup incomplete); skipping speaker labels for this run.",
                  kind="hint",
                  detail="Run manually: pipx inject whiz 'whiz[diarize]' && whiz models download-diarization")
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

    # Decode to WAV for a video, or audio whisper-cli cannot read itself.
    keep_wav = args.keep_wav
    if aud.needs_decoding(in_path):
        if args.dry_run:
            wav = aud.extract_audio(in_path, aud.find_ffmpeg(config.ffmpeg), dry_run=True)
            print(f"DRY-RUN: would extract audio ->> {wav}")
        else:
            ui.phase("extracting audio")
            ui.kv("Video" if aud.needs_extraction(in_path) else "Recording", in_path.name)
            dest = aud.extraction_dir(in_path)
            if dest is not None:
                ui.status(
                    f"{aud.extracted_wav_path(in_path).name} next to the recording is not whiz's "
                    "own extraction; leaving it untouched and extracting to a temp dir.",
                    kind="warn",
                )
            wav = aud.extract_audio(in_path, aud.find_ffmpeg(config.ffmpeg), dest_dir=dest)
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
    # We need a parseable whisper JSON to merge diarization against, to drive
    # the per-segment screenshots path (even without diarization), AND to build
    # the HTML transcript (even with no speaker labels at all). Force JSON
    # (in addition to any user-requested formats) so we can parse segments.
    if (diarize_enabled or screenshots or "html" in outputs) and "json" not in outputs and "json-full" not in outputs:
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
    naming_wav = _naming_wav(in_path, wav)
    of_base = Path(args.output).expanduser() if args.output else naming_wav.with_suffix("")
    if args.output:
        of_flag = ["-of", str(of_base)]
    elif naming_wav != wav:
        # The audio was extracted to a temp dir (a foreign <stem>.wav holds
        # the usual spot): name whisper's outputs as if it were in the usual
        # spot, so <stem>.wav.json lands beside the video where merge,
        # --resume and analyze look for it.
        of_flag = ["-of", str(naming_wav)]

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
    # Progress: -pp (print progress) is the DEFAULT on both branches: on a
    # TTY the user sees the live line; when piped, _run_whisper_streaming
    # swallows it and re-emits the `~ progress: N%` machine line the macOS
    # UI's status chip reads — a pipe is not "logs", it is the UI's channel.
    # --no-progress drops it; --print-progress forces it over any opt-out.
    #
    # -np is NOT the opposite of -pp: it is whisper-cli's --no-prints (model
    # load, Metal init and other diagnostics), and the two combine. Dropping
    # -np whenever -pp was on turned a 5-line run into ~120 lines of init
    # noise that crowded the UI's capped log. Only verbose lets it through.
    if args.print_progress or not args.no_progress:
        cmd.append("-pp")
    if not config.verbose and not args.verbose:
        cmd.append("-np")
    if config.extra_args:
        cmd += config.extra_args
    if args.extra:
        cmd += args.extra

    return cmd, model_path, wav, in_path, keep_wav, of_base, diarize_enabled, screenshots


def _naming_wav(in_path: Path, wav: Path) -> Path:
    """The WAV path output names hang off, wherever the audio really is.

    For a video (or audio whisper-cli cannot read) that is always
    ``<stem>.wav`` beside it, even when the audio was extracted to a temp dir
    to spare a foreign file at that path.
    """
    return aud.extracted_wav_path(in_path) if aud.needs_decoding(in_path) else wav


def _find_whisper_json(of_base: Path, wav: Path, of_passed: bool) -> Path | None:
    """Locate the whisper-cli JSON output.

    whisper-cli names outputs after the *input file stem*. When ``-of`` is NOT
    passed the input is the ``.wav`` file, so the JSON is ``<wav>.json`` (the
    ``.wav`` suffix is part of the stem, e.g. ``foo.wav.json``). When ``-of`` IS
    passed the JSON is ``<of_base>.json``.

    Candidates are built by APPENDING (not ``Path.with_suffix``) so a dotted
    output stem like ``-o /x/out.v2`` resolves to ``out.v2.json`` —
    ``with_suffix(".json")`` would replace ``.v2`` and silently look for
    ``out.json``, dropping the requested HTML merge or ingesting a stale
    transcript from an older run.
    """
    candidates: list[Path] = []
    if of_passed:
        # -of was passed: whisper-cli appends ".json" to the base exactly
        # once and writes nothing else, so <of_base>.json is the only real
        # candidate. The with_suffix fallbacks that used to sit here would
        # happily ingest a stale out.json from an older run whenever the
        # real out.v2.json was absent — exactly the failure the docstring
        # warns about — so they are gone. A missing candidate is returned
        # as-is so the caller warns instead of silently merging stale data.
        candidates.append(Path(str(of_base) + ".json"))
    else:
        # whisper-cli appends the format extension to the full input path.
        candidates.append(Path(str(wav) + ".json"))
        candidates.append(Path(str(of_base) + ".json"))
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
) -> dict[str, str | None]:
    """Interactively ask the user to name each detected speaker.

    Shows one representative quote per speaker (the longest utterance) and
    prompts for a real name. Returns a ``{"Speaker A": "Alice", ...}`` map
    whose values may be None: None is the explicit decline from an answer
    of ``-``, and the caller must DROP any name the label already carries —
    the default_names here are suggestions, and an auto-match seeded before
    the prompt would otherwise survive a rejection of it. Blank input keeps
    the default label; when ``default_names`` is supplied (from
    ``--speakers-names`` or a voice-profile auto-match), the suggested name
    is shown in the prompt and used as the value if the user presses Enter.

    The prompt works unchanged over a pipe (the macOS UI's world): each
    question is emitted as a ``? speaker-name:`` machine line by
    ``ui.speaker_prompt`` and the answer is read from stdin with ``input()`` —
    a UI holding the pipe open answers it by writing one line per question.
    The ``input()`` prompt TEXT is TTY-only: on a pipe it would echo to
    stdout (flushed, unterminated) and corrupt the merged stream the UI
    parses — see the loop body. An answer of ``-`` declines: the default
    label is kept even when a suggestion exists (a UI needs a way to reject
    a wrong voice-profile auto-match; confirming it would merge the wrong
    name into the profile as user-confirmed).
    """
    speakers = MR.speakers_in_order(merged)
    quotes = MR.representative_quotes(merged)
    name_map: dict[str, str | None] = {}
    ui.header("whiz", "name the speakers")
    ui.muted("A representative quote is shown for each. Type a real name, or")
    ui.muted("press Enter to accept the suggestion; '-' keeps the default label.")
    for label in speakers:
        quote = quotes.get(label, "(no quote)")
        suggestion = (default_names or {}).get(label)
        ui.note("")
        # One call renders both worlds: the TTY's colored label + quote, or
        # the piped machine line the UI turns into its naming bar.
        ui.speaker_prompt(label, quote, suggestion)
        prompt_text = f"Name for {label}"
        if suggestion:
            prompt_text = f"Name for {label} [{suggestion}]"
        # TTY-only prompt text. Over a pipe, input() echoes its prompt to
        # stdout — flushed, no trailing newline — and the UI merges stdout
        # and stderr into one stream, so that echo would glue itself onto
        # the NEXT ? speaker-name: machine line and the run would deadlock
        # waiting on an answer the UI never saw. The machine line already
        # told the UI everything; stdin stays the answer channel either way.
        displayed = prompt_text + ": " if sys.stdin.isatty() else ""
        try:
            name = input(displayed).strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            break
        # '-' declines: keep the default label even when a suggestion
        # exists. Without it there is no way to reject a wrong voice-profile
        # auto-match — every 'no name' answer would confirm it, upgrading
        # the wrong match to user-confirmed and merging it into the profile.
        # None (not a missing key) so the caller can tell "explicitly
        # rejected" from "never asked" and drop any seeded auto-match.
        if name == "-":
            name_map[label] = None
        elif name:
            name_map[label] = name
        elif suggestion:
            name_map[label] = suggestion
    ui.note("")
    return name_map


def _manifest_is_named(manifest_path: Path) -> bool:
    """True if an existing frames manifest carries real speaker labels.

    H5 (wave-1 audit): the degraded shot lists (every cue labeled the bare
    generic ``Speaker``) used to overwrite a NAMED manifest from an earlier
    diarized run — the manifest write was unconditional, so re-running a
    video without speakers destroyed the named per-segment labels (and the
    names any later HTML pass would inline). Mirrors ``_looks_degraded_*``:
    a manifest is "named" when any entry's speaker differs from the bare
    ``Speaker`` label (a letterized ``Speaker A`` or a real name counts);
    an UNREADABLE manifest reads as named — when in doubt, keep.
    """
    entries = SC.load_manifest(manifest_path)
    if entries is None:
        # load_manifest returns None for missing AND unreadable files: a
        # missing one is fine to write, an unreadable one must not be
        # clobbered by a degraded run.
        return manifest_path.exists()
    return any(e.speaker != "Speaker" for e in entries)


def _extract_and_manifest_screenshots(
    video: Path,
    merged: list[tuple[MR.WhisperSeg, str]],
    of_base: Path,
    ffmpeg: str,
    width: int,
    dry_run: bool = False,
) -> tuple[Path, Path, bool] | None:
    """Extract one frame per segment and write the .frames.json manifest.

    Frames go into ``<of_base>.frames/``; the manifest at ``<of_base>.frames.json``
    references frames by path only (never bytes) so it stays small and
    re-runnable. Returns ``(frames_dir, manifest_path, manifest_kept)`` or
    None if there are no segments. Only valid for video inputs (the caller
    checks).

    H5 (wave-1 audit): the manifest write is under the kept-outputs guard —
    same contract as ``_write_html_transcript``. When THIS run's list is
    degraded (every label the bare generic ``Speaker``) but an existing
    manifest is NAMED (an earlier diarized run's), the existing file is
    KEPT with a warning and ``manifest_kept`` is True — callers must not
    announce it as written. An existing degraded manifest (or none) is
    refreshed so re-runs with a different --model/--language stay
    idempotent.
    """
    if not merged:
        return None
    frames_dir = SC.frames_dir_for(of_base)
    manifest_path = SC.frames_manifest_path(of_base)
    incoming_degraded = all(label == "Speaker" for _seg, label in merged)
    if incoming_degraded and _manifest_is_named(manifest_path):
        ui.status(
            f"{manifest_path.name} already exists with named speakers — kept "
            "(this run has no speaker labels; a generic-label rewrite would "
            "destroy its speaker names).",
            kind="warn",
            detail=str(manifest_path),
        )
        return frames_dir, manifest_path, True
    entries = SC.extract_segment_frames(
        video, merged, frames_dir,
        ffmpeg=ffmpeg,
        width=width,
        dry_run=dry_run,
    )
    SC.write_manifest(entries, frames_dir, manifest_path)
    ok = sum(1 for e in entries if e.frame)
    ui.muted(f"Extracted {ok}/{len(entries)} frames -> {frames_dir}")
    return frames_dir, manifest_path, False


def _save_named_profiles(
    name_map: dict[str, str],
    cluster_embeddings: dict[int, list[float]],
    auto_labels: set[str] | None = None,
) -> None:
    """Save (or merge) a voice profile for each speaker that received a real name.

    ``name_map`` is keyed by ``Speaker A/B/...`` labels; we map those back to
    cluster ids via the merge module's letter ordering and persist the
    corresponding embedding under the chosen name. If a profile already exists
    for that name and the save is USER-confirmed, ``P.save_profile`` merges
    the new embedding with the stored one via a sample-weighted running mean,
    so re-confirming a speaker across recordings makes their profile more
    accurate over time.

    M3 (wave-1 audit): labels in ``auto_labels`` carry names sourced from a
    voice-profile auto-match, not a human confirmation. Those saves pass
    ``auto_match=True``: an auto-match may CREATE a profile (marked
    ``source: "auto"``) but never MERGE into or REPLACE an existing one — a
    chain of self-confirming matches used to silently drift the stored
    centroid. When the guard keeps an existing profile, the run SAYS so
    instead of reporting a merge that never happened.
    """
    from whiz.merge import _SPEAKER_LETTERS

    label_to_cid: dict[str, int] = {
        f"Speaker {letter}": i for i, letter in enumerate(_SPEAKER_LETTERS)
    }
    auto_labels = auto_labels or set()
    saved = 0
    merged_count = 0
    for label, name in name_map.items():
        cid = label_to_cid.get(label)
        if cid is None or cid not in cluster_embeddings:
            continue
        # Don't save a profile whose "name" is just the default Speaker label.
        if not name or name.startswith("Speaker "):
            continue
        is_auto = label in auto_labels
        try:
            existed = P._profile_path(name).exists()
            path = P.save_profile(name, cluster_embeddings[cid], samples=1, auto_match=is_auto)
            if is_auto and existed:
                # save_profile's no-clobber guard kept the existing file —
                # announce the keep, not a merge that did not happen.
                ui.status(
                    f"Voice profile '{name}' already exists — auto-match not "
                    "merged (only a name you confirm merges into it).",
                    kind="hint",
                    detail=str(path),
                )
                continue
            saved += 1
            if existed:
                merged_count += 1
                ui.status(f"Merged voice profile: {name}", kind="ok", detail=str(path))
            elif is_auto:
                ui.status(f"Saved voice profile (auto-match): {name}", kind="ok", detail=str(path))
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
) -> tuple[Path, Path, Path | None, dict[str, str]]:
    """Optionally relabel speakers, then write .speakers.srt and .speakers.txt.

    Returns the (srt_path, txt_path, html_path, name_map) used — ``html_path``
    is None unless ``html`` was requested. When ``html`` is True also writes
    a self-contained ``.speakers.html`` (frames inlined as base64 if
    ``frames_dir`` is given); the write is NOT announced here, so callers
    report artifacts in srt → txt → frames → html order, matching the
    summary panel. Naming precedence:

    1. Voice-profile auto-match (``profile_names``) seeds defaults.
    2. ``--speakers-names`` supplies a non-interactive list (assigned by total
       talk time) that overrides profile matches.
    3. ``--name-speakers`` then prompts interactively, with the combined names
       shown as defaults. An answer of '-' declines: the label's name is
       removed entirely (sources 1 and 2 included) and the default
       ``Speaker X`` label stays.

    When ``save_profiles`` is True and ``cluster_embeddings`` is provided, a
    voice profile is saved for each speaker that ended up with a real name
    (i.e. not ``Speaker X``), so later recordings can auto-match them.
    Auto-matched names are saved with ``auto_match=True`` — create-only,
    never a merge (M3, wave-1; see ``_save_named_profiles``); names a human
    supplied via ``--speakers-names`` or the interactive prompt merge
    normally.
    """
    name_map: dict[str, str] = {}
    # Labels whose name arrived via profile auto-match (M3, wave-1): tracked
    # so profile saving can flag those saves as machine-sourced. Any later
    # HUMAN source that writes the label — --speakers-names, the interactive
    # prompt, even accepting the suggested name with Enter — upgrades it to
    # user-confirmed.
    auto_labels: set[str] = set()
    # 1. Voice-profile auto-match seeds the defaults.
    if profile_names and merged:
        name_map.update(profile_names)
        auto_labels.update(profile_names)
        ui.info(f"Auto-matched {len(profile_names)} speaker(s) from voice profiles.")
        for lbl, nm in profile_names.items():
            ui.muted(f"  {lbl} -> {nm}")
    # 2. Non-interactive --speakers-names override profile matches.
    if speakers_names and merged:
        merged, list_map = _apply_speaker_names_list(merged, speakers_names)
        name_map.update(list_map)
        auto_labels.difference_update(list_map)
    # 3. Interactive prompt overrides/augments when both are given.
    if name_speakers and merged:
        interactive_map = _prompt_speaker_names(merged, default_names=name_map or None)
        for label, name in interactive_map.items():
            if name is None:
                # '-': an explicit decline — the label keeps its default
                # letter even over an auto-match or --speakers-names name,
                # and no voice profile is saved for it. A plain update()
                # would leave the rejected auto-match in place.
                name_map.pop(label, None)
            else:
                # Any human answer — including Enter confirming the
                # suggestion — upgrades provenance (M3): out of auto_labels.
                name_map[label] = name
            auto_labels.discard(label)
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
    html_out: Path | None = None
    if html:
        html_out = Path(str(of_base) + ".speakers.html")
        html_out.write_text(
            MR.format_speakers_html(merged, frames_dir=frames_dir, title=title),
            encoding="utf-8",
        )
    # Save voice profiles for speakers that received a real name.
    if save_profiles and cluster_embeddings and name_map:
        _save_named_profiles(name_map, cluster_embeddings, auto_labels=auto_labels)
    return srt_out, txt_out, html_out, name_map


# A degraded (unlabeled) .speakers.txt line: "Speaker (00:01:23): text" —
# every cue carries the bare generic label. A diarized txt always has
# letterized ("Speaker A (") or real-name ("Vadim (") labels on at least one
# line, so all-lines-match cleanly separates the two.
_DEGRADED_TXT_LINE = re.compile(r"^Speaker \(\d{2}:\d{2}:\d{2}\): ")


def _looks_degraded_html(path: Path) -> bool:
    """True if an existing .speakers.html is itself a degraded page.

    Degraded pages embed the ``_GENERIC_LABEL_NOTE`` provenance line — the
    note contains no HTML-escapable characters, so it appears verbatim in
    the file (the idempotence tests write a real degraded page and re-run,
    which guards this invariant against future note edits). A diarized
    page never carries the note. Unreadable files read as named — when in
    doubt, keep.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return _GENERIC_LABEL_NOTE in content


def _looks_degraded_txt(path: Path) -> bool:
    """True if an existing .speakers.txt carries only generic labels.

    Every content line matches the bare ``Speaker (HH:MM:SS): `` form; a
    diarized txt has letterized or real-name labels on at least one line.
    An empty file has no speaker names to destroy and reads as degraded.
    Unreadable files read as named — when in doubt, keep.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    lines = [ln for ln in content.splitlines() if ln.strip()]
    if not lines:
        return True
    return all(_DEGRADED_TXT_LINE.match(ln) for ln in lines)


# Provenance line rendered inside degraded (unlabeled) HTML transcripts: a
# generic-label page is otherwise indistinguishable from a one-speaker
# diarized transcript when the file is opened later without the run's
# stderr, so the durable file has to self-identify.
_GENERIC_LABEL_NOTE = "No speaker diarization — every cue carries a generic 'Speaker' label."


def _write_html_transcript(
    merged: list[tuple[MR.WhisperSeg, str]],
    of_base: Path,
    frames_dir: Path | None,
    title: str,
    note: str = "",
    transcript_txt: bool = False,
) -> tuple[list[Path], list[Path]]:
    """Write the unlabeled (degraded) HTML transcript shared by both fallbacks.

    Used when diarization produced no labels but HTML was explicitly
    requested: every cue gets a generic ``Speaker`` label, with ``note``
    rendered as a muted provenance line in the page. When ``transcript_txt``
    is set (audio runs — no frames manifest exists there), also writes a
    generic-label ``.speakers.txt`` so ``whiz analyze``, which needs a frames
    manifest or a ``.speakers.txt``, still finds a transcript.

    Returns ``(written, kept)`` — the paths this run wrote and the existing
    outputs it refused to touch. Per file: an existing output that carries
    real speaker labels (an earlier diarized — possibly named — run's file)
    is KEPT with a warning, because this degraded rewrite would collapse
    every cue to one generic ``Speaker`` (``format_dialogue_txt`` coalesces
    same-label lines), destroying the named data. An existing output that is
    ITSELF degraded (provenance note in the HTML / all-generic label lines
    in the txt) is overwritten: there are no speaker names in it to destroy,
    and keeping it would break idempotence — a re-run with a different
    --model/--language or audio must be able to refresh the degraded
    transcript (the guard protects named data, not whiz's own degraded
    output from an earlier fallback).
    """
    def _kept(existing: Path) -> None:
        ui.status(
            f"{existing.name} already exists — kept (this run had no speaker "
            "labels; a degraded rewrite would destroy its speaker names).",
            kind="warn",
            detail=str(existing),
        )

    def _write(out: Path, text: str, label: str) -> Path:
        if out.exists():
            ui.muted(
                f"{out.name}: existing file is also degraded (no speaker "
                "labels to protect) — overwriting."
            )
        out.write_text(text, encoding="utf-8")
        ui.wrote(label, out)
        return out

    written: list[Path] = []
    kept: list[Path] = []
    html_out = Path(str(of_base) + ".speakers.html")
    if html_out.exists() and not _looks_degraded_html(html_out):
        _kept(html_out)
        kept.append(html_out)
    else:
        written.append(_write(
            html_out,
            MR.format_speakers_html(merged, frames_dir=frames_dir, title=title, note=note),
            "Wrote HTML transcript",
        ))
    if transcript_txt:
        txt_out = Path(str(of_base) + ".speakers.txt")
        if txt_out.exists() and not _looks_degraded_txt(txt_out):
            _kept(txt_out)
            kept.append(txt_out)
        else:
            written.append(_write(
                txt_out,
                MR.format_dialogue_txt(merged) + "\n",
                "Wrote dialogue TXT",
            ))
    return written, kept


def _run_diarize_or_fallback(wav: Path, config: cfg.Config, args: argparse.Namespace) -> list[D.DiarSegment]:
    """Run diarization, returning [] and a hint if it is unavailable.

    Unavailability is the TYPED ``D.DiarizationUnavailable`` (missing
    package / models / failed config validation): an explicitly-requested
    diarization (``--speakers``) surfaces a clear hint but still returns []
    so the caller can fall back to the unlabeled (or screenshots-only) path
    instead of crashing. A truly transient failure (plain RuntimeError)
    is re-raised.
    """
    num_sp = args.speakers if args.speakers else 0
    thr = args.cluster_threshold if args.cluster_threshold is not None else config.cluster_threshold
    try:
        diar_segments = D.run_diarization(wav, config, num_speakers=num_sp, threshold=thr)
    except D.DiarizationUnavailable as e:
        # M2 (wave-1 audit): setup/unavailability problems arrive as the
        # TYPED DiarizationUnavailable (missing package / models / failed
        # config validation) — the old string matcher over RuntimeError
        # text ('sherpa_onnx'/'models not found'/'download-diarization')
        # missed the validate-failure path entirely and tied degradation
        # to message wording. Transient RuntimeErrors are NOT caught here
        # and stay loud, exactly as before.
        msg = str(e)
        # Explicitly requested --speakers degrades loudly (warn); merely
        # auto-enabled diarization stays a quiet hint (mirrors cmd_merge).
        kind = "warn" if args.speakers is not None else "hint"
        # The detail must describe what happens NEXT, not just what
        # failed: generic-label artifacts when the fallback will write
        # them (mirrors cmd_merge's wording), a plain skip otherwise.
        lead = ("Falling back to generic 'Speaker' labels. Enable with: "
                if _will_write_generic_labels(args) else
                "Skipping speaker labels for this run. Enable with: ")
        detail = lead + "pipx inject whiz 'whiz[diarize]' && whiz models download-diarization"
        extra = _discarded_naming_detail(args)
        if extra:
            detail += f" {extra}"
        ui.status(f"Speakers: diarization unavailable — {msg.splitlines()[0]}",
                  kind=kind,
                  detail=detail)
        return []
    if not diar_segments:
        # Same what-happens-next rule as the except branch above: the
        # fallback writes generic-label artifacts on video/explicit-html
        # runs; with nothing to write, it is a plain skip.
        outcome = ("falling back to unlabeled output"
                   if _will_write_generic_labels(args) else
                   "skipping speaker labels for this run")
        ui.status(f"Warning: diarization produced no segments; {outcome}.",
                  kind="warn",
                  detail=_discarded_naming_detail(args))
        _print_zero_segments_hints()
    return diar_segments


def _diarization_audio(wav: Path, config: cfg.Config, stack: contextlib.ExitStack) -> Path:
    """``wav`` itself when diarization can read it, else a temp 16 kHz copy.

    whisper-cli decodes m4a/mp3/44.1 kHz WAV on its own, but the diarization
    reader takes only 16 kHz mono 16-bit PCM — handing it anything else used
    to crash the run with an uncaught ``wave.Error`` after (or, in
    transcribe, before) the whole whisper pass. The temp copy is removed when
    ``stack`` closes.
    """
    if aud.is_diarization_ready(wav):
        return wav
    tmp = Path(tempfile.mkdtemp(prefix="whiz-diar-"))
    stack.callback(shutil.rmtree, tmp, ignore_errors=True)
    ui.muted(f"Converting {wav.name} to 16 kHz mono for diarization")
    return aud.extract_audio(wav, aud.find_ffmpeg(config.ffmpeg), dest_dir=tmp)


def _remove_intermediate_audio(wav: Path, in_path: Path, keep_wav: bool, *, quiet: bool = False) -> None:
    """Delete the WAV whiz extracted for this run — and nothing else.

    A temp-dir extraction (the usual spot held a foreign file) goes with its
    dir; ``--keep-wav`` keeps it and says where. Beside the source, only a
    file carrying whiz's extraction marker for THIS source is deleted, so a
    user's same-named recording can never be removed as "intermediate".
    """
    if wav == in_path or not wav.exists():
        return
    if wav != _naming_wav(in_path, wav):
        if keep_wav:
            if not quiet:
                ui.status(f"Kept extracted audio in a temp dir: {wav}", kind="info")
            return
        shutil.rmtree(wav.parent, ignore_errors=True)
        return
    if keep_wav or not aud.is_whiz_extracted(wav, in_path):
        return
    try:
        wav.unlink()
        if not quiet:
            ui.muted(f"Removed intermediate {wav}")
    except OSError:
        pass


def cmd_transcribe(args: argparse.Namespace) -> int:
    config = cfg.load()
    built = _build_transcribe_args(args, config)
    wav, in_path, keep_wav = built[2], built[3], built[4]
    # The extracted WAV (~115 MB/hour) and any diarization temp copy are
    # removed on EVERY exit — ^C during whisper or diarization included —
    # not only on the success path.
    with contextlib.ExitStack() as stack:
        if not args.dry_run:
            stack.callback(_remove_intermediate_audio, wav, in_path, keep_wav, quiet=True)
        rc = _run_transcribe(args, config, built, stack)
    if not args.dry_run:
        of_base = built[5]
        _remember(in_path, of_base,
                  _find_whisper_json(of_base, _naming_wav(in_path, wav), of_passed=bool(args.output)))
    return rc


def _remember(in_path: Path, of_base: Path, whisper_json: Path | None) -> None:
    """Add the run to the library (whiz mcp, search) — whatever it wrote,
    even on a degraded exit. Never fails the run: the transcript exists
    either way."""
    try:
        L.remember(in_path, of_base, whisper_json)
    except OSError as exc:
        ui.muted(f"Could not add this recording to the library ({L.library_path()}): {exc}")


def _run_transcribe(
    args: argparse.Namespace,
    config: cfg.Config,
    built: tuple,
    stack: contextlib.ExitStack,
) -> int:
    cmd, model_path, wav, in_path, keep_wav, of_base, diarize_enabled, screenshots = built
    naming_wav = _naming_wav(in_path, wav)

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
    json_path = _find_whisper_json(of_base, naming_wav, of_passed=bool(args.output))
    resuming = bool(getattr(args, "resume", False) and json_path.exists())
    diar_segments: list[D.DiarSegment] = []
    diar_wav = _diarization_audio(wav, config, stack) if diarize_enabled else wav
    if resuming:
        ui.info(f"--resume: found existing whisper JSON {json_path}; skipping transcription.")
        rc = 0
        # Diarization still runs so a new --speakers count / threshold takes
        # effect against the existing transcription.
        if diarize_enabled:
            ui.phase("diarizing")
            diar_segments = _run_diarize_or_fallback(diar_wav, config, args)
    else:
        # --- Diarization path ---
        if diarize_enabled:
            ui.phase("diarizing")
            diar_segments = _run_diarize_or_fallback(diar_wav, config, args)

        ui.phase("transcribing")
        proc = _run_whisper_streaming(cmd)
        rc = proc.returncode

    # --- Merge diarization with whisper output ---
    written: list[str] = []
    kept_outputs: list[Path] = []
    want_html = _outputs_include(args, config, "html")
    # The degraded (unlabeled) fallback honors only an EXPLICIT --outputs
    # html; config-supplied html keeps master's skip when diarization
    # yields nothing (a typed flag is a promise; a config default describes
    # the success path).
    explicit_html = _outputs_explicitly_include(args, "html")
    want_frames = screenshots and aud.needs_extraction(in_path)
    whisper_segs: list[MR.WhisperSeg] = []
    json_missing = False
    if (diarize_enabled or want_frames or want_html) and rc == 0:
        json_path = _find_whisper_json(of_base, naming_wav, of_passed=bool(args.output))
        if not json_path.exists():
            json_missing = True
            ui.status(f"Warning: expected whisper JSON output at {json_path} but it's missing; skipping merge.",
                      kind="warn")
        else:
            try:
                whisper_segs = MR.parse_whisper_json(json_path)
            except Exception as e:  # noqa: BLE001
                json_missing = True
                ui.status(f"Warning: failed to parse {json_path}: {e}", kind="warn")
                whisper_segs = []

        if diarize_enabled and whisper_segs and diar_segments:
            ui.phase("merging speakers")
            merged = MR.assign_speakers(whisper_segs, diar_segments)
            # Voice profiles: compute per-cluster embeddings and auto-match
            # against any stored profiles. The match seeds speaker names
            # (used as defaults); --speakers-names/--name-speakers can override.
            profile_names: dict[str, str] = {}
            cluster_embeddings: dict[int, list[float]] = {}
            if not args.no_voice_profiles:
                try:
                    cluster_embeddings = P.compute_speaker_embeddings(diar_wav, diar_segments, config)
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
            srt_out, txt_out, html_out, name_map = _write_labeled_outputs(
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
            if want_html and not want_frames and html_out is not None:
                # Audio diarized run: no frames pass follows, so announce
                # the HTML the helper wrote here — after srt/txt, matching
                # the order the summary panel lists the files in.
                ui.wrote("Wrote HTML transcript", html_out)
                written.append(str(html_out))
            # Apply the resolved names to the caller's merged list so the
            # screenshots manifest and the HTML pass carry real names too
            # (_write_labeled_outputs relabels a local copy only).
            if name_map:
                merged = MR.relabel(merged, name_map)
            # Video screenshots: one frame per segment, using the relabeled
            # merged list so the manifest carries final speaker names.
            frames_dir = None
            manifest_kept = False
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
                    frames_dir, manifest_path, manifest_kept = result
                    if not manifest_kept:
                        ui.wrote("Wrote frames manifest", manifest_path)
                        written.append(str(manifest_path))
                    else:
                        # H5: a named manifest from an earlier diarized run
                        # was KEPT, not overwritten — count it so a kept-only
                        # run reads as a no-op success (rc=0), mirroring
                        # `_write_html_transcript`'s kept contract.
                        kept_outputs.append(manifest_path)
            # Write HTML after frames exist so they can be inlined.
            if want_html and want_frames and frames_dir is not None:
                ui.phase("writing HTML transcript")
                _srt, _txt, html_out, _map = _write_labeled_outputs(
                    merged, of_base,
                    name_speakers=False,
                    speakers_names=None,
                    html=True,
                    frames_dir=frames_dir,
                    title=in_path.name,
                )
                if html_out is not None:
                    ui.wrote("Wrote HTML transcript", html_out)
                    written.append(str(html_out))
        elif whisper_segs and (want_frames or explicit_html):
            # --- Unlabeled fallback ---
            # Diarization was requested but produced no segments (sherpa-onnx
            # or its models missing, or no speech clusters found) — or the run
            # never had speaker labels at all. Honor --screenshots and an
            # EXPLICIT --outputs html with a generic "Speaker" label instead
            # of silently skipping them; config-supplied html keeps master's
            # quieter skip (a typed flag is a promise, a config default
            # describes the success path). The labeled .speakers.srt is NOT
            # faked here: it requires real diarization. The degraded writer
            # never overwrites existing speaker outputs, and on audio runs
            # (no frames manifest for `whiz analyze` to fall back on) it
            # writes a generic-label .speakers.txt so analyze still works.
            unlabeled = [(seg, "Speaker") for seg in whisper_segs]
            frames_dir = None
            manifest_kept = False
            if want_frames:
                ui.phase("capturing frames")
                width = args.screenshot_width if args.screenshot_width is not None else 1280
                result = _extract_and_manifest_screenshots(
                    in_path, unlabeled, of_base,
                    ffmpeg=aud.find_ffmpeg(config.ffmpeg),
                    width=width,
                    dry_run=args.dry_run,
                )
                if result is not None:
                    frames_dir, manifest_path, manifest_kept = result
                    if not manifest_kept:
                        ui.wrote("Wrote frames manifest", manifest_path)
                        written.append(str(manifest_path))
                    else:
                        kept_outputs.append(manifest_path)
            if explicit_html:
                ui.phase("writing HTML transcript")
                fallback_written, fallback_kept = _write_html_transcript(
                    unlabeled, of_base, frames_dir, in_path.name,
                    note=_GENERIC_LABEL_NOTE,
                    transcript_txt=not want_frames,
                )
                written.extend(str(p) for p in fallback_written)
                kept_outputs.extend(fallback_kept)
            elif want_html and not explicit_html:
                # L (wave-1 audit): config-supplied html was skipped on this
                # degraded run — say so, and NAME the format (a generic
                # "skipping outputs" would leave the user wondering which
                # one). The explicit path above explains itself.
                ui.status(
                    "html output (from config, not --outputs) skipped: no "
                    "speaker labels to write it with — pass --outputs html to "
                    "write a generic-label transcript.",
                    kind="hint",
                )
        elif whisper_segs and want_html and not explicit_html:
            # L (wave-1 audit): config-supplied html with no speaker labels
            # AND no frames/explicit-html fallback to enter — master's quiet
            # skip. Still say so, and NAME the format, so the user isn't left
            # wondering which output never appeared.
            ui.status(
                "html output (from config, not --outputs) skipped: no speaker "
                "labels to write it with — pass --outputs html to write a "
                "generic-label transcript anyway.",
                kind="hint",
            )

    # Clean up the intermediate WAV unless asked to keep it. (cmd_transcribe
    # repeats this quietly on any exit path; here it is visible.)
    _remove_intermediate_audio(wav, in_path, keep_wav)

    ui.summary(written)

    # The whisper JSON every merge/frames/html output is built from was
    # missing or unparseable: an explicitly requested speakers/html output
    # was not delivered, and exit 0 would tell a wrapper (or the macOS UI)
    # that it was. The plain transcript formats whisper wrote stand.
    if json_missing and (args.speakers is not None or _outputs_explicitly_include(args, "html")):
        return 1

    # L (wave-1 audit): rc symmetry with `whiz merge`. An EXPLICIT --speakers
    # that degraded to generic labels and wrote nothing speaker-related
    # used to exit 0 here while merge exits 1 for the same outcome — a
    # wrapper could not tell the transcribe failed to deliver the requested
    # speaker labels. rc stays 0 whenever anything was written or kept (the
    # transcript itself succeeded); it goes nonzero only when the explicit
    # --speakers request produced nothing at all. A whisper failure keeps
    # whisper's own rc (checked first below via `rc == 0`).
    if (
        args.speakers is not None
        and diarize_enabled
        and rc == 0
        and not diar_segments
        and not written
        and not kept_outputs
    ):
        return 1

    # Optional: chain into AI analysis after a successful transcription.
    # Runs the same auto-detect path as `whiz analyze <file>` so the user gets
    # summary+actions or an implementation plan without a second command.
    if getattr(args, "analyze", False) and rc == 0:
        from types import SimpleNamespace
        analyze_args = SimpleNamespace(
            file=str(in_path),
            # The transcript lives under -o when one was given.
            output=args.output or "",
            # --ai-model (the native-UI flag) rides along as the analyze
            # subcommand's --model; getattr keeps hand-built Namespaces in
            # tests working when the flag is absent.
            model=getattr(args, "ai_model", "") or "",
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
        # H4 (wave-1 audit): the old `except SystemExit: pass` swallowed the
        # chained analysis failure's message AND exit code — the comment
        # claimed "the hint was already printed", but cmd_analyze raises
        # SystemExit BEFORE printing its own hint for missing transcripts;
        # either way the run reported rc=0 for a --analyze that did nothing.
        # Now the message (when non-empty) is surfaced and the exit code is
        # honored: a successful transcription stays successful (rc unchanged)
        # unless the analysis step actually failed.
        analyze_rc = 0
        try:
            analyze_rc = cmd_analyze(analyze_args) or 0
        except SystemExit as e:
            message = str(e)
            # A message-carrying SystemExit (its code IS the message string)
            # is surfaced; a bare SystemExit(3) has no message to print —
            # only its code propagates below.
            if message and not isinstance(e.code, int):
                ui.status(f"Chained analysis failed: {message.splitlines()[0]}", kind="warn")
            analyze_rc = e.code if isinstance(e.code, int) else 1
        except RuntimeError as e:
            # AI/network failures (ai.py raises RuntimeError) would otherwise
            # reach main() and skip the reassurance below.
            reason = str(e).splitlines()[0] if str(e) else repr(e)
            ui.status(f"Chained analysis failed: {reason}", kind="warn")
            analyze_rc = 1
        if analyze_rc:
            ui.status(
                "The transcription itself succeeded — the artifacts above are "
                "usable — but the chained analysis did not complete.",
                kind="warn",
                detail="Re-run analysis directly: whiz analyze <file>",
            )
            return analyze_rc

    return rc


# ---------- models ----------

def cmd_models_list(args: argparse.Namespace) -> int:
    config = cfg.load()
    found = M.discover(config)
    if getattr(args, "json", False):
        # Data only, even when empty: the app renders "no models" itself.
        import json

        print(json.dumps(
            [{"alias": m.alias, "size_mb": round(m.size_mb, 1), "path": str(m.path)} for m in found],
            ensure_ascii=False,
        ))
        return 0
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
    ui.phase("downloading speech model")
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
    ui.phase("downloading silence model")
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


# ---------- doctor / setup ----------

_WHISPER_INSTALL_HINT = "brew install whisper-cpp"
_FFMPEG_INSTALL_HINT = "brew install ffmpeg"


def _tool_path(configured: str, *names: str) -> str | None:
    """A configured tool path if it exists, else the first name on PATH."""
    if configured:
        expanded = Path(configured).expanduser()
        return str(expanded) if expanded.exists() else shutil.which(configured)
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def _doctor_report(config: cfg.Config, *, check_ai: bool = False) -> dict:
    """What a run needs, and whether this machine has it.

    One report for two readers: ``whiz doctor`` in a terminal, and the macOS
    app's setup assistant (``--json``). Every check is a filesystem / PATH
    lookup except the opt-in AI probe (``check_ai``), which talks to the
    configured server. ``ready`` means a plain transcription can run;
    speakers and AI are optional extras with their own ``ok``.
    """
    whisper = _tool_path(config.whisper_cli, "whisper-cli", "whisper")
    ffmpeg = _tool_path(config.ffmpeg, "ffmpeg", aud.MEDIA_HELPER)

    model_ref = config.model
    model_path = M.resolve(model_ref, config) if model_ref else M.pick_best(config)
    model_alias = M._alias_from_name(model_path.name) if model_path else None
    model = {
        "ok": model_path is not None,
        "configured": model_ref,
        "path": str(model_path) if model_path else None,
        "alias": model_alias,
        "size_mb": round(model_path.stat().st_size / (1024 * 1024), 1) if model_path else None,
        # NS-15: a quantized model still runs (ok stays true), but one picked
        # only because nothing better is on disk is worth replacing; a
        # configured one is the user's informed choice and is not nagged.
        "quantized": bool(model_alias and M.is_quantized(model_alias)),
        "fix": f"whiz models download {model_ref or 'turbo'}",
    }

    vad_path = M.find_vad_model(config)
    seg = D.find_segmentation_model(config)
    emb = D.find_embedding_model(config)
    try:
        import importlib.util

        package = importlib.util.find_spec("sherpa_onnx") is not None
    except (ImportError, ValueError):
        package = False

    ai: dict = {
        "configured": bool(config.ai_model and config.ai_base_url),
        "base_url": config.ai_base_url,
        "model": config.ai_model,
    }
    if check_ai:
        names = AI.list_ollama_models(config.ai_base_url, timeout=5) if config.ai_base_url else []
        ai["reachable"] = bool(names)
        ai["models"] = names
        ai["model_available"] = bool(config.ai_model) and config.ai_model in names

    report = {
        "version": __version__,
        "checks": {
            "whisper_cli": {"ok": whisper is not None, "path": whisper, "fix": _WHISPER_INSTALL_HINT},
            "ffmpeg": {"ok": ffmpeg is not None, "path": ffmpeg, "fix": _FFMPEG_INSTALL_HINT},
            "model": model,
            # Downloaded on first use when missing, so never blocking.
            "vad_model": {"ok": vad_path is not None, "path": str(vad_path) if vad_path else None,
                          "fix": "whiz models download-vad"},
            "speakers": {"ok": package and seg is not None and emb is not None,
                         "package": package,
                         "segmentation": str(seg) if seg else None,
                         "embedding": str(emb) if emb else None,
                         "fix": "whiz setup speakers"},
        },
        "ai": ai,
        # Inside Whiz.app the tools ship with the app: nothing to brew.
        "bundled": _bundled(),
        "brew": _tool_path("", "brew") or next(
            (b for b in ("/opt/homebrew/bin/brew", "/usr/local/bin/brew") if Path(b).exists()), None),
    }
    report["ready"] = all(report["checks"][k]["ok"] for k in ("whisper_cli", "ffmpeg", "model"))
    return report


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check what transcription needs; exit 0 only when a run can work."""
    config = cfg.load()
    report = _doctor_report(config, check_ai=getattr(args, "ai", False))
    if getattr(args, "json", False):
        import json

        print(json.dumps(report, ensure_ascii=False))
        return 0 if report["ready"] else 1

    checks = report["checks"]
    labels = [
        ("whisper_cli", "whisper.cpp", True),
        ("ffmpeg", "ffmpeg", True),
        ("model", "speech model", True),
        ("vad_model", "silence model (VAD)", False),
        ("speakers", "speaker detection", False),
    ]
    rows = []
    for key, label, required in labels:
        check = checks[key]
        if key == "model" and check["ok"] and check["quantized"] and not check["configured"]:
            state = "quantized"
            detail = f"{check['alias']} costs accuracy — {check['fix']} for full quality"
        elif check["ok"]:
            state = "ok"
            detail = check.get("path") or check.get("segmentation") or ""
        else:
            state = "missing" if required else "optional"
            detail = check["fix"]
        rows.append([label, state, str(detail)])
    ai = report["ai"]
    if ai["configured"]:
        if "reachable" in ai:
            state = "ok" if ai["model_available"] else ("model missing" if ai["reachable"] else "unreachable")
        else:
            state = "configured"
        rows.append(["AI summary", state, f"{ai['model']} @ {ai['base_url']}"])
    else:
        rows.append(["AI summary", "optional", "whiz config set ai_model=<model>"])
    ui.table(f"whiz {report['version']}", [("Part", "left"), ("Status", "left"), ("Where / how to fix", "left")], rows)
    if report["ready"]:
        ui.status("Ready to transcribe.", kind="ok")
        return 0
    ui.status("Not ready yet — run the commands above.", kind="warn")
    return 1


def cmd_setup_speakers(args: argparse.Namespace) -> int:
    """One-time speaker-detection setup, on request.

    The same steps a diarized run performs on its own (sherpa-onnx into
    this environment, then the ~90 MB models), but asked for explicitly —
    running this command IS the consent, so nothing prompts.
    """
    config = cfg.load()
    if _diarization_available(config):
        ui.status("Speaker detection is already set up.", kind="ok")
        return 0
    try:
        import sherpa_onnx  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        ui.phase("installing speaker detection")
        if not _install_sherpa_onnx():
            return 1
    if D.find_segmentation_model(config) is None or D.find_embedding_model(config) is None:
        ui.phase("downloading speaker models")
        try:
            D.download_diarization_models()
        except Exception as e:  # noqa: BLE001
            ui.status(f"Speaker model download failed: {e}", kind="warn",
                      detail="Try again, or run: whiz models download-diarization")
            return 1
    if not _diarization_available(config):
        ui.status("Setup finished but speaker detection is still unavailable.", kind="warn",
                  detail="Check with: whiz doctor")
        return 1
    ui.status("Speaker detection is ready.", kind="ok")
    return 0


# ---------- analyze ----------
# ---------- analyze ----------

def _analysis_output_path(of_base: Path) -> Path:
    return Path(str(of_base) + ".analysis.md")


def _analysis_partial_path(of_base: Path) -> Path:
    return Path(str(of_base) + ".analysis.partial.md")


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
    if not sys.stdin.isatty():
        # Nobody to ask. The macOS UI holds stdin open for speaker answers,
        # so input() here blocked a GUI --analyze run forever; a script's
        # stdin would be consumed as a choice. Use the recommended live model
        # for THIS run and persist nothing — a pick nobody made must not
        # replace the user's configured model.
        ui.status(
            f"Using {default_name} for this run (no terminal to choose on; not saved).",
            kind="info",
            detail=f"Make it permanent with:  whiz config set ai_model={default_name}",
        )
        return default_name
    # Loop until the user picks a live model (or accepts the live default).
    while True:
        try:
            # Only reached on a TTY (the non-TTY case returned above).
            choice = input(
                f"Choose a model [1-{len(models)}] "
                f"(default {rec_idx + 1} = {default_name}): "
            ).strip()
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

    # For video inputs the manifest/transcript sit alongside, named after the
    # video stem (not the .wav). We just use the video stem directly — unless
    # transcribe was given -o, which moves every artifact under that base.
    output = getattr(args, "output", "") or ""
    of_base = Path(output).expanduser() if output else in_path.with_suffix("")
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
            # A merge that fails after 30 minutes of chunk work keeps the
            # chunk results here instead of losing them.
            partial_path=_analysis_partial_path(of_base),
        )

    # Write the .analysis.md (prompt + response) and print response to stdout.
    md = f"# whiz analysis — {in_path.name}\n\n"
    md += f"**Model:** {model}  **Vision:** {use_vision}  **Mode:** {detected_mode}\n\n"
    md += "## Prompt\n\n```\n" + prompt_template.replace("{transcript}", "<transcript omitted>") + "\n```\n\n"
    md += "## Response\n\n" + response + "\n"
    out_path = _analysis_output_path(of_base)
    out_path.write_text(md, encoding="utf-8")
    ui.wrote("Wrote analysis", out_path)
    # A leftover from an earlier failed run would read as current.
    _analysis_partial_path(of_base).unlink(missing_ok=True)
    _remember(in_path, of_base, L.discover(in_path)[1])
    # On a terminal the response is the point of the command. Over a pipe it
    # is already in .analysis.md, and echoing model-written text into the
    # stream the macOS UI parses let a reply line like "✓ Decision: ship
    # Friday" pose as an artifact.
    if sys.stdout.isatty():
        print(response)
    return 0


def _video_audio_for_reuse(in_path: Path, config: cfg.Config, stack: contextlib.ExitStack) -> Path:
    """merge / speakers match: the recording's WAV, reusing whiz's own copy.

    A ``<stem>.wav`` a ``--keep-wav`` transcribe left behind is reused (it is
    what the diarization cache is keyed on). Any OTHER file at that path —
    the user's own recording, a same-stem sibling's audio — used to be
    diarized as if it were this video's audio; now it is left alone and the
    audio is extracted to a temp dir that ``stack`` removes. A fresh
    extraction beside the video is kept, so the next merge is instant.
    """
    target = aud.extracted_wav_path(in_path)
    if target.exists() and aud.is_whiz_extracted(target, in_path):
        return target
    ui.phase("extracting audio")
    ui.kv("Video" if aud.needs_extraction(in_path) else "Recording", in_path.name)
    dest = aud.extraction_dir(in_path)
    if dest is not None:
        stack.callback(shutil.rmtree, dest, ignore_errors=True)
        ui.status(
            f"{target.name} next to the video is not whiz's own extraction of it; "
            "leaving it untouched and extracting to a temp dir.",
            kind="warn",
        )
    wav = aud.extract_audio(in_path, aud.find_ffmpeg(config.ffmpeg), dest_dir=dest)
    ui.kv("Audio", str(wav))
    return wav


def cmd_merge(args: argparse.Namespace) -> int:
    with contextlib.ExitStack() as stack:
        return _run_merge(args, stack)


def _run_merge(args: argparse.Namespace, stack: contextlib.ExitStack) -> int:
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

    # Proactive-first (user decision, 2026-09-05): merge re-runs diarization, so
    # the missing one-time setup is performed here too — same policy as
    # transcribe. Only when diarization will actually run (explicit --speakers
    # or video auto-enable); --no-speakers merges run straight to the JSON
    # path without paying an unrelated setup.
    if speakers_requested:
        setup_ready = _ensure_diarization_ready(
            config,
            dry_run=False,
            setup_allowed=not getattr(args, "no_auto_diarization_setup", False),
        )
    else:
        setup_ready = True
    # M1 (wave-1 audit): the diarization call below is gated on
    # speakers_requested — a `whiz merge --no-speakers` (or an audio file with
    # no --speakers) must NOT pay a model-resolution attempt inside
    # run_diarization; it merges straight to the JSON path, exactly like the
    # comment above (and the 1607-1611 gate) promise.

    # Resolve the audio (WAV) to diarize. Reuse an existing sibling WAV if the
    # transcribe run kept it; otherwise re-extract (a video, or audio that
    # transcribe decoded too, so the names below match its outputs).
    if aud.needs_decoding(in_path):
        wav = _video_audio_for_reuse(in_path, config, stack)
    else:
        wav = in_path

    # Locate the whisper JSON produced by a prior transcribe run.
    naming_wav = _naming_wav(in_path, wav)
    json_path = Path(args.json).expanduser() if args.json else _find_whisper_json(naming_wav.with_suffix(""), naming_wav, of_passed=False)
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

    want_html = _outputs_include(args, config, "html")
    # Degraded outputs honor only an EXPLICIT --outputs html (a typed flag
    # is a promise; config-supplied html describes the success path and
    # keeps master's skip when diarization yields nothing).
    explicit_html = _outputs_explicitly_include(args, "html")
    want_frames = screenshots and aud.needs_extraction(in_path)
    # L (wave-1 audit): config-supplied html on a run that will produce no
    # speaker labels is skipped — say so and NAME the format, instead of
    # leaving the user wondering why no HTML appeared.
    if want_html and not explicit_html and not speakers_requested:
        ui.status(
            "html output (from config, not --outputs) will be skipped: no "
            "speaker labels on this run — pass --outputs html to request a "
            "generic-label transcript anyway.",
            kind="hint",
        )
    # M14 (wave-1 audit): the degraded-output explanation must fire whenever
    # degraded output is actually requested (explicit html or frames), even
    # when diarization never runs — a --no-speakers merge still writes the
    # unlabeled artifacts below, and the reason they carry generic labels
    # must be stated at the point the artifacts exist, not only inside the
    # diarization branch.
    if (explicit_html or want_frames) and not speakers_requested:
        ui.status(
            "Speakers: diarization not requested (--no-speakers or audio with no --speakers); "
            "any HTML transcript / frames manifest from this run carries generic 'Speaker' labels.",
            kind="info",
        )
    # Set when the diarization-unavailable status below has already said
    # what happens next (skip / fall back to generic labels): the follow-up
    # "produced no segments" block would only repeat it (review: the
    # sherpa-missing path double-warned). It stays silent for the
    # genuinely-new case — diarization RAN and returned nothing.
    degraded_note_shown = False
    diar_segments: list[D.DiarSegment] = []
    diar_wav = _diarization_audio(wav, config, stack) if speakers_requested else wav
    if speakers_requested:
        try:
            ui.phase("diarizing")
            diar_segments = D.run_diarization(diar_wav, config, num_speakers=num_sp, threshold=thr)
        except D.DiarizationUnavailable as e:
            # M2 (wave-1 audit): setup/unavailability problems arrive as the
            # typed DiarizationUnavailable (missing package / models /
            # validate failure) — not as string-matched RuntimeError text
            # (which missed the validate-failure path entirely).
            msg = str(e)
            # Speaker-naming flags are silently dropped on every fallback
            # below — say so instead of letting the user believe names applied.
            naming = _discarded_naming_detail(args)
            if speakers_auto and args.speakers is None:
                # Auto-enabled only: fall back to unlabeled output, don't crash.
                detail = ("Skipping speaker labels. Enable with: pipx inject whiz 'whiz[diarize]' "
                          "&& whiz models download-diarization")
                if naming:
                    detail += f" {naming}"
                ui.status(f"Speakers: diarization unavailable — {msg.splitlines()[0]}",
                          kind="hint",
                          detail=detail)
                degraded_note_shown = True
            elif explicit_html or want_frames:
                # Explicitly requested, but an HTML transcript / screenshots
                # can still be produced: degrade to generic 'Speaker' labels
                # instead of crashing (mirrors the `whiz transcribe` fallback).
                detail = ("Falling back to generic 'Speaker' labels. Enable with: "
                          "pipx inject whiz 'whiz[diarize]' && whiz models download-diarization")
                if naming:
                    detail += f" {naming}"
                ui.status(f"Speakers: diarization unavailable — {msg.splitlines()[0]}",
                          kind="warn",
                          detail=detail)
                degraded_note_shown = True
            else:
                raise SystemExit(
                    f"{msg}\nEnable diarization with: pipx inject whiz 'whiz[diarize]' && "
                    f"whiz models download-diarization"
                )
    if speakers_requested and not diar_segments:
        if degraded_note_shown:
            pass  # already said what happens next — a second status is noise
        elif want_frames or explicit_html:
            ui.status("Diarization produced no segments; writing unlabeled output with generic 'Speaker' labels.",
                      kind="warn",
                      detail=_discarded_naming_detail(args))
            _print_zero_segments_hints()
        else:
            detail = _discarded_naming_detail(args)
            # L (wave-1 audit): name the skipped format — config-supplied
            # html is NOT written on a degraded run (only an explicit
            # --outputs html is), so the message must say html specifically,
            # not just "nothing to merge".
            if want_html and not explicit_html:
                note = ("html output (from config, not --outputs) skipped: no "
                        "speaker labels to write it with — pass --outputs html "
                        "for a generic-label transcript.")
                detail = f"{detail} {note}" if detail else note
            ui.status("Diarization produced no segments; nothing to merge.", kind="warn",
                      detail=detail)
            _print_zero_segments_hints()
    # L (wave-1 audit): a declined/failed one-time setup followed by a
    # SUCCESSFUL diarization is surprising — it can only happen via the
    # diarization cache (models on disk + a matching cached result make
    # the missing package irrelevant). Explain it post-hoc, at the point
    # the surprise is real; the degrade paths above already explain
    # themselves.
    if speakers_requested and not setup_ready and diar_segments:
        ui.status(
            "Speakers: diarization ran WITHOUT the one-time setup — a cached "
            "diarization result was reused for this WAV.",
            kind="info",
            detail=str(D.diar_cache_path(diar_wav)),
        )

    merged = MR.assign_speakers(whisper_segs, diar_segments) if diar_segments else []

    # Speaker tally to stderr for quick tuning feedback (before relabeling).
    if merged:
        from collections import Counter
        counts = Counter(label for _, label in merged)
        ui.tally(counts.most_common(), order=MR.speakers_in_order(merged))

    # Voice profiles: compute per-cluster embeddings and auto-match against any
    # stored profiles. Matched names seed the speaker labels; --speakers-names
    # / --name-speakers can override. Embeddings are reused for profile saving.
    profile_names: dict[str, str] = {}
    cluster_embeddings: dict[int, list[float]] = {}
    if merged and not args.no_voice_profiles:
        try:
            cluster_embeddings = P.compute_speaker_embeddings(diar_wav, diar_segments, config)
            if cluster_embeddings:
                profile_names, _matches = P.auto_assign_names(
                    cluster_embeddings, threshold=config.speaker_match_threshold,
                )
                profile_names = {k: v for k, v in profile_names.items() if v}
        except Exception as e:  # noqa: BLE001
            ui.status(f"Warning: voice-profile matching skipped: {e}", kind="warn")

    written: list[str] = []
    kept_outputs: list[Path] = []
    if merged:
        ui.phase("merging speakers")
        srt_out, txt_out, html_out, name_map = _write_labeled_outputs(
            merged, of_base,
            name_speakers=_name_speakers_enabled(args, diarize_enabled=True),
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
        if want_html and not want_frames and html_out is not None:
            # Audio merge: no frames pass follows, so announce the HTML the
            # helper wrote here — after srt/txt, matching the summary order.
            ui.wrote("Wrote HTML transcript", html_out)
            written.append(str(html_out))
        # Apply the resolved names to the caller's merged list so the
        # screenshots manifest and the HTML pass carry real names too.
        if name_map:
            merged = MR.relabel(merged, name_map)

    # Video screenshots: re-extract frames against the existing merged list.
    # Frame extraction is cheap (~seconds), so merge --screenshots re-runs it.
    # When no merged list exists (diarization unavailable), frames fall back
    # to a generic 'Speaker' label so the manifest + HTML still get written.
    frames_dir = None
    manifest_kept = False
    shot_list = merged if merged else [(seg, "Speaker") for seg in whisper_segs]
    if want_frames:
        ui.phase("capturing frames")
        width = args.screenshot_width if args.screenshot_width is not None else 1280
        result = _extract_and_manifest_screenshots(
            in_path, shot_list, of_base,
            ffmpeg=aud.find_ffmpeg(config.ffmpeg),
            width=width,
            dry_run=False,
        )
        if result is not None:
            frames_dir, manifest_path, manifest_kept = result
            if not manifest_kept:
                ui.wrote("Wrote frames manifest", manifest_path)
                written.append(str(manifest_path))
            else:
                # H5: a NAMED manifest from an earlier diarized run was KEPT
                # (this run is degraded) — count it so a kept-only run reads
                # as a no-op success (rc=0), like `_write_html_transcript`.
                kept_outputs.append(manifest_path)
    # Write HTML after frames exist so they can be inlined.
    if want_html and frames_dir is not None and merged:
        ui.phase("writing HTML transcript")
        _srt, _txt, html_out, _map = _write_labeled_outputs(
            merged, of_base,
            name_speakers=False,
            speakers_names=None,
            html=True,
            frames_dir=frames_dir,
            title=in_path.name,
        )
        if html_out is not None:
            ui.wrote("Wrote HTML transcript", html_out)
            written.append(str(html_out))
    # --- Unlabeled HTML fallback ---
    # Diarization produced nothing but HTML was EXPLICITLY requested: emit
    # the transcript with generic 'Speaker' labels instead of silently
    # skipping it (mirrors the `whiz transcribe` fallback); config-supplied
    # html keeps master's skip. The labeled .speakers.srt is not faked: it
    # requires real diarization; on audio runs a generic-label
    # .speakers.txt is still written so `whiz analyze` finds a transcript
    # (it needs a frames manifest or a .speakers.txt). Existing speaker
    # outputs from an earlier diarized run are never overwritten.
    if explicit_html and not merged and whisper_segs:
        ui.phase("writing HTML transcript")
        fallback_written, fallback_kept = _write_html_transcript(
            shot_list, of_base, frames_dir, in_path.name,
            note=_GENERIC_LABEL_NOTE,
            transcript_txt=not want_frames,
        )
        written.extend(str(p) for p in fallback_written)
        kept_outputs.extend(fallback_kept)
    ui.summary(written)
    if written or kept_outputs:
        _remember(in_path, of_base, json_path)
    if not written:
        if kept_outputs:
            # Everything this run would have written already existed and was
            # correctly KEPT (named outputs from an earlier diarized run) — a
            # no-op success, not a failure. rc=1 here would false-alarm
            # `whiz merge ... || alert` wrappers on identical re-runs.
            return 0
        # E.g. diarization produced no segments and no html/screenshots
        # fallback was requested: a silent rc=0 would read as success.
        return 1
    return 0


# ---------- dictate (moved to Mynah) ----------

# Dictation left whiz for https://github.com/ReidenXerx/mynah. The command stays
# for a release or two, because `whiz dictate` is in people's muscle memory, in
# their LaunchAgent, and in a shell history they will page back through: it
# should say where its replacement went rather than "unknown command".
MYNAH_MOVED = """\
Dictation moved to Mynah — its own tool now.

  pipx install git+https://github.com/ReidenXerx/mynah.git
  mynah setup

Your settings come with it: Mynah's first run reads the dictate_* keys out of
whiz's config.toml. https://duduphudu.app/mynah/

whiz still transcribes recordings: whiz transcribe, whiz analyze, whiz merge."""


def cmd_dictate(args: argparse.Namespace) -> int:
    print(MYNAH_MOVED, file=sys.stderr)
    return 1


# ---------- speakers (voice profiles) ----------

def cmd_speakers_list(args: argparse.Namespace) -> int:
    """List stored speaker voice profiles."""
    profiles = P.load_profiles()
    if getattr(args, "json", False):
        # Names and provenance only — the embeddings are biometric data and
        # the app has no use for them.
        import json

        print(json.dumps([
            {"name": prof.name, "samples": prof.samples, "created": prof.created,
             "source": prof.source, "path": str(P._profile_path(prof.name))}
            for prof in profiles
        ], ensure_ascii=False))
        return 0
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
    with contextlib.ExitStack() as stack:
        return _run_speakers_match(args, stack)


def _run_speakers_match(args: argparse.Namespace, stack: contextlib.ExitStack) -> int:
    """Show how a recording's clusters match against stored profiles (dry run).

    Runs diarization on the given file and prints the cosine-similarity scores
    of each cluster against every stored profile, plus the auto-assignment
    decision at the configured threshold. "Dry run" means nothing is
    relabeled or saved — it does NOT mean the machine stays untouched: this
    command needs diarization by definition, so the one-time setup may run
    first (consent prompt on a TTY; opt out with --no-auto-diarization-setup).
    """
    config = cfg.load()
    in_path = Path(args.file).expanduser()
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")
    if aud.needs_decoding(in_path):
        wav = _video_audio_for_reuse(in_path, config, stack)
    else:
        wav = in_path

    num_sp = args.speakers if args.speakers else 0
    thr = args.cluster_threshold if args.cluster_threshold is not None else config.cluster_threshold
    # Proactive-first: this command needs diarization by definition; attempt
    # the one-time setup before failing (unless the caller opts out).
    if not _ensure_diarization_ready(
        config,
        dry_run=False,
        setup_allowed=not getattr(args, "no_auto_diarization_setup", False),
    ):
        raise SystemExit(
            "Diarization unavailable (sherpa-onnx or models missing, setup "
            "failed or opted out).\n"
            "Run manually: pipx inject whiz 'whiz[diarize]' && "
            "whiz models download-diarization"
        )
    ui.phase("diarizing")
    diar_wav = _diarization_audio(wav, config, stack)
    diar_segments = D.run_diarization(diar_wav, config, num_speakers=num_sp, threshold=thr)
    if not diar_segments:
        raise SystemExit("Diarization produced no segments.")

    profiles = P.load_profiles()
    if not profiles:
        ui.info(f"No stored voice profiles in {P.profiles_dir()}; nothing to match against.")
        return 0

    cluster_embeddings = P.compute_speaker_embeddings(diar_wav, diar_segments, config)
    from whiz.merge import speaker_label
    matches = P.match_speakers(cluster_embeddings, profiles, threshold=config.speaker_match_threshold)
    rows = []
    for cid, emb in sorted(cluster_embeddings.items()):
        # M3 (wave-1 audit): cosine_similarity returns None when a stored
        # profile's dim doesn't match the run's embeddings (embedding model
        # swapped) — the pair is NOT comparable, so it renders n/a instead of
        # crashing the sort (None vs float) or the score format. When every
        # profile is incomparable there is no best score at all.
        scored: list[tuple[float, str]] = []
        skipped: list[str] = []
        for prof in profiles:
            s = P.cosine_similarity(emb, prof.embedding)
            if s is None:
                skipped.append(prof.name)
            else:
                scored.append((s, prof.name))
        scored.sort(reverse=True)
        all_str = ", ".join(
            [f"{nm}={s:.3f}" for s, nm in scored]
            + [f"{nm}=n/a" for nm in skipped]
        )
        m = matches.get(cid)
        best = f"{m[0]}" if m else "(no match)"
        if m:
            best_score = f"{m[1]:.3f}"
        elif scored:
            best_score = f"{scored[0][0]:.3f}"
        else:
            best_score = "n/a"
        rows.append([speaker_label(cid), best, best_score, all_str])
    ui.table(
        "Speaker match (dry run)",
        [("Cluster", "left"), ("Best name", "left"), ("Best score", "right"), ("All scores", "left")],
        rows,
    )
    ui.muted(f"Threshold: {config.speaker_match_threshold}")
    return 0


# ---------- config ----------

def cmd_library_list(args: argparse.Namespace) -> int:
    calls = L.calls()
    if args.json:
        print(json.dumps([{
            "id": c.id, "input": str(c.input), "title": c.title,
            "recorded": c.recorded().isoformat(timespec="seconds"),
            "duration": round(c.duration, 1), "speakers": c.speakers(),
            "has_summary": c.has_summary, "has_frames": c.has_frames,
        } for c in calls], ensure_ascii=False, indent=2))
        return 0
    if not calls:
        ui.muted(f"The library is empty ({L.library_path()}). Transcribed recordings are added as they finish; "
                 "add earlier ones with: whiz library add <recording>…")
        return 0
    for c in calls:
        speakers = ", ".join(c.speakers()) or "-"
        print(f"{c.id}  {c.recorded().astimezone():%Y-%m-%d %H:%M}  {_fmt_elapsed(c.duration):>8}  {c.title}  ({speakers})")
    return 0


def cmd_library_add(args: argparse.Namespace) -> int:
    """Register recordings transcribed before the library existed."""
    failed = 0
    for file in args.files:
        in_path = Path(file).expanduser()
        if not in_path.exists():
            ui.status(f"No such file: {in_path}", kind="warn")
            failed += 1
            continue
        base, whisper_json = L.discover(in_path)
        call = L.remember(in_path, base, whisper_json)
        if call is None:
            ui.status(f"No transcript beside {in_path.name}: transcribe it first", kind="warn")
            failed += 1
        else:
            ui.status(f"Added {in_path.name}", kind="ok", detail=call.id)
    return 1 if failed else 0


def cmd_library_forget(args: argparse.Namespace) -> int:
    """Hide a recording from the library (its files stay)."""
    L.forget(Path(args.file).expanduser())
    ui.status(f"Forgot {Path(args.file).name}; its files are untouched", kind="ok")
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from whiz import mcp

    return mcp.serve()


def cmd_config_show(args: argparse.Namespace) -> int:
    config = cfg.load()
    if getattr(args, "json", False):
        return _print_config_json(config)
    print(f"# {cfg.CONFIG_PATH}")
    print()
    for k, v in config.to_dict().items():
        if k in _SECRET_CONFIG_KEYS and isinstance(v, str) and v:
            # Terminal scrollback, screen shares and pasted bug reports are
            # where `config show` output ends up.
            print(f"{k} = {_mask_secret(v)!r}")
        elif isinstance(v, list):
            print(f"{k} = {v}")
        elif isinstance(v, str) and v == "":
            print(f'{k} = ""')
        elif v is None:
            # Tri-state fields (auto_diarization_setup) print as unset —
            # a raw {v!r} would show the Python None repr.
            print(f"{k} = <unset>")
        else:
            print(f"{k} = {v!r}")
    print()
    print("Model search dirs:")
    for d in cfg.model_search_dirs(config):
        marker = "+" if d.exists() else "-"
        print(f"  {marker} {d}")
    return 0


_SECRET_CONFIG_KEYS = {"ai_api_key"}


def _print_config_json(config: cfg.Config) -> int:
    """``config show --json``: the settings as data, for the macOS app.

    The text form prints Python reprs (``True``, ``['srt', 'json']``) that a
    second language would have to re-parse by guesswork. Secrets stay masked
    here too — the app only needs to know a key IS set (``secrets_set``),
    and stdout of a GUI's child process ends up in logs like any other.
    """
    import json

    values: dict[str, object] = {}
    secrets_set: dict[str, bool] = {}
    for k, v in config.to_dict().items():
        if k in _SECRET_CONFIG_KEYS:
            secrets_set[k] = bool(v)
            values[k] = _mask_secret(v) if isinstance(v, str) and v else ""
        else:
            values[k] = v
    payload = {
        "path": str(cfg.CONFIG_PATH),
        "values": values,
        "secrets_set": secrets_set,
        "model_search_dirs": [str(d) for d in cfg.model_search_dirs(config)],
    }
    print(json.dumps(payload, default=str, ensure_ascii=False))
    return 0


def _mask_secret(value: str) -> str:
    """``sk-…1234`` — enough to tell keys apart, not enough to use one."""
    return "…" if len(value) <= 8 else f"{value[:3]}…{value[-4:]}"


def cmd_config_edit(args: argparse.Namespace) -> int:
    import shlex

    config = cfg.load()
    path = cfg.save(config)
    # $EDITOR is a command line, not a program name: `code -w` and
    # `subl --wait` are the usual values and need splitting.
    editor = shlex.split(os.environ.get("EDITOR", "") or "vi")
    try:
        return subprocess.run([*editor, str(path)]).returncode
    except FileNotFoundError:
        raise SystemExit(f"Editor not found: {editor[0]} (from $EDITOR). Config is at {path}")


_TRUE_WORDS = {"1", "true", "yes", "on"}
_FALSE_WORDS = {"0", "false", "no", "off"}


def _coerce(value: str, field_type: type):
    # With `from __future__ import annotations`, dataclass field types are
    # strings (e.g. "bool") not the type objects. Normalize to a string name.
    ft = field_type if isinstance(field_type, str) else getattr(field_type, "__name__", str(field_type))
    if ft in ("bool", "bool | None", "Optional[bool]"):
        # "bool | None" arrives for tri-state fields (auto_diarization_setup):
        # the string must NOT fall through to the raw-string branch below —
        # `whiz config set auto_diarization_setup=false` would otherwise
        # store the STRING "false", which is truthy on every later load.
        word = value.lower()
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
        if word == "" and ft != "bool":
            # Empty clears a tri-state back to "never answered".
            return None
        # A typo (`vad=ture`) used to save False without a word.
        raise SystemExit(
            f"Expected a boolean ({', '.join(sorted(_TRUE_WORDS | _FALSE_WORDS))}), got {value!r}"
        )
    if ft == "int":
        return int(value)
    if ft == "float":
        return float(value)
    if ft.startswith("list"):
        # The annotation arrives as "list[str]"; matching only "list" let
        # `config set outputs=srt,txt` store the STRING "srt,txt", which every
        # later run iterated character by character.
        return [v.strip() for v in value.split(",") if v.strip()]
    return value


# Enum-like config fields with a fixed set of allowed values. Shared by both
# Enum-like config fields, validated wherever they are set so a typo cannot
# silently degrade behaviour.
_CONFIG_ENUM_VALUES: dict[str, set[str]] = {}


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
    if getattr(args, "stdin", False):
        # `config set ai_api_key --stdin`: the value never appears in argv,
        # where every local process can read it (ps). One line; the trailing
        # newline a pipe or a heredoc adds is not part of the value.
        if "=" in assignment:
            raise SystemExit("With --stdin, pass only the KEY (the value is read from stdin)")
        key, value = assignment, sys.stdin.readline().rstrip("\r\n")
    elif "=" not in assignment:
        raise SystemExit("Expected KEY=VALUE (e.g. whiz config set threads=8)")
    else:
        key, _, value = assignment.partition("=")
    key = key.strip()
    if key not in cfg.Config.__dataclass_fields__:
        raise SystemExit(f"Unknown config key '{key}'. Valid: {', '.join(cfg.Config.__dataclass_fields__)}")
    field_type = cfg.Config.__dataclass_fields__[key].type
    coerced = _coerce(value.strip(), field_type)
    _validate_config_value(key, coerced)
    setattr(config, key, coerced)
    path = cfg.save(config)
    if key in _SECRET_CONFIG_KEYS and isinstance(coerced, str) and coerced:
        print(f"Set {key} = {_mask_secret(coerced)!r}")
    else:
        print(f"Set {key} = {coerced!r}")
    print(f"Saved to {path}")
    return 0


# ---------- upgrade ----------

# The canonical install source. pipx installs from this git URL, so `whiz
# upgrade` re-runs the same install to pull the latest commit.
_INSTALL_SOURCE = "git+https://github.com/ReidenXerx/whiz.git"


def _diarize_extra_installed() -> bool:
    """True if sherpa-onnx (the 'diarize' extra) is importable."""
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return False
    return True


def _run_live(cmd: list[str]) -> int:
    """Run a command streaming stdout/stderr to the terminal (not captured).

    Unlike service._run (capture_output), this lets the user see pipx's
    download/install progress in real time. Returns the exit code.
    """
    try:
        proc = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        ui.status(f"{cmd[0]} not found on PATH", kind="warn")
        return 127
    return proc.returncode


def _tool_installer() -> str | None:
    """How this whiz was installed: "pipx", "uv" (uv tool), or None.

    Read from the running interpreter's venv: pipx keeps tools under
    ``…/pipx/venvs/<name>``, uv under ``…/uv/tools/<name>``. Anything else
    (a pip install, a dev checkout) is not something upgrade can redo —
    reinstalling through pipx there used to create a SECOND whiz that the
    user's PATH never runs.
    """
    parts = Path(sys.prefix).resolve().parts
    for i in range(len(parts) - 1):
        if parts[i] == "pipx" and parts[i + 1] == "venvs":
            return "pipx"
        if parts[i] == "uv" and parts[i + 1] == "tools":
            return "uv"
    return None


def cmd_upgrade(args: argparse.Namespace) -> int:
    """Reinstall whiz from git, keeping the extras that were already there.

    The trap this closes: `pipx install --force` gives you a new whiz, but an
    extra installed alongside it (diarization) is not re-injected, so the next
    run fails on an import that worked yesterday. Re-inject what was installed,
    and only what was installed — a transcription-only user should not be
    surprised by a download they never asked for.
    """
    ui.header("whiz", "upgrade")

    if _bundled():
        ui.status("This whiz is part of Whiz.app; it updates with the app.", kind="warn",
                  detail="Install the new version of Whiz.app to upgrade.")
        return 1

    installer = _tool_installer()
    if installer is None:
        ui.status(
            f"This whiz is not a pipx or uv tool install (running from {sys.prefix}); "
            "upgrade it the way it was installed.",
            kind="warn",
            detail=f"pipx:  pipx install --force {_INSTALL_SOURCE}\n"
                   f"uv:    uv tool install --force {_INSTALL_SOURCE}",
        )
        return 1

    had_diarize = _diarize_extra_installed()

    if installer == "uv":
        # uv tool installs carry their extras in the one spec, so there is no
        # separate inject step to forget.
        spec = f"whiz[diarize] @ {_INSTALL_SOURCE}" if had_diarize else _INSTALL_SOURCE
        ui.phase("reinstalling whiz")
        rc = _run_live(["uv", "tool", "install", "--force", "--reinstall", spec])
        if rc != 0:
            ui.status(f"uv tool install failed (exit {rc}) — nothing else was changed", kind="warn")
            return 1
        ui.status("whiz reinstalled" + (" with the diarize extra" if had_diarize else ""), kind="ok")
        ui.muted("\nDictation lives in Mynah now: https://github.com/ReidenXerx/mynah")
        return 0

    ui.phase("reinstalling whiz")
    rc = _run_live(["pipx", "install", "--force", _INSTALL_SOURCE])
    if rc != 0:
        ui.status(f"pipx install failed (exit {rc}) — nothing else was changed", kind="warn")
        return 1
    ui.status("whiz reinstalled", kind="ok")

    if had_diarize:
        ui.phase("refreshing the diarize extra")
        rc = _run_live(["pipx", "inject", "whiz", "whiz[diarize]"])
        if rc != 0:
            ui.status(
                f"pipx inject whiz[diarize] failed (exit {rc}). Speaker detection may be "
                "stale — re-run: pipx inject whiz 'whiz[diarize]'",
                kind="warn",
            )
            # An "OK" upgrade that left speaker detection broken is not OK.
            return 1
        else:
            ui.status("diarize extra refreshed", kind="ok")
    else:
        ui.muted("diarize extra not installed — skipping")

    ui.muted("\nDictation lives in Mynah now: https://github.com/ReidenXerx/mynah")
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
    t.add_argument("--print-progress", action="store_true", help="Print progress (force on; default: on unless --no-progress)")
    t.add_argument("--no-progress", dest="no_progress", action="store_true", help="Disable whisper-cli progress passthrough (forces -np)")
    t.add_argument("--keep-wav", action="store_true", help="Keep the intermediate extracted WAV (default: deleted after)")
    t.add_argument("--no-auto-vad-download", action="store_true", help="Don't auto-download the Silero VAD model when VAD is enabled and missing")
    t.add_argument("--no-auto-diarization-setup", dest="no_auto_diarization_setup", action="store_true", help="Don't auto-install sherpa-onnx / auto-download diarization models when diarization is enabled and missing (one-time setup, ~90 MB)")
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
    t.add_argument("--ai-model", dest="ai_model", default="", help="With --analyze, the AI model for the chained analysis (default: config ai_model, e.g. llava, qwen3.5:9b). Mirrors --model on `whiz analyze`; the native macOS UI passes it.")
    t.set_defaults(func=cmd_transcribe)

    # merge
    mg = sub.add_parser("merge", help="Re-run diarization + merge against an existing whisper JSON (skip transcription)")
    mg.add_argument("file", help="Input audio/video file (used to find the whisper JSON and re-extract WAV if needed)")
    mg.add_argument("--json", default="", help="Explicit path to the whisper JSON (default: auto-find next to input)")
    mg.add_argument("--outputs", default=None, help="Comma-separated whiz post-merge output formats: html (others are whisper-cli formats, ignored here)")
    mg.add_argument("--speakers", type=int, default=None, nargs="?", const=0, help="Known speaker count; omit = auto-detect. Auto-enabled for video inputs (see --no-speakers)")
    mg.add_argument("--no-speakers", dest="no_speakers", action="store_true", help="Disable the auto-enabled speaker diarization for video inputs (opt out)")
    mg.add_argument("--no-auto-diarization-setup", dest="no_auto_diarization_setup", action="store_true", help="Don't auto-install sherpa-onnx / auto-download diarization models when diarization is enabled and missing (one-time setup, ~90 MB)")
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
    ml = msub.add_parser("list", aliases=["ls"])
    ml.add_argument("--json", action="store_true", help="Print the discovered models as JSON")
    ml.set_defaults(func=cmd_models_list)
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
    an.add_argument("-o", "--output", default="", help="Output base the transcript was written under (as passed to `whiz transcribe -o`; default: the input path without its extension)")
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
    spl = spsub.add_parser("list", aliases=["ls"])
    spl.add_argument("--json", action="store_true", help="Print the profiles as JSON (names and provenance, no embeddings)")
    spl.set_defaults(func=cmd_speakers_list)
    sf = spsub.add_parser("forget", aliases=["rm"], help="Delete a stored speaker voice profile by name")
    sf.add_argument("name", help="Speaker name to forget")
    sf.set_defaults(func=cmd_speakers_forget)
    sm = spsub.add_parser("match", help="Show how a recording's clusters match stored profiles (relabels/saves nothing — may still run the one-time diarization setup if missing)")
    sm.add_argument("file", help="Input audio/video file")
    sm.add_argument("--speakers", type=int, default=None, nargs="?", const=0, help="Known speaker count; omit = auto-detect")
    sm.add_argument("--cluster-threshold", type=float, default=None, help="Clustering threshold when auto-detecting (default 0.9)")
    sm.add_argument("--no-auto-diarization-setup", dest="no_auto_diarization_setup", action="store_true", help="Don't auto-install sherpa-onnx / auto-download diarization models when diarization is enabled and missing (one-time setup, ~90 MB)")
    sm.set_defaults(func=cmd_speakers_match)

    # library + mcp
    lb = sub.add_parser("library", aliases=["lib"], help="The recordings whiz has transcribed (what `whiz mcp` serves)")
    lbsub = lb.add_subparsers(dest="library_command", required=True)
    lbl = lbsub.add_parser("list", aliases=["ls"], help="List transcribed recordings, newest first")
    lbl.add_argument("--json", action="store_true", help="Print the list as JSON")
    lbl.set_defaults(func=cmd_library_list)
    lba = lbsub.add_parser("add", help="Add recordings transcribed before the library existed")
    lba.add_argument("files", nargs="+", help="The recordings (not their transcripts)")
    lba.set_defaults(func=cmd_library_add)
    lbf = lbsub.add_parser("forget", aliases=["rm"], help="Remove a recording from the library (its files stay)")
    lbf.add_argument("file", help="The recording")
    lbf.set_defaults(func=cmd_library_forget)
    mc = sub.add_parser("mcp", help="Serve past calls to AI agents (Claude, Cursor…) over MCP on stdin/stdout")
    mc.set_defaults(func=cmd_mcp)

    # dictate
    dt = sub.add_parser("dictate", aliases=["d"], help="Moved to Mynah (github.com/ReidenXerx/mynah)")
    dt.add_argument("rest", nargs="*", help=argparse.SUPPRESS)
    dt.set_defaults(func=cmd_dictate)

    # config
    cp = sub.add_parser("config", aliases=["c"], help="View or edit configuration")
    csub = cp.add_subparsers(dest="config_command", required=True)
    cshow = csub.add_parser("show", aliases=["cat"])
    cshow.add_argument("--json", action="store_true", help="Print the settings as JSON (secrets masked)")
    cshow.set_defaults(func=cmd_config_show)
    csub.add_parser("edit", help="Open config in $EDITOR").set_defaults(func=cmd_config_edit)
    cs = csub.add_parser("set", help="Set a value: KEY=VALUE")
    cs.add_argument("assignment", help="KEY=VALUE, e.g. threads=8 or model=turbo (just KEY with --stdin)")
    cs.add_argument("--stdin", action="store_true", help="Read the value from stdin (keeps secrets such as ai_api_key out of argv)")
    cs.set_defaults(func=cmd_config_set)

    # doctor / setup
    doc = sub.add_parser("doctor", help="Check what transcription needs (whisper.cpp, ffmpeg, a model, speakers, AI)")
    doc.add_argument("--json", action="store_true", help="Print the report as JSON")
    doc.add_argument("--ai", action="store_true", help="Also contact the AI server and check the configured model")
    doc.set_defaults(func=cmd_doctor)
    st = sub.add_parser("setup", help="One-time setup steps")
    stsub = st.add_subparsers(dest="setup_command", required=True)
    stsub.add_parser("speakers", help="Install speaker detection (sherpa-onnx + ~90 MB models)").set_defaults(func=cmd_setup_speakers)

    # upgrade
    up = sub.add_parser("upgrade", aliases=["up"], help="One-command upgrade: reinstall whiz from git (pipx or uv tool, whichever installed it), keeping the diarize extra if it was installed.")
    up.set_defaults(func=cmd_upgrade)

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