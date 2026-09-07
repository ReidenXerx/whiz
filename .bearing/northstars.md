# North-stars — whiz

Falsifiable propositions about what this project IS. This file outranks every
other doc (README, ARCHITECTURE.md, SWIFT-APP.md, code comments) — on conflict,
the north-star wins and the other source is stale. Owned by the user; agents
propose diffs, never edit silently.

## Invariants — must always hold
- **NS-1** — Every implementation (Python engine, Swift app, future Rust core) compiles the same segmentation constants, pinned against `tuning/tuning.toml` by that implementation's test suite; drift is a build failure, never a silent behavior change. — src: tests/test_tuning.py, macos/Tests/WhizAppTests/TuningTests.swift, docs/ARCHITECTURE.md
- **NS-2** — Every golden corpus case yields identical speech-region boundaries and energy-gate verdict across implementations, and the verdict is invariant under both trailing-silence policies; the generator refuses to emit a corpus where they differ. — src: tuning/golden/generate.py, docs/ARCHITECTURE.md
- **NS-3** — A tuning change lands as one commit: `tuning.toml` + every pinning test + regenerated corpus; corpus regeneration is byte-identical and test-enforced. — src: tests/test_tuning.py

## Semantics — exact meaning here
- **NS-4** — `utterance end` = start time of the first closing silent frame; buffered PCM spans [start, end + frame); Swift's 0.2 s trim removes from the tail of that span. NOT "last speech frame", NOT "buffer end". — src: tuning/golden/expected.json, docs/ARCHITECTURE.md
- **NS-5** — `rejected_by_energy_gate` = whole-buffer RMS below the calibrated (or default) utterance gate. NOT "VAD found no speech", NOT "transcription failed". — src: tuning/golden/expected.json, whiz/dictate/engine.py
- **NS-6** — Hallucination filtering = exact, trimmed, case-sensitive membership in the 21-phrase set in `tuning.toml`. NOT fuzzy or normalized matching. — src: tuning/tuning.toml, tests/test_tuning.py

## Evidence — what counts as proof
- **NS-7** — A segmentation-behavior claim requires a corpus run (existing case or a new committed case); reasoning about the state machine from source is NOT evidence. — src: docs/ARCHITECTURE.md
- **NS-8** — "Tests pass" claims must name the command actually run — pytest via the pipx venv, Swift via the sandbox-exec CLT-deny workaround (bare `swift test` is unreliable on this machine's dual toolchain; a bare claim is NOT evidence). — src: docs/ARCHITECTURE.md, local toolchain state

## Settled — decided, do not relitigate
- **NS-9** — `tuning.toml` is data, never read at runtime — the runtime-read alternative was rejected because an absent/edited file yields silently inconsistent cross-platform behavior instead of a build failure. — src: docs/ARCHITECTURE.md
- **NS-10** — Linux support is Wayland-only; X11 has no security story for global hotkey + text injection — do not propose X11 backends. — src: docs/LINUX-APP.md
- **NS-11** — Known divergences (min-utterance gate placement, trailing-silence policy, secondary VAD) are deliberate and unpinned; aligning one requires a decision + corpus update, not a quiet code change. — src: docs/ARCHITECTURE.md
- **NS-15** — Whisper models are used UNQUANTIZED, always and everywhere — quantization corrupts transcription quality (user decision, 2026-09-03; ea49da8's 4-bit-turbo garbling is the recorded evidence, and the user extends it to all quantized variants). Defaults must prefer the unquantized variant of each class (`models.py:PREFERENCE` batch order and `WhisperModel.preference` must put unquantized first and every `-q*` variant behind its unquantized class); quantized files stay available for explicit, informed use only. Preference is PER-CLASS (user + review decision, 2026-09-05): classes rank large-v3-turbo > large-v3 > medium > small > base; a quantized variant is reachable once its OWN class's unquantized model is absent — never blocked behind unrelated unquantized classes. `tiny` (and `tiny-q5_0`) is excluded from KNOWN_MODELS/PREFERENCE entirely (user decision, 2026-09-05: useless quality); it still downloads/runs when named explicitly. — src: docs/SWIFT-APP.md, whiz/models.py, macos/Sources/WhizApp/STT/WhisperModel.swift

## Graveyard — tried and rejected / validated
- **NS-12** — REJECTED: runtime tuning-file dependency (see NS-9). Re-propose only with an answer to "what happens when the file is missing mid-session".
- **NS-13** — VALIDATED: golden corpus driven through the real callback/detector, not a reimplementation — it caught the FlatTOML multi-line array drop and the shared poisoned-calibration defect. Don't replace with mock-level tests.
- **NS-14** — RESOLVED: poisoned calibration — speech filling the 1 s window drove the utterance gate to ≈3× speech RMS and silently dropped the first word in BOTH implementations. Fixed by speech-aware calibration (calibration frames ≥ `calibration_speech_floor` are excluded from the median; fewer than `noise_min_samples` quiet frames aborts calibration to the static gates). SCOPE: the fix covers speech at/above `calibration_speech_floor` (0.03) — speech BELOW the floor can still enter the median and poison it; that residual is ACCEPTED (an absolute floor trades a quiet-speech hole for speech/noise discrimination — see ARCHITECTURE.md). Corpus cases `speech_during_calibration` (now `rejected_by_energy_gate: false`) and `speech_over_noise_in_calibration` pin the fix and the exclusion mechanism; trade-off documented in ARCHITECTURE.md: steady noise ≥ 0.03 RMS is no longer calibratable (VAD + hallucination filter own it, best-effort in Python). — src: tuning/tuning.toml, docs/ARCHITECTURE.md

## Open — explicitly unresolved (do NOT assume either way)
(none — NS-15 resolved 2026-09-03)
