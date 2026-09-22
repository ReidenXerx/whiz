# North-stars — whiz

Falsifiable propositions about what this project IS. This file outranks every
other doc (README, ARCHITECTURE.md, code comments) — on conflict, the
north-star wins and the other source is stale. Owned by the user; agents
propose diffs, never edit silently.

## Invariants — must always hold

- **NS-1** — Whisper models are used UNQUANTIZED, always and everywhere — quantization corrupts transcription quality. Within each class the unquantized variant ranks first; a `-q*` variant resolves only when its OWN class's unquantized model is absent (preference is per-class, never blocking behind unrelated classes). `tiny` is excluded from KNOWN_MODELS/PREFERENCE entirely. Quantized files stay runnable when named explicitly. — src: whiz/models.py, tests/test_models.py *(carried over as old NS-15, unchanged in substance)*
- **NS-2** — A voice-profile save knows its provenance: an auto-match may CREATE a profile (`source: "auto"`) but may never MERGE into an existing one; only human confirmation merges and upgrades `auto` → `user`. A profile is discarded, not averaged, when the embedding dimension changes. — src: whiz/profiles.py, tests/test_profiles.py *(the wave-1 M3 contract, now a whiz invariant)*
- **NS-3** — The diarization cache is keyed on the parameters that produced it AND a fingerprint of the input; a stale cache can never be reused against different audio or settings. — src: whiz/diarize.py, tests/test_diarize_cache.py *(wave-1 H1)*
- **NS-4** — A degraded diarization run is loud, never silent: explicit `--speakers` writes no speaker-labeled artifacts → nonzero exit; an explicitly passed `--outputs html` is never dropped; real-labeled outputs are never clobbered by a degraded re-run; `--speakers-names` are never silently discarded. — src: whiz/cli.py, whiz/merge.py, tests/test_cli.py
- **NS-5** — `save()` preserves every config key it does not know — including the `dictate_*` keys Mynah has not imported yet. Unset tri-state (`None`) is omitted from emitted TOML, never written. — src: whiz/config.py, tests/test_config.py
- **NS-6** — A "tests pass" claim names the exact command actually run; a bare claim is NOT evidence. — src: local toolchain state *(carried over as old NS-8, Swift clause dropped — the Swift suite moved to mynah)*

## Settled — decided, do not relitigate

- **NS-7** — The split (2026-09): whiz is a transcription CLI; dictation (engine, tuning contract, golden corpus, Swift app, whisper.cpp submodule) moved to mynah with its history. `whiz dictate` remains a nonzero-exit pointer for a release or two; `dictate_*` keys stay readable until Mynah imports them. Do not re-propose re-merging the products. — src: commit a92c8d3, README.md
- **NS-8** — The dictation segmentation stars (old NS-1..NS-6, NS-9..NS-14: tuning contract, golden corpus, cross-implementation divergences, poisoned-calibration fix) moved with the product to mynah. whiz does not segment audio at session speed; its pipeline is batch (whisper-cli VAD + sherpa-onnx diarization). If those stars are wanted in mynah — especially against its C++ rework — they should be ported THERE, not kept here as dead letters. — src: mynah/tuning/tuning.toml (byte-equal values at split)

## Graveyard — tried and rejected / validated

- **NS-9** — VALIDATED (carried): voice-profile provenance contract — it caught PR #5's Swift port re-introducing centroid drift through the shared profile store (PR closed 2026-09-22, obsolete post-split). Keep the pattern.
- **NS-10** — REJECTED: runtime tuning-file dependency (old NS-12) — moot for whiz (no tuning file), recorded so the pattern stays rejected if a cross-implementation contract ever returns.

## Open — explicitly unresolved (do NOT assume either way)

- Whether the C++ mynah rework should port the segmentation north-stars (NS-1..NS-6 old numbering) into that repo — outside whiz's scope; flagged for the mynah track, not decided here.
