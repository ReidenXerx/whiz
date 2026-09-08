---
name: tests
description: "Skill for the Tests area of whiz. 158 symbols across 17 files."
---

# Tests

158 symbols | 17 files | Cohesion: 86%

## When to Use

- Working with code in `tests/`
- Understanding how test_idle_unload_unloads_model_when_session_inactive, test_idle_unload_skips_when_session_active, test_schedule_idle_unload_zero_is_noop work
- Modifying tests-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `tests/test_dictate.py` | _make_engine, test_idle_unload_unloads_model_when_session_inactive, test_idle_unload_skips_when_session_active, test_schedule_idle_unload_zero_is_noop, test_start_session_loads_model_and_shows_indicator (+72) |
| `tests/test_merge.py` | _seg, test_assign_speakers_max_overlap, test_assign_speakers_no_diar, test_speakers_by_talk_time_orders_most_first, test_speakers_in_order_of_appearance (+9) |
| `tests/test_segmentation_golden.py` | _load_expected, _load_wav, _engine_settings, _make_replaying_engine, _run_capture (+8) |
| `tests/test_models.py` | _make_isolated_config, _touch, test_resolve_turbo_short_alias, test_pick_best_prefers_turbo_q5, test_pick_best_falls_back_to_anything (+8) |
| `tests/test_cli.py` | _resolve, test_resolve_vision_no_vision_always_disables, test_resolve_vision_explicit_with_frames_enables, test_resolve_vision_explicit_without_frames_warns, test_resolve_vision_explicit_but_text_model_overrides (+6) |
| `tests/test_diarize_cache.py` | test_diar_cache_path_uses_string_append, test_diar_cache_round_trip, test_diar_cache_miss_on_param_mismatch, test_diar_cache_missing_file_returns_none, test_diar_cache_threshold_epsilon_tolerance |
| `whiz/dictate/providers/macos_rumps.py` | do_toggle, do_quit, _on_toggle, _on_quit, on_state |
| `whiz/dictate/providers/base.py` | STTProvider, TextInjector, DictationIndicator, NullIndicator |
| `whiz/diarize.py` | diar_cache_path, load_diarization_cache, _write_diarization_cache |
| `whiz/dictate/providers/macos_indicator.py` | MacIndicator, show, hide |

## Entry Points

Start here when exploring this area:

- **`test_idle_unload_unloads_model_when_session_inactive`** (Function) — `tests/test_dictate.py:380`
- **`test_idle_unload_skips_when_session_active`** (Function) — `tests/test_dictate.py:393`
- **`test_schedule_idle_unload_zero_is_noop`** (Function) — `tests/test_dictate.py:403`
- **`test_start_session_loads_model_and_shows_indicator`** (Function) — `tests/test_dictate.py:418`
- **`test_start_session_skips_load_if_already_loaded`** (Function) — `tests/test_dictate.py:430`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `MacIndicator` | Class | `whiz/dictate/providers/macos_indicator.py` | 54 |
| `FakeSTT` | Class | `tests/test_dictate.py` | 149 |
| `STTProvider` | Class | `whiz/dictate/providers/base.py` | 24 |
| `MlxWhisperProvider` | Class | `whiz/dictate/providers/mlx.py` | 49 |
| `FakeInjector` | Class | `tests/test_dictate.py` | 182 |
| `TextInjector` | Class | `whiz/dictate/providers/base.py` | 51 |
| `MacTextInjector` | Class | `whiz/dictate/providers/macos_inject.py` | 38 |
| `FakeIndicator` | Class | `tests/test_dictate.py` | 196 |
| `DictationIndicator` | Class | `whiz/dictate/providers/base.py` | 73 |
| `NullIndicator` | Class | `whiz/dictate/providers/base.py` | 113 |
| `test_idle_unload_unloads_model_when_session_inactive` | Function | `tests/test_dictate.py` | 380 |
| `test_idle_unload_skips_when_session_active` | Function | `tests/test_dictate.py` | 393 |
| `test_schedule_idle_unload_zero_is_noop` | Function | `tests/test_dictate.py` | 403 |
| `test_start_session_loads_model_and_shows_indicator` | Function | `tests/test_dictate.py` | 418 |
| `test_start_session_skips_load_if_already_loaded` | Function | `tests/test_dictate.py` | 430 |
| `test_end_session_noop_when_not_active` | Function | `tests/test_dictate.py` | 456 |
| `test_toggle_session_starts_then_ends` | Function | `tests/test_dictate.py` | 463 |
| `test_transcribe_and_inject_skips_silence_energy_gate` | Function | `tests/test_dictate.py` | 542 |
| `test_indicator_states_on_session` | Function | `tests/test_dictate.py` | 584 |
| `test_indicator_update_level_records_values` | Function | `tests/test_dictate.py` | 595 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Run_diarization → Diar_cache_path` | cross_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Providers | 1 calls |

## How to Explore

1. `context({name: "test_idle_unload_unloads_model_when_session_inactive"})` — see callers and callees
2. `query({search_query: "tests"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
