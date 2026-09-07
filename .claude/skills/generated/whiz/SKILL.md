---
name: whiz
description: "Skill for the Whiz area of whiz. 122 symbols across 11 files."
---

# Whiz

122 symbols | 11 files | Cohesion: 86%

## When to Use

- Working with code in `whiz/`
- Understanding how header, rule, phase work
- Modifying whiz-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `whiz/cli.py` | _analysis_output_path, _recommend_model, _looks_vision_capable, _resolve_vision, _pick_model_interactive (+31) |
| `whiz/ai.py` | resolve_prompt_auto, transcript_text, _fmt_clock, chat_text, _frame_manifest (+17) |
| `whiz/merge.py` | _fmt_clock, format_dialogue_txt, flush, _speaker_color, speaker_palette (+9) |
| `whiz/ui.py` | _is_tty, header, rule, phase, status (+7) |
| `whiz/profiles.py` | compute_speaker_embeddings, _average_vectors, profiles_dir, _profile_path, _load_profile_raw (+7) |
| `whiz/diarize.py` | _import_sherpa, find_embedding_model, run_diarization, _read_wav_pcm, _default_diarization_dir (+3) |
| `whiz/models.py` | _alias_from_name, _short_alias, discover, resolve, pick_best (+3) |
| `whiz/config.py` | to_dict, _emit_toml, load, save |
| `whiz/screenshots.py` | _frame_name, extract_segment_frames, _extract_one |
| `whiz/dictate/providers/macos_rumps.py` | run, _open |

## Entry Points

Start here when exploring this area:

- **`header`** (Function) — `whiz/ui.py:46`
- **`rule`** (Function) — `whiz/ui.py:76`
- **`phase`** (Function) — `whiz/ui.py:83`
- **`status`** (Function) — `whiz/ui.py:95`
- **`wrote`** (Function) — `whiz/ui.py:127`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `header` | Function | `whiz/ui.py` | 46 |
| `rule` | Function | `whiz/ui.py` | 76 |
| `phase` | Function | `whiz/ui.py` | 83 |
| `status` | Function | `whiz/ui.py` | 95 |
| `wrote` | Function | `whiz/ui.py` | 127 |
| `summary` | Function | `whiz/ui.py` | 166 |
| `spinner` | Function | `whiz/ui.py` | 201 |
| `write` | Function | `whiz/ui.py` | 233 |
| `format_dialogue_txt` | Function | `whiz/merge.py` | 158 |
| `flush` | Function | `whiz/merge.py` | 168 |
| `speaker_palette` | Function | `whiz/merge.py` | 202 |
| `format_speakers_html` | Function | `whiz/merge.py` | 221 |
| `speaker_label_line` | Function | `whiz/ui.py` | 146 |
| `tally` | Function | `whiz/ui.py` | 152 |
| `resolve_prompt_auto` | Function | `whiz/ai.py` | 438 |
| `transcript_text` | Function | `whiz/ai.py` | 479 |
| `chat_text` | Function | `whiz/ai.py` | 589 |
| `chunk_entries` | Function | `whiz/ai.py` | 630 |
| `analyze` | Function | `whiz/ai.py` | 664 |
| `load` | Function | `whiz/config.py` | 175 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Save_profile → Profiles_dir` | intra_community | 4 |
| `Auto_assign_names → Profiles_dir` | cross_community | 4 |
| `Cmd_transcribe → _video_auto_flags` | cross_community | 3 |
| `Cmd_transcribe → _diarization_available` | cross_community | 3 |
| `Cmd_transcribe → _auto_threads` | cross_community | 3 |
| `Cmd_transcribe → _find_whisper_cli` | cross_community | 3 |
| `Run_diarization → Diar_cache_path` | cross_community | 3 |
| `Run_diarization → _default_diarization_dir` | cross_community | 3 |
| `Cmd_merge → _apply_speaker_names_list` | cross_community | 3 |
| `Cmd_merge → _prompt_speaker_names` | cross_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Tests | 3 calls |

## How to Explore

1. `context({name: "header"})` — see callers and callees
2. `query({search_query: "whiz"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
