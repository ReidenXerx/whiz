---
name: golden
description: "Skill for the Golden area of whiz. 10 symbols across 1 files."
---

# Golden

10 symbols | 1 files | Cohesion: 91%

## When to Use

- Working with code in `tuning/`
- Understanding how main, reference_speech_regions, close_region work
- Modifying golden-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `tuning/golden/generate.py` | _sinusoid_segment, _silence_segment, _click, _write_wav, main (+5) |

## Entry Points

Start here when exploring this area:

- **`main`** (Function) — `tuning/golden/generate.py:290`
- **`reference_speech_regions`** (Function) — `tuning/golden/generate.py:166`
- **`close_region`** (Function) — `tuning/golden/generate.py:230`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `main` | Function | `tuning/golden/generate.py` | 290 |
| `reference_speech_regions` | Function | `tuning/golden/generate.py` | 166 |
| `close_region` | Function | `tuning/golden/generate.py` | 230 |
| `_sinusoid_segment` | Function | `tuning/golden/generate.py` | 118 |
| `_silence_segment` | Function | `tuning/golden/generate.py` | 125 |
| `_click` | Function | `tuning/golden/generate.py` | 129 |
| `_write_wav` | Function | `tuning/golden/generate.py` | 136 |
| `_rms` | Function | `tuning/golden/generate.py` | 146 |
| `_frames` | Function | `tuning/golden/generate.py` | 153 |
| `_energy_gate_verdict` | Function | `tuning/golden/generate.py` | 212 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Reference_speech_regions → _rms` | intra_community | 4 |

## How to Explore

1. `context({name: "main"})` — see callers and callees
2. `query({search_query: "golden"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
