---
name: dictate
description: "Skill for the Dictate area of whiz. 51 symbols across 4 files."
---

# Dictate

51 symbols | 4 files | Cohesion: 89%

## When to Use

- Working with code in `whiz/`
- Understanding how on_activate, callback, plist_path work
- Modifying dictate-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `whiz/dictate/engine.py` | _wait_for_accessibility, add_state_listener, run, stop, _setup_menu_bar (+26) |
| `whiz/dictate/service.py` | plist_path, build_plist, _run, install, uninstall (+7) |
| `whiz/dictate/setup.py` | _dictate_extra_installed, _inject_extra, _check_extra, run_checks, _mark (+2) |
| `whiz/dictate/providers/base.py` | check_permissions |

## Entry Points

Start here when exploring this area:

- **`on_activate`** (Function) — `whiz/dictate/engine.py:956`
- **`callback`** (Function) — `whiz/dictate/engine.py:590`
- **`plist_path`** (Function) — `whiz/dictate/service.py:43`
- **`build_plist`** (Function) — `whiz/dictate/service.py:191`
- **`install`** (Function) — `whiz/dictate/service.py:249`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `on_activate` | Function | `whiz/dictate/engine.py` | 956 |
| `callback` | Function | `whiz/dictate/engine.py` | 590 |
| `plist_path` | Function | `whiz/dictate/service.py` | 43 |
| `build_plist` | Function | `whiz/dictate/service.py` | 191 |
| `install` | Function | `whiz/dictate/service.py` | 249 |
| `uninstall` | Function | `whiz/dictate/service.py` | 290 |
| `status` | Function | `whiz/dictate/service.py` | 304 |
| `run_checks` | Function | `whiz/dictate/setup.py` | 221 |
| `setup` | Function | `whiz/dictate/setup.py` | 230 |
| `on_press` | Function | `whiz/dictate/engine.py` | 1011 |
| `on_release` | Function | `whiz/dictate/engine.py` | 1018 |
| `add_state_listener` | Method | `whiz/dictate/engine.py` | 321 |
| `run` | Method | `whiz/dictate/engine.py` | 347 |
| `stop` | Method | `whiz/dictate/engine.py` | 382 |
| `check_permissions` | Method | `whiz/dictate/providers/base.py` | 59 |
| `toggle_session` | Method | `whiz/dictate/engine.py` | 390 |
| `ptt_press` | Method | `whiz/dictate/engine.py` | 401 |
| `ptt_release` | Method | `whiz/dictate/engine.py` | 408 |
| `_wait_for_accessibility` | Function | `whiz/dictate/engine.py` | 185 |
| `_is_macos` | Function | `whiz/dictate/engine.py` | 1032 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Install → _venv_bin_dir` | cross_community | 6 |
| `Run_dictate → _set_state` | cross_community | 5 |
| `Run_dictate → Add_state_listener` | cross_community | 5 |
| `Run_dictate → _is_macos` | cross_community | 5 |
| `Run_dictate → _start_ptt_listener` | cross_community | 5 |
| `Run_dictate → _start_toggle_listener` | cross_community | 5 |
| `Run → _start_ptt_listener` | intra_community | 5 |
| `Run → _start_toggle_listener` | intra_community | 5 |
| `Run → _build_menu` | cross_community | 5 |
| `_run_with_appkit → _set_state` | cross_community | 5 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Providers | 1 calls |

## How to Explore

1. `context({name: "on_activate"})` — see callers and callees
2. `query({search_query: "dictate"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
