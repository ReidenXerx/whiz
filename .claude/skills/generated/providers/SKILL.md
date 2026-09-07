---
name: providers
description: "Skill for the Providers area of whiz. 32 symbols across 8 files."
---

# Providers

32 symbols | 8 files | Cohesion: 92%

## When to Use

- Working with code in `whiz/`
- Understanding how test_select_stt_provider_override, resolve_settings, run_dictate work
- Modifying providers-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `whiz/dictate/providers/macos_indicator.py` | drawRect_, drawRect_, setup, _ensure_panel, _create_panel (+6) |
| `whiz/dictate/providers/__init__.py` | _platform_default, select_stt_provider, select_injector, select_indicator, _register_macos (+1) |
| `whiz/dictate/providers/macos_inject.py` | type_text, _keystroke, _paste, _char_to_keycode |
| `tests/test_dictate.py` | test_select_stt_provider_override, test_menubar_setup_noop_if_already_setup, test_indicator_panel_disables_hides_on_deactivate |
| `whiz/dictate/providers/macos_rumps.py` | _ensure_icon_pngs, setup, _build_menu |
| `whiz/dictate/engine.py` | resolve_settings, run_dictate |
| `whiz/dictate/providers/mlx.py` | load, transcribe |
| `whiz/dictate/providers/macos_logo.py` | draw_whiz_logo |

## Entry Points

Start here when exploring this area:

- **`test_select_stt_provider_override`** (Function) — `tests/test_dictate.py:273`
- **`resolve_settings`** (Function) — `whiz/dictate/engine.py:232`
- **`run_dictate`** (Function) — `whiz/dictate/engine.py:1036`
- **`select_stt_provider`** (Function) — `whiz/dictate/providers/__init__.py:68`
- **`select_injector`** (Function) — `whiz/dictate/providers/__init__.py:83`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `test_select_stt_provider_override` | Function | `tests/test_dictate.py` | 273 |
| `resolve_settings` | Function | `whiz/dictate/engine.py` | 232 |
| `run_dictate` | Function | `whiz/dictate/engine.py` | 1036 |
| `select_stt_provider` | Function | `whiz/dictate/providers/__init__.py` | 68 |
| `select_injector` | Function | `whiz/dictate/providers/__init__.py` | 83 |
| `select_indicator` | Function | `whiz/dictate/providers/__init__.py` | 96 |
| `test_menubar_setup_noop_if_already_setup` | Function | `tests/test_dictate.py` | 2353 |
| `draw_whiz_logo` | Function | `whiz/dictate/providers/macos_logo.py` | 35 |
| `test_indicator_panel_disables_hides_on_deactivate` | Function | `tests/test_dictate.py` | 1874 |
| `drawRect_` | Method | `whiz/dictate/providers/macos_indicator.py` | 272 |
| `drawRect_` | Method | `whiz/dictate/providers/macos_indicator.py` | 381 |
| `setup` | Method | `whiz/dictate/providers/macos_rumps.py` | 130 |
| `setup` | Method | `whiz/dictate/providers/macos_indicator.py` | 64 |
| `whizFadeIn_` | Method | `whiz/dictate/providers/macos_indicator.py` | 406 |
| `whizFadeOut_` | Method | `whiz/dictate/providers/macos_indicator.py` | 409 |
| `type_text` | Method | `whiz/dictate/providers/macos_inject.py` | 41 |
| `load` | Method | `whiz/dictate/providers/mlx.py` | 60 |
| `transcribe` | Method | `whiz/dictate/providers/mlx.py` | 85 |
| `_platform_default` | Function | `whiz/dictate/providers/__init__.py` | 59 |
| `_ensure_icon_pngs` | Function | `whiz/dictate/providers/macos_rumps.py` | 58 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Run_dictate → _set_state` | cross_community | 5 |
| `Run_dictate → Add_state_listener` | cross_community | 5 |
| `Run_dictate → _is_macos` | cross_community | 5 |
| `Run_dictate → _start_ptt_listener` | cross_community | 5 |
| `Run_dictate → _start_toggle_listener` | cross_community | 5 |
| `Run → _build_menu` | cross_community | 5 |
| `Setup → _create_objc_view_class` | intra_community | 5 |
| `Run_dictate → Check_permissions` | cross_community | 4 |
| `_setup_menu_bar → Draw_whiz_logo` | cross_community | 4 |
| `Run_dictate → _platform_default` | intra_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Tests | 2 calls |
| Dictate | 1 calls |

## How to Explore

1. `context({name: "test_select_stt_provider_override"})` — see callers and callees
2. `query({search_query: "providers"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
