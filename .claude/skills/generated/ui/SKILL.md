---
name: ui
description: "Skill for the UI area of whiz. 22 symbols across 7 files."
---

# UI

22 symbols | 7 files | Cohesion: 93%

## When to Use

- Working with code in `macos/`
- Understanding how start, pickFile, build work
- Modifying ui-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `macos/Sources/WhizKit/UI/TranscriptionWindow.swift` | start, pickFile, build, show, directory (+3) |
| `macos/Sources/WhizKit/WhizApplication.swift` | startTranscription, applicationDidFinishLaunching, updateIndicator, AppDelegate, showSettings |
| `macos/Sources/WhizKit/UI/IndicatorPanel.swift` | IndicatorPanel, setup, show |
| `macos/Sources/WhizKit/UI/SettingsWindow.swift` | SettingsWindow, show, build |
| `macos/Tests/WhizKitTests/TranscriptionFlowTests.swift` | outputDirectoryNaming |
| `macos/Sources/WhizKit/Session/SessionController.swift` | refreshPermissions |
| `macos/Sources/WhizKit/Input/HotkeyManager.swift` | HotkeyManager |

## Entry Points

Start here when exploring this area:

- **`start`** (Function) — `macos/Sources/WhizKit/UI/TranscriptionWindow.swift:21`
- **`pickFile`** (Function) — `macos/Sources/WhizKit/UI/TranscriptionWindow.swift:42`
- **`build`** (Function) — `macos/Sources/WhizKit/UI/TranscriptionWindow.swift:73`
- **`show`** (Function) — `macos/Sources/WhizKit/UI/TranscriptionWindow.swift:87`
- **`directory`** (Function) — `macos/Sources/WhizKit/UI/TranscriptionWindow.swift:111`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `TranscriptionViewModel` | Class | `macos/Sources/WhizKit/UI/TranscriptionWindow.swift` | 149 |
| `IndicatorPanel` | Class | `macos/Sources/WhizKit/UI/IndicatorPanel.swift` | 19 |
| `HotkeyManager` | Class | `macos/Sources/WhizKit/Input/HotkeyManager.swift` | 12 |
| `SettingsWindow` | Class | `macos/Sources/WhizKit/UI/SettingsWindow.swift` | 19 |
| `TranscriptionWindow` | Class | `macos/Sources/WhizKit/UI/TranscriptionWindow.swift` | 14 |
| `AppDelegate` | Class | `macos/Sources/WhizKit/WhizApplication.swift` | 69 |
| `start` | Function | `macos/Sources/WhizKit/UI/TranscriptionWindow.swift` | 21 |
| `pickFile` | Function | `macos/Sources/WhizKit/UI/TranscriptionWindow.swift` | 42 |
| `build` | Function | `macos/Sources/WhizKit/UI/TranscriptionWindow.swift` | 73 |
| `show` | Function | `macos/Sources/WhizKit/UI/TranscriptionWindow.swift` | 87 |
| `directory` | Function | `macos/Sources/WhizKit/UI/TranscriptionWindow.swift` | 111 |
| `handle` | Function | `macos/Sources/WhizKit/UI/TranscriptionWindow.swift` | 203 |
| `startTranscription` | Function | `macos/Sources/WhizKit/WhizApplication.swift` | 132 |
| `outputDirectoryNaming` | Function | `macos/Tests/WhizKitTests/TranscriptionFlowTests.swift` | 81 |
| `refreshPermissions` | Function | `macos/Sources/WhizKit/Session/SessionController.swift` | 222 |
| `setup` | Function | `macos/Sources/WhizKit/UI/IndicatorPanel.swift` | 33 |
| `show` | Function | `macos/Sources/WhizKit/UI/IndicatorPanel.swift` | 65 |
| `applicationDidFinishLaunching` | Function | `macos/Sources/WhizKit/WhizApplication.swift` | 83 |
| `updateIndicator` | Function | `macos/Sources/WhizKit/WhizApplication.swift` | 151 |
| `show` | Function | `macos/Sources/WhizKit/UI/SettingsWindow.swift` | 29 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `ShowSettings → ForCharacter` | cross_community | 7 |
| `ShowSettings → Combo` | cross_community | 6 |
| `ShowSettings → Language` | cross_community | 6 |
| `ShowSettings → Commit` | cross_community | 6 |
| `ApplicationDidFinishLaunching → WhizLogo` | intra_community | 5 |
| `ShowSettings → UpdateConfig` | cross_community | 5 |
| `ApplicationDidFinishLaunching → BarHeight` | cross_community | 4 |
| `ApplicationDidFinishLaunching → VisualEffectBackground` | intra_community | 4 |
| `StartTranscription → Directory` | intra_community | 3 |
| `StartTranscription → Build` | intra_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Session | 3 calls |

## How to Explore

1. `context({name: "start"})` — see callers and callees
2. `query({search_query: "ui"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
