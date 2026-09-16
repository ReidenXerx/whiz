---
name: input
description: "Skill for the Input area of whiz. 10 symbols across 5 files."
---

# Input

10 symbols | 5 files | Cohesion: 82%

## When to Use

- Working with code in `macos/`
- Understanding how type, keystroke, paste work
- Modifying input-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `macos/Sources/WhizKit/Input/TextInjector.swift` | type, keystroke, paste, forCharacter |
| `macos/Sources/WhizKit/Session/SessionController.swift` | deliver, shutdownBlocking |
| `macos/Sources/WhizKit/Input/HotkeyManager.swift` | register, unregister |
| `macos/Sources/WhizKit/STT/TranscriptFilter.swift` | isHallucination |
| `macos/Sources/WhizKit/WhizApplication.swift` | applicationWillTerminate |

## Entry Points

Start here when exploring this area:

- **`type`** (Function) — `macos/Sources/WhizKit/Input/TextInjector.swift:14`
- **`keystroke`** (Function) — `macos/Sources/WhizKit/Input/TextInjector.swift:39`
- **`paste`** (Function) — `macos/Sources/WhizKit/Input/TextInjector.swift:65`
- **`forCharacter`** (Function) — `macos/Sources/WhizKit/Input/TextInjector.swift:125`
- **`isHallucination`** (Function) — `macos/Sources/WhizKit/STT/TranscriptFilter.swift:78`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `type` | Function | `macos/Sources/WhizKit/Input/TextInjector.swift` | 14 |
| `keystroke` | Function | `macos/Sources/WhizKit/Input/TextInjector.swift` | 39 |
| `paste` | Function | `macos/Sources/WhizKit/Input/TextInjector.swift` | 65 |
| `forCharacter` | Function | `macos/Sources/WhizKit/Input/TextInjector.swift` | 125 |
| `isHallucination` | Function | `macos/Sources/WhizKit/STT/TranscriptFilter.swift` | 78 |
| `deliver` | Function | `macos/Sources/WhizKit/Session/SessionController.swift` | 293 |
| `register` | Function | `macos/Sources/WhizKit/Input/HotkeyManager.swift` | 24 |
| `unregister` | Function | `macos/Sources/WhizKit/Input/HotkeyManager.swift` | 53 |
| `shutdownBlocking` | Function | `macos/Sources/WhizKit/Session/SessionController.swift` | 366 |
| `applicationWillTerminate` | Function | `macos/Sources/WhizKit/WhizApplication.swift` | 120 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `ApplicationWillTerminate → ForCharacter` | cross_community | 7 |
| `ApplicationWillTerminate → Paste` | cross_community | 7 |
| `ShowSettings → ForCharacter` | cross_community | 7 |
| `RegisterHotkey → IsHallucination` | cross_community | 7 |
| `Ingest → ForCharacter` | cross_community | 6 |
| `Ingest → Paste` | cross_community | 6 |
| `ApplicationWillTerminate → Utterance` | cross_community | 5 |
| `ApplicationWillTerminate → IsHallucination` | cross_community | 5 |
| `RegisterHotkey → ForCharacter` | cross_community | 5 |
| `ApplicationWillTerminate → Reset` | cross_community | 4 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Session | 1 calls |
| WhizKitTests | 1 calls |

## How to Explore

1. `context({name: "type"})` — see callers and callees
2. `query({search_query: "input"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
