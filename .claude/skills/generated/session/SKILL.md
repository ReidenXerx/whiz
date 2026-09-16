---
name: session
description: "Skill for the Session area of whiz. 36 symbols across 9 files."
---

# Session

36 symbols | 9 files | Cohesion: 78%

## When to Use

- Working with code in `macos/`
- Understanding how rms, process, makeUtterance work
- Modifying session-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `macos/Sources/WhizKit/Session/SessionController.swift` | endSession, ingest, enqueue, finishTranscription, scheduleIdleUnload (+10) |
| `macos/Sources/WhizKit/Session/AudioCapture.swift` | stop, AudioCapture, start, convert, InputBox (+1) |
| `macos/Sources/WhizKit/STT/UtteranceDetector.swift` | process, makeUtterance, flush, reset, applyCalibration |
| `macos/Tests/WhizKitTests/TuningTests.swift` | loadWavSamples, loadExpected, goldenSegmentation, poisonedCalibrationPinsTheDefect |
| `macos/Sources/WhizKit/WhizApplication.swift` | registerHotkey, handleTrigger |
| `macos/Sources/WhizKit/STT/TranscriptFilter.swift` | rms |
| `macos/Sources/WhizKit/Config/WhizConfig.swift` | load |
| `macos/Sources/WhizKit/System/Permissions.swift` | requestMicrophone |
| `macos/Sources/WhizKit/UI/SettingsView.swift` | binding |

## Entry Points

Start here when exploring this area:

- **`rms`** (Function) — `macos/Sources/WhizKit/STT/TranscriptFilter.swift:70`
- **`process`** (Function) — `macos/Sources/WhizKit/STT/UtteranceDetector.swift:58`
- **`makeUtterance`** (Function) — `macos/Sources/WhizKit/STT/UtteranceDetector.swift:112`
- **`flush`** (Function) — `macos/Sources/WhizKit/STT/UtteranceDetector.swift:123`
- **`reset`** (Function) — `macos/Sources/WhizKit/STT/UtteranceDetector.swift:130`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `AudioCapture` | Class | `macos/Sources/WhizKit/Session/AudioCapture.swift` | 11 |
| `SessionController` | Class | `macos/Sources/WhizKit/Session/SessionController.swift` | 13 |
| `InputBox` | Class | `macos/Sources/WhizKit/Session/AudioCapture.swift` | 87 |
| `rms` | Function | `macos/Sources/WhizKit/STT/TranscriptFilter.swift` | 70 |
| `process` | Function | `macos/Sources/WhizKit/STT/UtteranceDetector.swift` | 58 |
| `makeUtterance` | Function | `macos/Sources/WhizKit/STT/UtteranceDetector.swift` | 112 |
| `flush` | Function | `macos/Sources/WhizKit/STT/UtteranceDetector.swift` | 123 |
| `reset` | Function | `macos/Sources/WhizKit/STT/UtteranceDetector.swift` | 130 |
| `applyCalibration` | Function | `macos/Sources/WhizKit/STT/UtteranceDetector.swift` | 139 |
| `stop` | Function | `macos/Sources/WhizKit/Session/AudioCapture.swift` | 46 |
| `endSession` | Function | `macos/Sources/WhizKit/Session/SessionController.swift` | 156 |
| `ingest` | Function | `macos/Sources/WhizKit/Session/SessionController.swift` | 182 |
| `enqueue` | Function | `macos/Sources/WhizKit/Session/SessionController.swift` | 234 |
| `finishTranscription` | Function | `macos/Sources/WhizKit/Session/SessionController.swift` | 289 |
| `scheduleIdleUnload` | Function | `macos/Sources/WhizKit/Session/SessionController.swift` | 343 |
| `unloadModel` | Function | `macos/Sources/WhizKit/Session/SessionController.swift` | 384 |
| `loadWavSamples` | Function | `macos/Tests/WhizKitTests/TuningTests.swift` | 170 |
| `loadExpected` | Function | `macos/Tests/WhizKitTests/TuningTests.swift` | 208 |
| `goldenSegmentation` | Function | `macos/Tests/WhizKitTests/TuningTests.swift` | 232 |
| `poisonedCalibrationPinsTheDefect` | Function | `macos/Tests/WhizKitTests/TuningTests.swift` | 301 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `ApplicationWillTerminate → ForCharacter` | cross_community | 7 |
| `ApplicationWillTerminate → Paste` | cross_community | 7 |
| `RegisterHotkey → Utterance` | cross_community | 7 |
| `RegisterHotkey → IsHallucination` | cross_community | 7 |
| `Ingest → ForCharacter` | cross_community | 6 |
| `Ingest → Paste` | cross_community | 6 |
| `RegisterHotkey → Reset` | cross_community | 6 |
| `RegisterHotkey → Rms` | cross_community | 6 |
| `RegisterHotkey → FinishTranscription` | cross_community | 6 |
| `RegisterHotkey → UnloadModel` | cross_community | 6 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Input | 3 calls |
| WhizKitTests | 2 calls |
| STT | 1 calls |
| UI | 1 calls |

## How to Explore

1. `context({name: "rms"})` — see callers and callees
2. `query({search_query: "session"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
