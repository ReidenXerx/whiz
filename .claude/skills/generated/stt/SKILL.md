---
name: stt
description: "Skill for the STT area of whiz. 46 symbols across 15 files."
---

# STT

46 symbols | 15 files | Cohesion: 74%

## When to Use

- Working with code in `macos/`
- Understanding how looksVisionCapable, resolveVision, reportMarkdown work
- Modifying stt-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `macos/Sources/WhizKit/STT/WhisperBatchTranscriber.swift` | WhisperBatchTranscriber, unload, transcribe, cancel, BatchBox (+8) |
| `macos/Tests/WhizKitTests/WhisperBatchTranscriberTests.swift` | unloadedThrows, boxCancellationAndProgress, ValueCollector, profileParamsMirrorWhisperCLI, timestampsConvertCentiseconds |
| `macos/Sources/WhizKit/STT/ModelDownloader.swift` | downloadModel, downloadVAD, start, DownloadDelegate |
| `macos/Sources/WhizKit/AI/AnalysisEngine.swift` | looksVisionCapable, resolveVision, reportMarkdown |
| `macos/Sources/WhizKit/STT/NativeTranscriptionBackend.swift` | transcribe, log, resolveVAD |
| `macos/Sources/WhizKit/STT/WhisperModel.swift` | resolve, resolveBatch, resolveVAD |
| `macos/Sources/WhizKit/STT/SpeakerProfiles.swift` | computeSpeakerEmbeddings, runExtraction, average |
| `macos/Sources/WhizKit/STT/Diarization.swift` | run, runOnQueue |
| `macos/Tests/WhizKitTests/AIAnalysisTests.swift` | visionHeuristic, reportMarkdown |
| `macos/Sources/WhizKit/STT/SileroVAD.swift` | SileroVAD, load |

## Entry Points

Start here when exploring this area:

- **`looksVisionCapable`** (Function) — `macos/Sources/WhizKit/AI/AnalysisEngine.swift:439`
- **`resolveVision`** (Function) — `macos/Sources/WhizKit/AI/AnalysisEngine.swift:448`
- **`reportMarkdown`** (Function) — `macos/Sources/WhizKit/AI/AnalysisEngine.swift:507`
- **`run`** (Function) — `macos/Sources/WhizKit/STT/Diarization.swift:96`
- **`runOnQueue`** (Function) — `macos/Sources/WhizKit/STT/Diarization.swift:138`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `WhisperBatchTranscriber` | Class | `macos/Sources/WhizKit/STT/WhisperBatchTranscriber.swift` | 47 |
| `BatchBox` | Class | `macos/Sources/WhizKit/STT/WhisperBatchTranscriber.swift` | 338 |
| `ValueCollector` | Class | `macos/Tests/WhizKitTests/WhisperBatchTranscriberTests.swift` | 111 |
| `SileroVAD` | Class | `macos/Sources/WhizKit/STT/SileroVAD.swift` | 24 |
| `WhisperEngine` | Class | `macos/Sources/WhizKit/STT/WhisperEngine.swift` | 19 |
| `ContextHandle` | Class | `macos/Sources/WhizKit/STT/WhisperBatchTranscriber.swift` | 328 |
| `DownloadDelegate` | Class | `macos/Sources/WhizKit/STT/ModelDownloader.swift` | 132 |
| `looksVisionCapable` | Function | `macos/Sources/WhizKit/AI/AnalysisEngine.swift` | 439 |
| `resolveVision` | Function | `macos/Sources/WhizKit/AI/AnalysisEngine.swift` | 448 |
| `reportMarkdown` | Function | `macos/Sources/WhizKit/AI/AnalysisEngine.swift` | 507 |
| `run` | Function | `macos/Sources/WhizKit/STT/Diarization.swift` | 96 |
| `runOnQueue` | Function | `macos/Sources/WhizKit/STT/Diarization.swift` | 138 |
| `transcribe` | Function | `macos/Sources/WhizKit/STT/NativeTranscriptionBackend.swift` | 134 |
| `log` | Function | `macos/Sources/WhizKit/STT/NativeTranscriptionBackend.swift` | 139 |
| `unload` | Function | `macos/Sources/WhizKit/STT/WhisperBatchTranscriber.swift` | 96 |
| `transcribe` | Function | `macos/Sources/WhizKit/STT/WhisperBatchTranscriber.swift` | 121 |
| `cancel` | Function | `macos/Sources/WhizKit/STT/WhisperBatchTranscriber.swift` | 153 |
| `reportProgress` | Function | `macos/Sources/WhizKit/STT/WhisperBatchTranscriber.swift` | 367 |
| `visionHeuristic` | Function | `macos/Tests/WhizKitTests/AIAnalysisTests.swift` | 374 |
| `reportMarkdown` | Function | `macos/Tests/WhizKitTests/AIAnalysisTests.swift` | 409 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Transcribe → Append` | cross_community | 4 |
| `Transcribe → ExtractedAudio` | cross_community | 4 |
| `EnsureModelLoaded → RegisterOnce` | cross_community | 3 |
| `Run → ReportProgress` | intra_community | 3 |
| `Run → DiarSegment` | intra_community | 3 |
| `Transcribe → ReportSegment` | cross_community | 3 |
| `Transcribe → ProfileParams` | cross_community | 3 |
| `Transcribe → Segment` | cross_community | 3 |
| `Transcribe → Launch` | cross_community | 3 |
| `ComputeSpeakerEmbeddings → Samples` | intra_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| WhizKitTests | 17 calls |
| Merge | 5 calls |
| AI | 2 calls |
| Media | 1 calls |

## How to Explore

1. `context({name: "looksVisionCapable"})` — see callers and callees
2. `query({search_query: "stt"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
