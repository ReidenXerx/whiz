---
name: merge
description: "Skill for the Merge area of whiz. 18 symbols across 5 files."
---

# Merge

18 symbols | 5 files | Cohesion: 70%

## When to Use

- Working with code in `macos/`
- Understanding how transcriptText, formatDialogueTXT, flush work
- Modifying merge-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `macos/Sources/WhizKit/Merge/TranscriptFormatter.swift` | clock, segmentLogLine, srtTimestamp, srt, json (+2) |
| `macos/Tests/WhizKitTests/TranscriptFormatterTests.swift` | clockTimestamps, segmentLogLines, srtTimestamps, srtOutput, emptyJSONIsStillValid |
| `macos/Sources/WhizKit/Merge/LabeledTranscript.swift` | formatDialogueTXT, flush, formatLabeledSRT |
| `macos/Tests/WhizKitTests/TranscriptMergeTests.swift` | formatDialogueTXTMergesConsecutive, formatLabeledSRTStructure |
| `macos/Sources/WhizKit/AI/AnalysisEngine.swift` | transcriptText |

## Entry Points

Start here when exploring this area:

- **`transcriptText`** (Function) — `macos/Sources/WhizKit/AI/AnalysisEngine.swift:141`
- **`formatDialogueTXT`** (Function) — `macos/Sources/WhizKit/Merge/LabeledTranscript.swift:171`
- **`flush`** (Function) — `macos/Sources/WhizKit/Merge/LabeledTranscript.swift:177`
- **`clock`** (Function) — `macos/Sources/WhizKit/Merge/TranscriptFormatter.swift:34`
- **`segmentLogLine`** (Function) — `macos/Sources/WhizKit/Merge/TranscriptFormatter.swift:91`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `transcriptText` | Function | `macos/Sources/WhizKit/AI/AnalysisEngine.swift` | 141 |
| `formatDialogueTXT` | Function | `macos/Sources/WhizKit/Merge/LabeledTranscript.swift` | 171 |
| `flush` | Function | `macos/Sources/WhizKit/Merge/LabeledTranscript.swift` | 177 |
| `clock` | Function | `macos/Sources/WhizKit/Merge/TranscriptFormatter.swift` | 34 |
| `segmentLogLine` | Function | `macos/Sources/WhizKit/Merge/TranscriptFormatter.swift` | 91 |
| `clockTimestamps` | Function | `macos/Tests/WhizKitTests/TranscriptFormatterTests.swift` | 23 |
| `segmentLogLines` | Function | `macos/Tests/WhizKitTests/TranscriptFormatterTests.swift` | 88 |
| `formatDialogueTXTMergesConsecutive` | Function | `macos/Tests/WhizKitTests/TranscriptMergeTests.swift` | 121 |
| `formatLabeledSRT` | Function | `macos/Sources/WhizKit/Merge/LabeledTranscript.swift` | 153 |
| `srtTimestamp` | Function | `macos/Sources/WhizKit/Merge/TranscriptFormatter.swift` | 23 |
| `srt` | Function | `macos/Sources/WhizKit/Merge/TranscriptFormatter.swift` | 45 |
| `srtTimestamps` | Function | `macos/Tests/WhizKitTests/TranscriptFormatterTests.swift` | 14 |
| `srtOutput` | Function | `macos/Tests/WhizKitTests/TranscriptFormatterTests.swift` | 31 |
| `formatLabeledSRTStructure` | Function | `macos/Tests/WhizKitTests/TranscriptMergeTests.swift` | 110 |
| `json` | Function | `macos/Sources/WhizKit/Merge/TranscriptFormatter.swift` | 57 |
| `ms` | Function | `macos/Sources/WhizKit/Merge/TranscriptFormatter.swift` | 78 |
| `escapeJSON` | Function | `macos/Sources/WhizKit/Merge/TranscriptFormatter.swift` | 84 |
| `emptyJSONIsStillValid` | Function | `macos/Tests/WhizKitTests/TranscriptFormatterTests.swift` | 80 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `MapReduceVision → Clock` | cross_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| WhizKitTests | 2 calls |

## How to Explore

1. `context({name: "transcriptText"})` — see callers and callees
2. `query({search_query: "merge"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
