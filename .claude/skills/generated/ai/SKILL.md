---
name: ai
description: "Skill for the AI area of whiz. 15 symbols across 3 files."
---

# AI

15 symbols | 3 files | Cohesion: 73%

## When to Use

- Working with code in `macos/`
- Understanding how send, framesForEntries, frameManifest work
- Modifying ai-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `macos/Sources/WhizKit/AI/AnalysisEngine.swift` | complete, send, framesForEntries, frameManifest, mapReduceText (+7) |
| `macos/Tests/WhizKitTests/AIHTTPClientTests.swift` | listModels, mockSession |
| `macos/Tests/WhizKitTests/AIAnalysisTests.swift` | autoDetectRoutes |

## Entry Points

Start here when exploring this area:

- **`send`** (Function) — `macos/Sources/WhizKit/AI/AnalysisEngine.swift:95`
- **`framesForEntries`** (Function) — `macos/Sources/WhizKit/AI/AnalysisEngine.swift:196`
- **`frameManifest`** (Function) — `macos/Sources/WhizKit/AI/AnalysisEngine.swift:205`
- **`mapReduceText`** (Function) — `macos/Sources/WhizKit/AI/AnalysisEngine.swift:287`
- **`mapReduceVision`** (Function) — `macos/Sources/WhizKit/AI/AnalysisEngine.swift:321`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `send` | Function | `macos/Sources/WhizKit/AI/AnalysisEngine.swift` | 95 |
| `framesForEntries` | Function | `macos/Sources/WhizKit/AI/AnalysisEngine.swift` | 196 |
| `frameManifest` | Function | `macos/Sources/WhizKit/AI/AnalysisEngine.swift` | 205 |
| `mapReduceText` | Function | `macos/Sources/WhizKit/AI/AnalysisEngine.swift` | 287 |
| `mapReduceVision` | Function | `macos/Sources/WhizKit/AI/AnalysisEngine.swift` | 321 |
| `synthesize` | Function | `macos/Sources/WhizKit/AI/AnalysisEngine.swift` | 364 |
| `runningContext` | Function | `macos/Sources/WhizKit/AI/AnalysisEngine.swift` | 384 |
| `resolvePromptAuto` | Function | `macos/Sources/WhizKit/AI/AnalysisEngine.swift` | 396 |
| `probeModel` | Function | `macos/Sources/WhizKit/AI/AnalysisEngine.swift` | 494 |
| `autoDetectRoutes` | Function | `macos/Tests/WhizKitTests/AIAnalysisTests.swift` | 156 |
| `listModels` | Function | `macos/Sources/WhizKit/AI/AnalysisEngine.swift` | 460 |
| `getModelNames` | Function | `macos/Sources/WhizKit/AI/AnalysisEngine.swift` | 472 |
| `listModels` | Function | `macos/Tests/WhizKitTests/AIHTTPClientTests.swift` | 217 |
| `mockSession` | Function | `macos/Tests/WhizKitTests/AIHTTPClientTests.swift` | 247 |
| `complete` | Method | `macos/Sources/WhizKit/AI/AnalysisEngine.swift` | 21 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `MapReduceText → Subsample` | cross_community | 4 |
| `MapReduceText → Send` | intra_community | 4 |
| `Analyze → Subsample` | cross_community | 3 |
| `Analyze → Send` | cross_community | 3 |
| `MapReduceVision → Subsample` | cross_community | 3 |
| `MapReduceVision → Send` | intra_community | 3 |
| `MapReduceVision → Clock` | cross_community | 3 |
| `ProbeModel → Subsample` | cross_community | 3 |
| `ProbeModel → Send` | intra_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| WhizKitTests | 3 calls |
| Merge | 2 calls |

## How to Explore

1. `context({name: "send"})` — see callers and callees
2. `query({search_query: "ai"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
