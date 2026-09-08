---
name: media
description: "Skill for the Media area of whiz. 8 symbols across 2 files."
---

# Media

8 symbols | 2 files | Cohesion: 86%

## When to Use

- Working with code in `macos/`
- Understanding how recognize, frameDigest, frames work
- Modifying media-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `macos/Sources/WhizKit/Media/FrameOCR.swift` | recognize, frameDigest, frames, runBatch |
| `macos/Tests/WhizKitTests/FrameOCRTests.swift` | recognizesRenderedText, batchDedupeAndAlignment, writeTextFrame, tempURL |

## Entry Points

Start here when exploring this area:

- **`recognize`** (Function) — `macos/Sources/WhizKit/Media/FrameOCR.swift:31`
- **`frameDigest`** (Function) — `macos/Sources/WhizKit/Media/FrameOCR.swift:133`
- **`frames`** (Function) — `macos/Sources/WhizKit/Media/FrameOCR.swift:150`
- **`runBatch`** (Function) — `macos/Sources/WhizKit/Media/FrameOCR.swift:178`
- **`recognizesRenderedText`** (Function) — `macos/Tests/WhizKitTests/FrameOCRTests.swift:94`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `recognize` | Function | `macos/Sources/WhizKit/Media/FrameOCR.swift` | 31 |
| `frameDigest` | Function | `macos/Sources/WhizKit/Media/FrameOCR.swift` | 133 |
| `frames` | Function | `macos/Sources/WhizKit/Media/FrameOCR.swift` | 150 |
| `runBatch` | Function | `macos/Sources/WhizKit/Media/FrameOCR.swift` | 178 |
| `recognizesRenderedText` | Function | `macos/Tests/WhizKitTests/FrameOCRTests.swift` | 94 |
| `batchDedupeAndAlignment` | Function | `macos/Tests/WhizKitTests/FrameOCRTests.swift` | 113 |
| `writeTextFrame` | Function | `macos/Tests/WhizKitTests/FrameOCRTests.swift` | 161 |
| `tempURL` | Function | `macos/Tests/WhizKitTests/FrameOCRTests.swift` | 201 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Frames → OnProgress` | intra_community | 3 |
| `Frames → Outcome` | intra_community | 3 |
| `Frames → FrameDigest` | intra_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| WhizKitTests | 2 calls |

## How to Explore

1. `context({name: "recognize"})` — see callers and callees
2. `query({search_query: "media"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
