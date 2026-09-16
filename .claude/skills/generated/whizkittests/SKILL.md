---
name: whizkittests
description: "Skill for the WhizKitTests area of whiz. 186 symbols across 25 files."
---

# WhizKitTests

186 symbols | 25 files | Cohesion: 83%

## When to Use

- Working with code in `macos/`
- Understanding how speakersInOrder, speakersByTalkTime, representativeQuotes work
- Modifying whizkittests-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `macos/Tests/WhizKitTests/AIAnalysisTests.swift` | MockChatClient, chunkEntriesSplits, chunkTextSplits, essentialsMarkers, analyzeShortSingleCall (+16) |
| `macos/Tests/WhizKitTests/SpeakerProfilesTests.swift` | saveCreatesNew, saveMergesWithExisting, saveDimMismatchReplaces, saveAccumulatesAcrossWrites, loadRoundTripsMerged (+14) |
| `macos/Tests/WhizKitTests/ConfigTests.swift` | parsesScalars, parsesEscapes, parsesArrays, parsesCommaInString, stripsTrailingComments (+12) |
| `macos/Tests/WhizKitTests/TranscriptMergeTests.swift` | seg, assignSpeakersMaxOverlap, assignSpeakersNoDiar, speakersByTalkTimeOrdersMostFirst, speakersInOrderOfAppearance (+9) |
| `macos/Tests/WhizKitTests/FrameOCRTests.swift` | htmlScreenBlock, normalizeCollapses, normalizeMinChars, normalizeMaxCharsBoundary, normalizeMaxCharsSingleLine (+7) |
| `macos/Tests/WhizKitTests/AIHTTPClientTests.swift` | reset, visionContent, error400NotRetried, noChoices, client (+5) |
| `macos/Tests/WhizKitTests/TuningTests.swift` | flatTOMLParsesMultiLineArrays, flatTOMLUnterminatedArrayIsContain, flatTOMLSingleLineArrayStillWorks, tuning, number (+4) |
| `macos/Sources/WhizKit/STT/SpeakerProfiles.swift` | profilePath, loadProfiles, saveProfile, forgetProfile, jsonString (+4) |
| `macos/Tests/WhizKitTests/AudioFileDecoderTests.swift` | writeWav, loudStereoDoesNotExceedFullScale, tempURL, decodesStereo44kHzToMono16kHz, decodesNativeRateWithoutConversion (+4) |
| `macos/Tests/WhizKitTests/FrameExtractorTests.swift` | capturesFramesAtSegmentStarts, writeColoredVideo, centerPixels, ProgressRecorder, record (+4) |

## Entry Points

Start here when exploring this area:

- **`speakersInOrder`** (Function) — `macos/Sources/WhizKit/Merge/LabeledTranscript.swift:42`
- **`speakersByTalkTime`** (Function) — `macos/Sources/WhizKit/Merge/LabeledTranscript.swift:54`
- **`representativeQuotes`** (Function) — `macos/Sources/WhizKit/Merge/LabeledTranscript.swift:74`
- **`relabel`** (Function) — `macos/Sources/WhizKit/Merge/LabeledTranscript.swift:107`
- **`assignSpeakers`** (Function) — `macos/Sources/WhizKit/Merge/LabeledTranscript.swift:120`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `MockChatClient` | Class | `macos/Tests/WhizKitTests/AIAnalysisTests.swift` | 15 |
| `ProgressRecorder` | Class | `macos/Tests/WhizKitTests/AIAnalysisTests.swift` | 422 |
| `ProgressRecorder` | Class | `macos/Tests/WhizKitTests/DiarizationTests.swift` | 146 |
| `EventCollector` | Class | `macos/Tests/WhizKitTests/TranscriptionFlowTests.swift` | 245 |
| `ProgressRecorder` | Class | `macos/Tests/WhizKitTests/FrameExtractorTests.swift` | 258 |
| `speakersInOrder` | Function | `macos/Sources/WhizKit/Merge/LabeledTranscript.swift` | 42 |
| `speakersByTalkTime` | Function | `macos/Sources/WhizKit/Merge/LabeledTranscript.swift` | 54 |
| `representativeQuotes` | Function | `macos/Sources/WhizKit/Merge/LabeledTranscript.swift` | 74 |
| `relabel` | Function | `macos/Sources/WhizKit/Merge/LabeledTranscript.swift` | 107 |
| `assignSpeakers` | Function | `macos/Sources/WhizKit/Merge/LabeledTranscript.swift` | 120 |
| `speakerColor` | Function | `macos/Sources/WhizKit/Merge/SpeakersHTML.swift` | 20 |
| `htmlEscape` | Function | `macos/Sources/WhizKit/Merge/SpeakersHTML.swift` | 26 |
| `format` | Function | `macos/Sources/WhizKit/Merge/SpeakersHTML.swift` | 41 |
| `htmlScreenBlock` | Function | `macos/Tests/WhizKitTests/FrameOCRTests.swift` | 141 |
| `seg` | Function | `macos/Tests/WhizKitTests/TranscriptMergeTests.swift` | 17 |
| `assignSpeakersMaxOverlap` | Function | `macos/Tests/WhizKitTests/TranscriptMergeTests.swift` | 33 |
| `assignSpeakersNoDiar` | Function | `macos/Tests/WhizKitTests/TranscriptMergeTests.swift` | 49 |
| `speakersByTalkTimeOrdersMostFirst` | Function | `macos/Tests/WhizKitTests/TranscriptMergeTests.swift` | 58 |
| `speakersInOrderOfAppearance` | Function | `macos/Tests/WhizKitTests/TranscriptMergeTests.swift` | 69 |
| `relabelReplacesNames` | Function | `macos/Tests/WhizKitTests/TranscriptMergeTests.swift` | 82 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `ShowSettings → ForCharacter` | cross_community | 7 |
| `ShowSettings → Combo` | cross_community | 6 |
| `Save → SplitTopLevel` | cross_community | 5 |
| `Save → Unquote` | cross_community | 5 |
| `Load → SplitTopLevel` | cross_community | 5 |
| `Load → Unquote` | cross_community | 5 |
| `Load → SplitTopLevel` | cross_community | 5 |
| `Load → Unquote` | cross_community | 5 |
| `StartSession → FirstUnquotedHash` | cross_community | 5 |
| `StartSession → WhizConfig` | cross_community | 5 |

## Connected Areas

| Area | Connections |
|------|-------------|
| AI | 9 calls |
| STT | 3 calls |
| Merge | 2 calls |
| Session | 2 calls |
| Config | 1 calls |

## How to Explore

1. `context({name: "speakersInOrder"})` — see callers and callees
2. `query({search_query: "whizkittests"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
