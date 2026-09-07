---
name: config
description: "Skill for the Config area of whiz. 4 symbols across 1 files."
---

# Config

4 symbols | 1 files | Cohesion: 89%

## When to Use

- Working with code in `macos/`
- Understanding how parseValue, unquote, parseArray work
- Modifying config-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `macos/Sources/WhizKit/Config/FlatTOML.swift` | parseValue, unquote, parseArray, splitTopLevel |

## Entry Points

Start here when exploring this area:

- **`parseValue`** (Function) — `macos/Sources/WhizKit/Config/FlatTOML.swift:108`
- **`unquote`** (Function) — `macos/Sources/WhizKit/Config/FlatTOML.swift:120`
- **`parseArray`** (Function) — `macos/Sources/WhizKit/Config/FlatTOML.swift:143`
- **`splitTopLevel`** (Function) — `macos/Sources/WhizKit/Config/FlatTOML.swift:170`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `parseValue` | Function | `macos/Sources/WhizKit/Config/FlatTOML.swift` | 108 |
| `unquote` | Function | `macos/Sources/WhizKit/Config/FlatTOML.swift` | 120 |
| `parseArray` | Function | `macos/Sources/WhizKit/Config/FlatTOML.swift` | 143 |
| `splitTopLevel` | Function | `macos/Sources/WhizKit/Config/FlatTOML.swift` | 170 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Save → SplitTopLevel` | cross_community | 5 |
| `Save → Unquote` | cross_community | 5 |
| `Load → SplitTopLevel` | cross_community | 5 |
| `Load → Unquote` | cross_community | 5 |
| `Load → SplitTopLevel` | cross_community | 5 |
| `Load → Unquote` | cross_community | 5 |

## How to Explore

1. `context({name: "parseValue"})` — see callers and callees
2. `query({search_query: "config"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
