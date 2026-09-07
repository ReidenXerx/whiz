---
name: cluster-88
description: "Skill for the Cluster_88 area of whiz. 5 symbols across 1 files."
---

# Cluster_88

5 symbols | 1 files | Cohesion: 80%

## When to Use

- Working with code in `scripts/`
- Understanding how getProjectTmpDir, dfMount, parseUsePct work
- Modifying cluster_88-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `scripts/lib/project-tmp.mjs` | getProjectTmpDir, dfMount, parseUsePct, tmpSpaceReport, enospcHelp |

## Entry Points

Start here when exploring this area:

- **`getProjectTmpDir`** (Function) — `scripts/lib/project-tmp.mjs:15`
- **`dfMount`** (Function) — `scripts/lib/project-tmp.mjs:34`
- **`parseUsePct`** (Function) — `scripts/lib/project-tmp.mjs:57`
- **`tmpSpaceReport`** (Function) — `scripts/lib/project-tmp.mjs:65`
- **`enospcHelp`** (Function) — `scripts/lib/project-tmp.mjs:104`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `getProjectTmpDir` | Function | `scripts/lib/project-tmp.mjs` | 15 |
| `dfMount` | Function | `scripts/lib/project-tmp.mjs` | 34 |
| `parseUsePct` | Function | `scripts/lib/project-tmp.mjs` | 57 |
| `tmpSpaceReport` | Function | `scripts/lib/project-tmp.mjs` | 65 |
| `enospcHelp` | Function | `scripts/lib/project-tmp.mjs` | 104 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Run → GetProjectTmpDir` | cross_community | 5 |
| `Run → DfMount` | cross_community | 5 |
| `Run → ParseUsePct` | cross_community | 5 |
| `LoadStaleness → GetProjectTmpDir` | cross_community | 3 |
| `MarkRefreshOutcome → GetProjectTmpDir` | cross_community | 3 |

## How to Explore

1. `context({name: "getProjectTmpDir"})` — see callers and callees
2. `query({search_query: "cluster_88"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
