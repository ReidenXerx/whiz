---
name: scripts
description: "Skill for the Scripts area of whiz. 52 symbols across 6 files."
---

# Scripts

52 symbols | 6 files | Cohesion: 87%

## When to Use

- Working with code in `scripts/`
- Understanding how verifyInstall, parseChangedSymbols, answered work
- Modifying scripts-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `scripts/bearing-verify.mjs` | readStealth, readRuntime, checkFile, checkManifest, checkPackageGates (+16) |
| `scripts/bearing-ci.mjs` | git, gn, repoName, collectDiff, detectChanges (+9) |
| `scripts/bearing-token-benchmark.mjs` | tok, gn, cypher, pickTargets, graphCost (+2) |
| `scripts/bearing-agent.mjs` | loadStaleness, run, runAllowFail, markRefreshOutcome, git (+2) |
| `scripts/lib/project-tmp.mjs` | withProjectTmpEnv, isEnospcError |
| `scripts/bearing-test-order.mjs` | parseChangedSymbols |

## Entry Points

Start here when exploring this area:

- **`verifyInstall`** (Function) — `scripts/bearing-verify.mjs:395`
- **`parseChangedSymbols`** (Function) — `scripts/bearing-test-order.mjs:81`
- **`answered`** (Function) — `scripts/bearing-token-benchmark.mjs:163`
- **`withProjectTmpEnv`** (Function) — `scripts/lib/project-tmp.mjs:25`
- **`isEnospcError`** (Function) — `scripts/lib/project-tmp.mjs:95`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `verifyInstall` | Function | `scripts/bearing-verify.mjs` | 395 |
| `parseChangedSymbols` | Function | `scripts/bearing-test-order.mjs` | 81 |
| `answered` | Function | `scripts/bearing-token-benchmark.mjs` | 163 |
| `withProjectTmpEnv` | Function | `scripts/lib/project-tmp.mjs` | 25 |
| `isEnospcError` | Function | `scripts/lib/project-tmp.mjs` | 95 |
| `readStealth` | Function | `scripts/bearing-verify.mjs` | 43 |
| `readRuntime` | Function | `scripts/bearing-verify.mjs` | 56 |
| `checkFile` | Function | `scripts/bearing-verify.mjs` | 108 |
| `checkManifest` | Function | `scripts/bearing-verify.mjs` | 113 |
| `checkPackageGates` | Function | `scripts/bearing-verify.mjs` | 123 |
| `checkRuntimeCoversAgent` | Function | `scripts/bearing-verify.mjs` | 180 |
| `checkModuleDelivery` | Function | `scripts/bearing-verify.mjs` | 199 |
| `checkRetiredHookKeys` | Function | `scripts/bearing-verify.mjs` | 250 |
| `checkSkillsStore` | Function | `scripts/bearing-verify.mjs` | 272 |
| `checkHooksJson` | Function | `scripts/bearing-verify.mjs` | 308 |
| `checkZed` | Function | `scripts/bearing-verify.mjs` | 330 |
| `checkHookExecutable` | Function | `scripts/bearing-verify.mjs` | 383 |
| `git` | Function | `scripts/bearing-ci.mjs` | 49 |
| `gn` | Function | `scripts/bearing-ci.mjs` | 57 |
| `repoName` | Function | `scripts/bearing-ci.mjs` | 73 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Run → GetProjectTmpDir` | cross_community | 5 |
| `Run → DfMount` | cross_community | 5 |
| `Run → ParseUsePct` | cross_community | 5 |
| `Main → ReadStealth` | cross_community | 4 |
| `Main → RuntimeSet` | intra_community | 4 |
| `Main → Git` | intra_community | 3 |
| `Main → Gn` | intra_community | 3 |
| `Main → Num` | intra_community | 3 |
| `Main → ParseChangedSymbols` | intra_community | 3 |
| `Run → IsEnospcError` | intra_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Cluster_88 | 2 calls |

## How to Explore

1. `context({name: "verifyInstall"})` — see callers and callees
2. `query({search_query: "scripts"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
