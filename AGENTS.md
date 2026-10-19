# Hermes Agent - Development Guide

For AI coding assistants and developers working on hermes-agent. This root file is a hub: what
applies everywhere, then a routing table. Each area's `AGENTS.md` loads automatically when you work
in that directory; read it before editing there. `python scripts/check` caps this file at 12k chars
and every root-to-area chain at 30k, so it loads whole on 128k+ models: long form goes in the guide.

**Never give up on the right solution.**

## What Hermes Is

Hermes is a personal AI agent that runs the same agent core across a CLI, a messaging
gateway (Telegram, Discord, Slack, ~20 platforms), a TUI, and an Electron desktop app. It
learns across sessions (memory + skills), delegates to subagents, runs scheduled jobs, and
drives a real terminal and browser. It is extended primarily through **plugins and skills**,
not by growing the core.

Two invariants shape almost every design decision and are the lens for reviewing any change:

- **Per-conversation prompt caching is sacred.** Mutating past context, swapping toolsets,
  reloading memories or rebuilding the system prompt mid-conversation breaks the cached prefix and
  multiplies the user's cost; the ONE exception is context compression. Slash commands that change
  system-prompt state defer to the next session, with an opt-in `--now` (`/skills install --now`).
- **The core is a narrow waist; capability lives at the edges.** Every model tool is sent on
  every API call, so the bar for a new *core* tool is high. New capability should arrive as a
  CLI command + skill, a service-gated tool, or a plugin — not as core surface.

## Contribution rubric

The project's intent layer, for contributors and for the triage sweeper (which may only close on
`implemented_on_main`, `cannot_reproduce` or `incoherent`; taste-based closes are a maintainer's
call, and when in doubt a PR stays open). Long form with examples:
`website/docs/developer-guide/contributing.md` § Contribution rubric.

**Wanted:** real bug fixes (repro on `main`, the exact line, the whole class incl. sibling paths);
reach at the edges (adapters, providers, models, UI features) wired into the existing setup UX;
god-file → module refactors; extending before duplicating (3+ PRs in one category → an ABC +
orchestrator); behaviour-contract tests; E2E with real imports against a temp `HERMES_HOME` for
resolution, config, security and I/O changes; salvage by cherry-pick so authorship survives.

**Rejected even when well-built:** hooks with no concrete consumer; new `HERMES_*` env vars for
non-secret config (`.env` is secrets only, behaviour goes in `config.yaml`); a new core tool when
terminal + file or a skill already does the job; `offset`/`limit` pagination on instructional tools;
"fixes" that destroy the feature they secure; outbound telemetry without an opt-in gate;
change-detector tests; plugins that touch core files; third-party products in the core tree (ship a
standalone plugin repo).

**Before you call it a bug,** verify the claim AND the intent (`git log -p -S "<symbol>"`): the
isolation is often the design (profiles are islands on purpose), an absence can be load-bearing,
and a fix that cannot point to the line where the bug manifests has an unverified premise.

**Security:** `SECURITY.md` is the scope authority. A §3.1 finding goes private (GitHub Security
Advisories or security@nousresearch.com), never into a public issue, PR, commit or comment; §3.2
hardening is ordinary public work. Name the §2 boundary crossed, with a repro on `main`.

**Footprint ladder** (take the highest rung that solves it): extend existing code → CLI command +
skill → service-gated tool (`check_fn` answers reachability/opt-in, never per-session surface:
`tools/AGENTS.md`) → plugin → MCP server in the catalog → new core tool (fundamental, broadly useful,
unreachable otherwise).

## Development Environment

`source ./activate` (fish: `activate.fish`, PowerShell: `activate.ps1`) provisions and activates the
PM environment; pick an isolated `HERMES_HOME`/`HERMES_RUNTIME_DIR` first
(`website/docs/reference/package-management.md#developer-workflow`). Tests need the separate test
environment in `CONTRIBUTING.md`. **`python scripts/check`** runs every blocking lint check CI
runs, with CI's pinned tools; `--install-hook pre-push` runs it on every push (re-run after
pulling to refresh the hook). `# noqa` does not waive a ratchet finding:
`# health: allow <RULE> -- <why>` does.

## Project Structure

Counts shift constantly; the filesystem is canonical. Load-bearing entry points:

```
hermes-agent/
├── run_agent.py          # AIAgent facade; the turn loop lives in agent/turn_*.py
├── model_tools.py        # Tool orchestration, discover_builtin_tools(), handle_function_call()
├── toolsets.py           # TOOLSETS dict, _HERMES_CORE_TOOLS
├── cli.py                # HermesCLI (REPL, slash dispatch) + hermes_cli/cli_*_mixin.py
├── hermes_state.py       # SessionDB facade; hermes_state_*.py siblings
├── hermes_constants.py   # get_hermes_home(), display_hermes_home() — profile-aware paths
├── agent/                # turn loop phases, providers, memory, compression, prompt builder
├── hermes_cli/           # CLI subcommands, setup, config, plugins loader, updater, web_routers/
├── tools/                # Tool implementations (tools/registry.py) + environments/ backends
├── gateway/              # run.py facade + run_*.py phases + session*.py + platforms/
├── plugins/              # memory/, context_engine/, model-providers/, kanban/, image_gen/, ...
├── skills/               # Built-in skills (by category)   optional-skills/: shipped, not active
├── ui-tui/, tui_gateway/ # Ink terminal UI + its Python JSON-RPC backend (also serves Desktop)
├── apps/desktop/         # Electron desktop app (+ apps/shared)   web/: dashboard SPA
├── cron/                 # jobs.py + scheduler.py (+ scheduler_*.py)
├── pm/, hermes_platform/ # dependency/environment manager; machine facts + executable lookup
├── scripts/              # check, run_tests.sh, code_health/, ci/
├── website/              # Docusaurus docs (developer-guide/ holds the long-form area docs)
└── tests/                # Pytest suite, mirrors the source tree
```

**User state:** `~/.hermes/config.yaml` (settings), `.env` (secrets only), `logs/` (`hermes logs`);
all profile-aware via `get_hermes_home()`.

### Facade + siblings layout

Every former god file is a **facade** (public entry points + the names other packages import) plus
**siblings** `<stem>_<topic>.py`, each owning one topic (`hermes_state.py`, `gateway/run.py`,
`tools/mcp_tool.py`, `hermes_cli/kanban.py`, `hermes_cli/web_server.py`, `cli.py` →
`hermes_cli/cli_*_mixin.py`, `run_agent.py` → `agent/turn_*.py`).

- **Find code by topic:** `grep -rn "def name" <dir>/<stem>_*.py`, not by reading the facade.
- **Siblings late-import the facade** inside functions; never a module-level cycle.
- **Patch where production reads:** a sibling doing `from <facade> import name` inside the function
  makes the facade the seam; a patch on the defining module passes silently.
- **Size and complexity are ratcheted per unit** (`scripts/code_health/config.py`): new functions
  CC ≤ 20, ≤ 300 lines, nesting ≤ 6; files ≤ 2,000 lines; units already over may only go down (move
  a function into a sibling to offset growth, or put new tests in a new test file). Behaviour goes
  in a sibling, never a facade; name ladders become a dict → handler.
- **No re-export shims for internal moves;** internal paths are not API. Moving a symbol means
  fixing its docs in the same PR (grep `website/docs`, `skills/`, every `AGENTS.md`).

## Rules that apply everywhere

- **No defensive clutter:** no wrappers or `try/except: pass` around code that cannot fail, no flags
  nobody sets, no dead code wired in without E2E proof. Comments keep the WHY, cut the WHAT.
- **Never infer process identity from argv substrings;** use
  `gateway.status.looks_like_gateway_command_line` / `hermes_cli.update_cmd._hermes_holder_subcommand`
  (HX003; details `hermes_cli/AGENTS.md`).
- **Never hardcode `~/.hermes`:** `get_hermes_home()` for paths, `display_hermes_home()` for text
  (HX001). `_get_profiles_root()` is HOME-anchored on purpose.
- **One process serves many profiles.** Code that runs outside a turn (boot probes, eviction,
  tickers, deferred callbacks, RPC methods, thread hops, child spawns) binds the owning profile scope
  explicitly; `os.environ`, module globals and import-time values hold the launch profile's, so an
  unbound read is a silent default-profile leak. Binding points and seams: `gateway/AGENTS.md`
  § Profile scope (HX002/HX004/HX005/HX012, PS-P05/P06).
- **Machine facts and executable lookup go through `hermes_platform`** (`hermes_platform/AGENTS.md`).
- **Dependencies carry upper bounds** (`>=floor,<next_major`; git URLs and Actions pinned to a SHA);
  after editing `pyproject.toml` run `hermes pm lock` and commit `uv.lock`; never mutate a Hermes
  environment with raw pip/uv. Full policy: `pm/AGENTS.md`.
- **TypeScript (desktop, TUI, website):** feature-owned nanostores over threaded state, thin route
  roots, narrow hooks and colocated action modules, `interface` for props and shared shapes,
  table-driven dispatch over condition ladders; `src/app` routes, `src/store` atoms, `src/lib` pure
  helpers.
- **Commits and PRs:** rebase onto `main` before merging (a squash from a stale branch silently
  reverts newer fixes); 1–2 invariant tests per fix, proven red on the base.
- **Tests:** always `scripts/run_tests.sh`, never bare `pytest`; behaviour contracts, never
  change-detectors or source-reading tests; host-specific behaviour is tested on that host with
  `@pytest.mark.platforms(...)`, never by faking `sys.platform`; tests never write to `~/.hermes/`.
  Everything else: `tests/AGENTS.md`.

## Routing Table — working in X → read X/AGENTS.md

| Area | Read | Covers |
|---|---|---|
| `run_agent.py`, `agent/` | `agent/AGENTS.md` | turn phases, caching and message-flow invariants, compression, model/aux resolution |
| `cli.py`, `hermes_cli/` | `hermes_cli/AGENTS.md` | CLI mixins, slash registry, config system, skins, `hermes update`, profiles / multiplex |
| `gateway/` | `gateway/AGENTS.md` | adapters, message guards, streaming, notifications, token locks, § Profile scope |
| `gateway/platforms/` new adapter | `gateway/platforms/ADDING_A_PLATFORM.md` | step-by-step adapter guide |
| `tools/`, `toolsets.py`, `model_tools.py` | `tools/AGENTS.md` | adding tools, registry, toolsets, delegation, session-scoped surface tools |
| `plugins/`, `hermes_cli/plugins*.py` | `plugins/AGENTS.md` | plugin kinds, native compat contract, in-tree policy |
| `plugin-catalog/` | `plugin-catalog/README.md` | catalog admission rules (mirrored in the developer guide; keep identical) |
| `tui_gateway/`, `ui-tui/` | `tui_gateway/AGENTS.md` | process model, JSON-RPC transport, slash flow |
| `web/`, `hermes_cli/web_routers/` | `web/AGENTS.md` | dashboard embeds the real TUI |
| `apps/desktop/` | `apps/desktop/AGENTS.md`, `apps/desktop/src/AGENTS.md` | `serve` backend, slash palette, Bot Mode |
| `skills/`, `optional-skills/`, `agent/curator*.py` | `skills/AGENTS.md` | frontmatter, authoring standards, curator |
| `cron/`, kanban | `cron/AGENTS.md` | scheduler invariants, job fields, kanban dispatcher |
| `tests/` | `tests/AGENTS.md` | runner, placement, OS markers, `wine2e`, banned test shapes |
| `pm/`, `pyproject.toml` | `pm/AGENTS.md` | pinning policy, PM-owned environments, plugin quarantine |
| `hermes_platform/` | `hermes_platform/AGENTS.md` | host facts, resolvers |

Long-form background: `website/docs/developer-guide/`. Workflow rules (PR/issue/review/salvage
process) live in the `hermes-agent-dev` skill, not here.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **hermes-agent** (291625 symbols, 680583 relationships, 1175 execution flows).

> Index stale? Run `node .gitnexus/run.cjs analyze --index-only` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? Bootstrap with `npx`, `bunx`, or `pnpm dlx` — e.g. `bunx gitnexus@latest analyze` (npm 11 npx crash; #1939).
> On query/context/impact/cypher object results, read staleness.status and branch/lastCommit. Re-analyze only for behind or diverged — current is clone HEAD, not main.

## Always Do

- **MUST run impact before editing.** Use `impact({target: "symbolName", direction: "upstream"})` or `node .gitnexus/run.cjs impact "symbolName" --direction upstream --repo .`; report callers, processes, and risk. Never substitute grep for graph analysis.
- **MUST analyze graph changes before committing.** Use `detect_changes({scope: "all"})` (MCP) or `node .gitnexus/run.cjs detect-changes --scope all --repo .` (CLI fallback). `partial: true` or `truncated: true` is not a clean check — a zero means unseen, not unaffected; re-run it. For regression review: `detect_changes({scope: "compare", base_ref: "main"})` or `node .gitnexus/run.cjs detect-changes --scope compare --base-ref "main" --repo .`.
- MUST warn on HIGH/CRITICAL `risk` pre-edit; never use `riskSharedAxes` to waive a HIGH/CRITICAL `risk` warning. Compare File/symbol: MCP File omits axes; Graph-RAG expands File.
- **MUST treat `risk: UNKNOWN` as unresolved, not as low.** An empty caller set is not evidence the symbol is unused — it can also mean the callers are not resolvable by the index (plain-object property access, dynamic dispatch, cross-language calls). `impact` pairs `UNKNOWN` with a `riskNote` saying so. Confirm with a text search before treating the symbol as safe to change or delete; do not proceed on the strength of a zero.
- **MUST use `query({search_query: "concept"})` for concepts/flows, `context({name: "symbolName"})` for a named symbol, or `impact` for blast radius, on read-only callers, dependencies, imports, or execution flow.** Graph first; text search only for empty/`UNKNOWN`/literals.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method before MCP/CLI impact analysis.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis, and never read `UNKNOWN` as an all-clear — it means the walk could not answer, which is the one verdict that requires confirming by other means.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit before MCP/CLI graph change analysis.

## Resources

| Resource | Use for |
| --- | --- |
| `gitnexus://repo/hermes-agent/context` | Codebase overview, check index freshness |
| `gitnexus://repo/hermes-agent/clusters` | All functional areas |
| `gitnexus://repo/hermes-agent/processes` | All execution flows |
| `gitnexus://repo/hermes-agent/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
| --- | --- |
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
