"""hermes-memory-store — holographic memory plugin using MemoryProvider interface.

Registers as a MemoryProvider plugin, giving the agent structured fact storage
with entity resolution, trust scoring, and HRR-based compositional retrieval.

Original plugin by dusterbloom (PR #2351), adapted to the MemoryProvider ABC.

Config in $HERMES_HOME/config.yaml (profile-scoped):
  plugins:
    hermes-memory-store:
      db_path: $HERMES_HOME/memory_store.db   # omit to use the default
      auto_extract: false
      default_trust: 0.5
      min_trust_threshold: 0.3
      temporal_decay_half_life: 0
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error
from utils import is_truthy_value
from .store import MemoryStore
from .retrieval import FactRetriever, no_confident_match
from hermes_cli.config import cfg_get

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas (unchanged from original PR)
# ---------------------------------------------------------------------------

FACT_STORE_SCHEMA = {
    "name": "fact_store",
    "description": (
        "Deep structured memory with algebraic reasoning. "
        "Use alongside the memory tool — memory for always-on context, "
        "fact_store for deep recall and compositional queries.\n\n"
        "ACTIONS (simple → powerful):\n"
        "• add — Store a fact the user would expect you to remember.\n"
        "• search — Ranked recall. Since 2026-09-04 it is keyword AND semantic: "
        "candidates come from a keyword index UNIONed with an embedding index, so "
        "a question that shares no words with the stored fact ('why did the "
        "overnight job keep redoing finished work') can still find it. Use it for "
        "prose questions, not only for literal terms like 'editor config'.\n"
        "• probe — Entity recall: ALL facts about a person/thing.\n"
        "• related — What connects to an entity? Structural adjacency.\n"
        "• reason — Compositional: facts connected to MULTIPLE entities simultaneously.\n"
        "• contradict — Memory hygiene: find facts making conflicting claims.\n"
        "• get — Exact fetch of ONE fact by fact_id (verify a write landed; "
        "search cannot prove absence).\n"
        "• update/remove/list — CRUD operations.\n\n"
        "IMPORTANT: Before answering questions about the user, ALWAYS probe or reason first.\n\n"
        "ABSTENTION: a search reply may carry \"no_confident_match\": true alongside "
        "its results. It means the best match scored below a calibrated relevance "
        "floor — the rows are still there and may still be useful, but the store is "
        "telling you it probably does not hold the answer. Treat it as a reason to "
        "say you do not know, or to look elsewhere, rather than as a reason to "
        "present the returned rows as the answer. Its absence is not a guarantee of "
        "correctness; it only fires when the cross-encoder ran."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "search", "probe", "related", "reason", "contradict", "get", "update", "remove", "list"],
            },
            "content": {"type": "string", "description": "Fact content (required for 'add')."},
            "query": {"type": "string", "description": "Search query (required for 'search')."},
            "entity": {"type": "string", "description": "Entity name for 'probe'/'related'."},
            "entities": {"type": "array", "items": {"type": "string"}, "description": "Entity names for 'reason'."},
            "fact_id": {"type": "integer", "description": "Fact ID for 'get'/'update'/'remove'."},
            # Free-form on purpose: 'add' stores whatever is passed, so an enum
            # here only misleads. The previous four-value enum went stale as
            # soon as callers invented categories, which is immediately —
            # leaving the schema declaring most stored rows invalid.
            # NAME THE ACTIONS. Every sibling here says which actions it serves
            # ("required for 'add'", "for 'get'/'update'/'remove'", "Trust
            # adjustment for 'update'") and until 2026-08-29 this one said none,
            # while "defaults to 'general'" made it read as insert-time only. A
            # cron run concluded from exactly that — correctly, on the text —
            # that 'update' takes no category, left three answered questions in
            # the open-question/hypothesis queue with the verdict written into
            # their bodies, and stored the misreading as a fact that reached the
            # morning briefing as a priority item. The comment below already
            # concedes the flat property bag cannot express this in JSON Schema,
            # "so that contract lived only in prose" — this was the one
            # parameter whose prose was never written.
            "category": {
                "type": "string",
                "description": (
                    "Free-form label. For 'add': the new fact's category, "
                    "defaults to 'general'. For 'update': CHANGES the category "
                    "— this is how an answered open-question or a confirmed "
                    "hypothesis leaves its queue and becomes 'researched'; a "
                    "verdict prefix in the body does NOT move it. Common: "
                    "paper, researched, activity, lesson, project, synthesis, "
                    "hypothesis, open-question, user_pref, tool."
                ),
            },
            "tags": {"type": "string", "description": "Comma-separated tags."},
            "trust_delta": {"type": "number", "description": "Trust adjustment for 'update'."},
            "min_trust": {"type": "number", "description": "Minimum trust filter (default: 0.3)."},
            "limit": {"type": "integer", "description": "Max results (default: 10)."},
        },
        "required": ["action"],
    },
}

# The schema above can only express ``required: ["action"]`` — nine actions
# share one flat property bag, so JSON Schema cannot state that 'related' needs
# 'entity' while 'update' needs 'fact_id'. That contract lived only in prose,
# and models routinely got it wrong. Three shapes recur, in frequency order:
# 'update' with content/trust_delta but no fact_id; 'search' passing 'entity';
# 'related' passing 'fact_id'. The old ``except KeyError`` reported just the
# missing key, which is not enough to self-correct from, so unattended runs
# repeated the same malformed call night after night.
#
# These tables drive a pre-dispatch check that names the action, the missing
# argument, and the action that would have worked.

_ACTION_REQUIRED_ARGS = {
    "add": ("content",),
    "search": ("query",),
    "probe": ("entity",),
    "related": ("entity",),
    "reason": ("entities",),
    "contradict": (),
    "get": ("fact_id",),
    "update": ("fact_id",),
    "remove": ("fact_id",),
    "list": (),
}

_ARG_MEANING = {
    "content": "the fact text to store",
    "query": "free-text keywords to match",
    "entity": "an entity NAME, e.g. \"Postgres\"",
    "entities": "a non-empty list of entity names",
    "fact_id": "the integer id of an existing fact",
}

# Corrections for the confusions actually observed in the cron logs, keyed by
# (action, argument the model supplied instead).
_ARG_CONFUSIONS = {
    ("search", "entity"): (
        "For entity lookups use action='probe' (all facts about an entity) or "
        "action='related' (structural adjacency). action='search' matches free "
        "text only."
    ),
    ("probe", "fact_id"): (
        "'probe' looks up entities, not facts. Facts are linked to entities "
        "automatically when added, so probe an entity name from the fact's "
        "content instead."
    ),
    ("related", "fact_id"): (
        "'related' looks up entities, not facts. Facts are linked to entities "
        "automatically when added, so probe an entity name from the fact's "
        "content instead."
    ),
}

# Fallback guidance when the action needs an id the model does not have. This
# is the single most common failure: there is no update-by-content path, so a
# model wanting to revise a fact has to look the id up first.
_ID_LOOKUP_HINT = (
    "Look it up first — action='search' or action='probe' returns a fact_id "
    "for every result. There is no update-by-content path."
)


def _validate_action_args(action, args):
    """Return an error string when *args* is missing what *action* requires.

    Treats empty values as missing: ``entities=[]`` is as unusable as no
    ``entities`` key at all.
    """
    required = _ACTION_REQUIRED_ARGS.get(action)
    if not required:
        return None

    missing = [
        name for name in required
        if args.get(name) is None or args.get(name) == "" or args.get(name) == []
    ]
    if not missing:
        return None

    name = missing[0]
    supplied = sorted(k for k in args if k != "action" and args.get(k) is not None)

    parts = [
        f"fact_store action='{action}' requires '{name}' "
        f"({_ARG_MEANING.get(name, 'see the tool schema')})."
    ]
    parts.append(
        f"You supplied: {', '.join(supplied)}." if supplied
        else "You supplied no other arguments."
    )
    for wrong in supplied:
        hint = _ARG_CONFUSIONS.get((action, wrong))
        if hint:
            parts.append(hint)
            break
    else:
        if name == "fact_id":
            parts.append(_ID_LOOKUP_HINT)
    return " ".join(parts)


def _content_wipe_refusal(existing: dict, args: dict):
    """Return an error string when an update's ``content`` would destroy a body.

    ``content`` is a full-replacement field, and twice a model has fed it
    something that was never a body: 2026-08-10 (three ANSWERED facts rewritten
    into one-liners taken from 60-char digest previews) and 2026-09-02 (a
    lifecycle tag word — experiment-design wiped fids 882/901 to
    "retired-experiment"/"designed" while correctly setting tags and trust in
    the same call, against a prompt that said not to touch content; the bodies
    only survived as fact_history snapshots). Prompt prose failed both times;
    like the ``remove_fact`` protected-category refusal, the invariant lives
    here instead.
    """
    new = (args.get("content") or "").strip()
    old = existing.get("content") or ""
    tag_pool = f"{existing.get('tags') or ''},{args.get('tags') or ''}"
    if new in {t.strip() for t in tag_pool.split(",") if t.strip()}:
        return (
            f"fact_store action='update' refused: content={new!r} is one of "
            f"this fact's tags, and content REPLACES the whole {len(old)}-char "
            f"body. To set lifecycle tags, re-issue the update with tags= (the "
            f"full comma-separated list) and NO content field — the body then "
            f"stays intact. Only pass content when rewriting the body in full."
        )
    if len(old) > 200 and len(new) < 40:
        return (
            f"fact_store action='update' refused: content of {len(new)} chars "
            f"would replace a {len(old)}-char body. Omit content to leave the "
            f"body unchanged (tags/trust_delta/category still apply), or "
            f"supply the complete replacement text."
        )
    return None


def _update_result(fid: int, updated: bool, before: dict, after: dict) -> dict:
    """Build the action='update' result so a no-op is visible to the caller.

    ``update_fact`` changes only the fields passed and returns True whenever
    the row exists, so ``{"updated": true}`` used to come back for a call that
    changed nothing. On 2026-09-03 consolidate-synthesize re-sent one such
    call eight times: it meant to strip stale queue tags from fid 1056 but
    never passed ``tags``, and nothing in the result said the tags were
    untouched. Now the result lists the fields that differ before/after,
    echoes the row's current category, tags and trust, and names an empty
    diff as the no-op it is.
    """
    changed = [
        k for k in ("content", "category", "tags")
        if (before.get(k) or "") != (after.get(k) or "")
    ]
    old_trust = float(before.get("trust_score") or 0.0)
    new_trust = float(after.get("trust_score") or 0.0)
    if abs(old_trust - new_trust) > 1e-9:
        changed.append("trust_score")
    result = {
        "updated": updated,
        "fact_id": fid,
        "changed": changed,
        "category": after.get("category"),
        "tags": after.get("tags"),
        "trust_score": after.get("trust_score"),
    }
    if updated and not changed:
        result["note"] = (
            "no-op: nothing differed from the stored row. An update changes only "
            "the fields you pass — to change tags, pass tags=<the complete "
            "replacement list>; to change category, pass category=. Do not "
            "re-send this call unchanged."
        )
    return result


FACT_FEEDBACK_SCHEMA = {
    "name": "fact_feedback",
    "description": (
        "Rate a fact after using it. Mark 'helpful' if accurate, 'unhelpful' if outdated. "
        "This trains the memory — good facts rise, bad facts sink."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["helpful", "unhelpful"]},
            "fact_id": {"type": "integer", "description": "The fact ID to rate."},
        },
        "required": ["action", "fact_id"],
    },
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_plugin_config() -> dict:
    try:
        # Canonical loader: behavioral read now honors the managed-scope
        # overlay + ${VAR} expansion (e.g. an api key template) too.
        from hermes_cli.config import load_config_readonly
        all_config = load_config_readonly()
        return cfg_get(all_config, "plugins", "hermes-memory-store", default={}) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

# The serving layer's XML->JSON tool-call converter sometimes fails to terminate
# the LAST `<parameter=...>` value at its `</parameter>`, and runs on through
# `</function></tool_call>` into whatever the model emitted next. Hermes receives
# well-formed JSON whose envelope is correct — right tool name, other arguments
# intact — with one string field carrying the tail. Observed 7 times between
# 2026-07-06 and 2026-08-30 across TWO different local models
# (genesis35-mtp-cron and qwen36-27b-cron), so it is the parser, not the model,
# and swapping models does not avoid it. Rare (~0.015% of calls) but it costs a
# whole call each time: on 2026-08-30 it ate BOTH of consolidate-synthesize's
# `action=reason` entity-pair calls, which is the one step that lane needs.
#
# The intended value is always the text before the first tag, so recovering it is
# exact. Anchored on the literal closing tags rather than a bare "<" because a
# fact body may legitimately contain angle brackets (`|lambda|>1`, `<=2 samples`);
# `</function>` and `</tool_call>` cannot occur in real prose here. Logged at
# WARNING with the field name so a silent repair is still greppable in agent.log.
_TOOL_TAG_TAIL = re.compile(r"</(?:parameter|function|tool_call)>|<tool_call>|<function=")


def _strip_tool_call_tail(args: dict) -> None:
    """Repair string arguments polluted by a mis-terminated tool-call parse."""
    for key, val in list(args.items()):
        if not isinstance(val, str):
            continue
        m = _TOOL_TAG_TAIL.search(val)
        if not m:
            continue
        cleaned = val[: m.start()].rstrip().rstrip(">").rstrip()
        logger.warning(
            "fact_store: repaired tool-call tail in argument %r (%d chars -> %d); "
            "the serving layer did not terminate the parameter value",
            key, len(val), len(cleaned),
        )
        args[key] = cleaned


class HolographicMemoryProvider(MemoryProvider):
    """Holographic memory with structured facts, entity resolution, and HRR retrieval."""

    def __init__(self, config: dict | None = None):
        self._config = config or _load_plugin_config()
        self._store = None
        self._retriever = None
        self._min_trust = float(self._config.get("min_trust_threshold", 0.3))

    @property
    def name(self) -> str:
        return "holographic"

    def is_available(self) -> bool:
        return True  # SQLite is always available, numpy is optional

    def save_config(self, values, hermes_home):
        """Write config to config.yaml under plugins.hermes-memory-store."""
        from pathlib import Path
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            # Write-back round-trip: raw read is correct (merged defaults
            # must not be persisted back into the user's file).
            from hermes_cli.config import read_user_config_raw
            existing = read_user_config_raw(config_path)
            existing.setdefault("plugins", {})
            existing["plugins"]["hermes-memory-store"] = values
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception:
            pass

    def get_config_schema(self):
        from hermes_constants import display_hermes_home
        _default_db = f"{display_hermes_home()}/memory_store.db"
        return [
            {"key": "db_path", "description": "SQLite database path", "default": _default_db},
            {"key": "auto_extract", "description": "Auto-extract facts at session end", "default": "false", "choices": ["true", "false"]},
            {"key": "default_trust", "description": "Default trust score for new facts", "default": "0.5"},
            {"key": "hrr_dim", "description": "HRR vector dimensions", "default": "1024"},
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home
        _hermes_home = str(get_hermes_home())
        _default_db = _hermes_home + "/memory_store.db"
        db_path = self._config.get("db_path", _default_db)
        # Expand $HERMES_HOME in user-supplied paths so config values like
        # "$HERMES_HOME/memory_store.db" or "~/.hermes/memory_store.db" both
        # resolve to the active profile's directory.
        if isinstance(db_path, str):
            db_path = db_path.replace("$HERMES_HOME", _hermes_home)
            db_path = db_path.replace("${HERMES_HOME}", _hermes_home)
        default_trust = float(self._config.get("default_trust", 0.5))
        hrr_dim = int(self._config.get("hrr_dim", 1024))
        hrr_weight = float(self._config.get("hrr_weight", 0.3))
        temporal_decay = int(self._config.get("temporal_decay_half_life", 0))

        self._store = MemoryStore(db_path=db_path, default_trust=default_trust, hrr_dim=hrr_dim)
        self._retriever = FactRetriever(
            store=self._store,
            temporal_decay_half_life=temporal_decay,
            hrr_weight=hrr_weight,
            hrr_dim=hrr_dim,
        )
        self._session_id = session_id

    def system_prompt_block(self) -> str:
        if not self._store:
            return ""
        try:
            total = self._store._conn.execute(
                "SELECT COUNT(*) FROM facts"
            ).fetchone()[0]
        except Exception:
            total = 0
        if total == 0:
            return (
                "# Holographic Memory\n"
                "Active. Empty fact store — proactively add facts the user would expect you to remember.\n"
                "Use fact_store(action='add') to store durable structured facts about people, projects, preferences, decisions.\n"
                "Use fact_feedback to rate facts after using them (trains trust scores)."
            )
        return (
            f"# Holographic Memory\n"
            f"Active. {total} facts stored with entity resolution and trust scoring.\n"
            f"Use fact_store to search, probe entities, reason across entities, or add facts.\n"
            f"Use fact_feedback to rate facts after using them (trains trust scores)."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._retriever or not query:
            return ""
        try:
            results = self._retriever.search(query, min_trust=self._min_trust, limit=5)
            if not results:
                return ""
            lines = []
            for r in results:
                trust = r.get("trust_score", r.get("trust", 0))
                lines.append(f"- [{trust:.1f}] {r.get('content', '')}")
            return "## Holographic Memory\n" + "\n".join(lines)
        except Exception as e:
            logger.debug("Holographic prefetch failed: %s", e)
            return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        # Holographic memory stores explicit facts via tools, not auto-sync.
        # The on_session_end hook handles auto-extraction if configured.
        pass

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [FACT_STORE_SCHEMA, FACT_FEEDBACK_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "fact_store":
            return self._handle_fact_store(args)
        elif tool_name == "fact_feedback":
            return self._handle_fact_feedback(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        # is_truthy_value: the config schema declares auto_extract as a string
        # enum ("false"/"true"), and a plain truthiness check treats the string
        # "false" as enabled (#57682).
        if not is_truthy_value(self._config.get("auto_extract", False)):
            return
        if not self._store or not messages:
            return
        self._auto_extract_facts(messages)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes as facts."""
        if action == "add" and self._store and content:
            try:
                category = "user_pref" if target == "user" else "general"
                self._store.add_fact(
                    content,
                    category=category,
                    source_session=getattr(self, "_session_id", "") or "",
                )
            except Exception as e:
                logger.debug("Holographic memory_write mirror failed: %s", e)

    def shutdown(self) -> None:
        # Release the shared SQLite connection deterministically on the
        # caller's thread. Dropping the reference alone leaves fd finalization
        # to GC, which keeps the connection (and its write lock) alive on a
        # long-running gateway and prolongs the "database is locked" contention
        # this store's shared-connection refcounting is meant to eliminate.
        # close() is idempotent and refcount-guarded, so siblings stay safe.
        if self._store is not None:
            try:
                self._store.close()
            except Exception as e:
                logger.debug("Holographic shutdown close() failed: %s", e)
        self._store = None
        self._retriever = None

    # -- Tool handlers -------------------------------------------------------

    def _handle_fact_store(self, args: dict) -> str:
        try:
            _strip_tool_call_tail(args)
            action = args["action"]
            store = self._store
            retriever = self._retriever

            arg_error = _validate_action_args(action, args)
            if arg_error:
                return tool_error(arg_error)

            if action == "add":
                fact_id = store.add_fact(
                    args["content"],
                    category=args.get("category", "general"),
                    tags=args.get("tags", ""),
                    source_session=getattr(self, "_session_id", "") or "",
                )
                return json.dumps({"fact_id": fact_id, "status": "added"})

            elif action == "search":
                results, meta = retriever.search(
                    args["query"],
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", self._min_trust)),
                    limit=int(args.get("limit", 10)),
                    with_meta=True,
                )
                payload = {"results": results, "count": len(results)}
                # Sibling keys, never a mutation of `results` or a filter on it:
                # the handoff's rule is that rows below the floor are still
                # returned and the caller decides. Same shape as the
                # {"found": false, ...} precedent in action=get — an unhelpful
                # answer is a RESULT, not an error. The verdict comes from
                # retrieval.no_confident_match so this door and the MCP bridge
                # cannot drift apart; see ABSTAIN_FLOOR.
                verdict = no_confident_match(meta)
                if verdict:
                    payload["no_confident_match"] = True
                    payload.update(verdict)
                return json.dumps(payload)

            elif action == "probe":
                results = retriever.probe(
                    args["entity"],
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "related":
                results = retriever.related(
                    args["entity"],
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "reason":
                # Emptiness is caught by _validate_action_args above.
                results = retriever.reason(
                    args["entities"],
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "contradict":
                results = retriever.contradict(
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "get":
                fact = store.get_fact(int(args["fact_id"]))
                # A missing fact is a RESULT, not an error — the caller is
                # usually verifying whether a claimed write landed.
                if fact is None:
                    return json.dumps(
                        {"found": False, "fact": None, "fact_id": int(args["fact_id"])}
                    )
                return json.dumps({"found": True, "fact": fact})

            elif action == "update":
                fid = int(args["fact_id"])
                before = store.get_fact(fid)
                if before is None:
                    return json.dumps({"updated": False, "fact_id": fid})
                if args.get("content") is not None:
                    refusal = _content_wipe_refusal(before, args)
                    if refusal:
                        return tool_error(refusal)
                updated = store.update_fact(
                    fid,
                    content=args.get("content"),
                    trust_delta=float(args["trust_delta"]) if "trust_delta" in args else None,
                    tags=args.get("tags"),
                    category=args.get("category"),
                    changed_by=getattr(self, "_session_id", "") or "",
                )
                after = store.get_fact(fid) or before
                return json.dumps(_update_result(fid, updated, before, after))

            elif action == "remove":
                fid = int(args["fact_id"])
                existing = store.get_fact(fid)
                removed = store.remove_fact(
                    fid, changed_by=getattr(self, "_session_id", "") or ""
                )
                if not removed and existing is not None:
                    # Distinguish a refusal from "no such fact" — a bare
                    # {"removed": false} reads as already-gone and invites a
                    # retry loop.
                    return json.dumps({
                        "removed": False,
                        "reason": (
                            f"category '{existing.get('category')}' is protected "
                            f"and cannot be removed; supersede it with "
                            f"action=update instead"
                        ),
                    })
                return json.dumps({"removed": removed})

            elif action == "list":
                # memory-entry is what MEMORY.md renders, and the dream job's
                # nightly incumbent review must judge every one of them — a
                # browsing-sized default silently unsatisfies that mandate
                # (2026-08-19: 10 of 13 reviewed, and the ones lost at the
                # trust tie were the newest promotions). Every other category
                # keeps the small default that holds cron context down; an
                # explicit limit always wins.
                category = args.get("category")
                limit = args.get("limit")
                if limit is None:
                    limit = 200 if category == "memory-entry" else 10
                facts = store.list_facts(
                    category=category,
                    min_trust=float(args.get("min_trust", 0.0)),
                    limit=int(limit),
                )
                return json.dumps({"facts": facts, "count": len(facts)})

            else:
                return tool_error(f"Unknown action: {action}")

        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    def _handle_fact_feedback(self, args: dict) -> str:
        try:
            fact_id = int(args["fact_id"])
            helpful = args["action"] == "helpful"
            result = self._store.record_feedback(
                fact_id,
                helpful=helpful,
                changed_by=getattr(self, "_session_id", "") or "",
            )
            return json.dumps(result)
        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    # -- Auto-extraction (on_session_end) ------------------------------------

    def _auto_extract_facts(self, messages: list) -> None:
        # Local import (pattern used in initialize()): the compressor module is
        # heavier than this plugin and is only needed when auto_extract is on.
        from agent.context_compressor import (
            _MERGED_PRIOR_CONTEXT_HEADER,
            _MERGED_SUMMARY_DELIMITER,
            is_compaction_summary_message,
        )

        def _pre_delimiter_user_segment(msg: dict):
            """Return the genuine user text preceding a merged-into-tail
            compaction summary, or None when the whole message is a summary.

            Merge-into-tail messages (agent/context_compressor.py ~3163-3190)
            wrap real prior tail content BEFORE ``_MERGED_SUMMARY_DELIMITER``,
            prefixed with ``_MERGED_PRIOR_CONTEXT_HEADER``, then append the
            generated handoff summary AFTER the delimiter. Dropping the whole
            row (as ``is_compaction_summary_message`` alone would suggest)
            discards that genuine pre-delimiter content too (#57690 review).
            Only the summary suffix must be excluded from harvesting.
            """
            content = msg.get("content", "")
            if not isinstance(content, str) or _MERGED_SUMMARY_DELIMITER not in content:
                return None
            pre = content.split(_MERGED_SUMMARY_DELIMITER, 1)[0]
            if pre.startswith(_MERGED_PRIOR_CONTEXT_HEADER):
                pre = pre[len(_MERGED_PRIOR_CONTEXT_HEADER):]
            pre = pre.strip()
            return pre or None

        _PREF_PATTERNS = [
            re.compile(r'\bI\s+(?:prefer|like|love|use|want|need)\s+(.+)', re.IGNORECASE),
            re.compile(r'\bmy\s+(?:favorite|preferred|default)\s+\w+\s+is\s+(.+)', re.IGNORECASE),
            re.compile(r'\bI\s+(?:always|never|usually)\s+(.+)', re.IGNORECASE),
        ]
        _DECISION_PATTERNS = [
            re.compile(r'\bwe\s+(?:decided|agreed|chose)\s+(?:to\s+)?(.+)', re.IGNORECASE),
            re.compile(r'\bthe\s+project\s+(?:uses|needs|requires)\s+(.+)', re.IGNORECASE),
        ]

        extracted = 0
        for msg in messages:
            if msg.get("role") != "user":
                continue
            # Compaction handoff summaries can be inserted as role="user"
            # messages; their prose reliably matches the decision patterns, so
            # without this guard the compactor's own output is stored as a
            # durable "fact" on every rollover (#57682). A merge-into-tail
            # summary also carries genuine pre-delimiter user content in the
            # SAME row; harvest that segment instead of dropping the whole
            # message (#57690 review).
            pre_delimiter_segment = _pre_delimiter_user_segment(msg)
            if pre_delimiter_segment is not None:
                content = pre_delimiter_segment
            elif is_compaction_summary_message(msg):
                continue
            else:
                content = msg.get("content", "")
            if not isinstance(content, str) or len(content) < 10:
                continue

            for pattern in _PREF_PATTERNS:
                if pattern.search(content):
                    try:
                        self._store.add_fact(content[:400], category="user_pref")
                        extracted += 1
                    except Exception:
                        pass
                    break

            for pattern in _DECISION_PATTERNS:
                if pattern.search(content):
                    try:
                        self._store.add_fact(content[:400], category="project")
                        extracted += 1
                    except Exception:
                        pass
                    break

        if extracted:
            logger.info("Auto-extracted %d facts from conversation", extracted)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the holographic memory provider with the plugin system."""
    config = _load_plugin_config()
    provider = HolographicMemoryProvider(config=config)
    ctx.register_memory_provider(provider)
