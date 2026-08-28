"""
SQLite-backed fact store with entity resolution and trust scoring.
Single-user Hermes memory store plugin.
"""

import logging
import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hrr_vector      BLOB,
    source_session  TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    entity_type TEXT DEFAULT 'unknown',
    aliases     TEXT DEFAULT '',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_entities (
    fact_id   INTEGER REFERENCES facts(fact_id),
    entity_id INTEGER REFERENCES entities(entity_id),
    PRIMARY KEY (fact_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_facts_trust    ON facts(trust_score DESC);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
CREATE INDEX IF NOT EXISTS idx_entities_name  ON entities(name);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, content=facts, content_rowid=fact_id);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TABLE IF NOT EXISTS memory_banks (
    bank_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_name  TEXT NOT NULL UNIQUE,
    vector     BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    fact_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_history (
    history_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id        INTEGER NOT NULL,
    op             TEXT NOT NULL,
    content        TEXT NOT NULL,
    category       TEXT,
    tags           TEXT,
    trust_score    REAL,
    source_session TEXT,
    changed_by_session TEXT DEFAULT '',
    fact_created_at TIMESTAMP,
    changed_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_fact_history_fact ON fact_history(fact_id);
"""

# Trust adjustment constants
_HELPFUL_DELTA   =  0.05
_UNHELPFUL_DELTA = -0.10
_TRUST_MIN       =  0.0
_TRUST_MAX       =  1.0

# Categories remove_fact() refuses without force=True. These are the durable
# lanes — a paper read, a lesson learned, a stated user preference — that an
# unattended consolidation run must never prune. fact_history now keeps a
# tombstone of every delete and update, so loss is recoverable, but recovery
# is manual archaeology; refusing the delete in the first place stays the
# invariant. The verdict lanes (researched, synthesis, hypothesis,
# open-question, general) stay prunable — superseding them is how
# consolidation is supposed to work. "memory-entry" rows ARE the
# always-in-context MEMORY.md (a renderer materializes the file from them);
# retiring one is a trust decay via update, never a delete.
PROTECTED_CATEGORIES = frozenset(
    {"paper", "project", "lesson", "user_pref", "activity", "memory-entry"}
)

# Entity extraction patterns
_RE_CAPITALIZED  = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
# Bounded: an unbounded span crosses newlines and swallows whole log records.
_RE_DOUBLE_QUOTE = re.compile(r'"([^"\n]{1,200})"')
_RE_AKA          = re.compile(
    r'(\w+(?:\s+\w+)*)\s+(?:aka|also known as)\s+(\w+(?:\s+\w+)*)',
    re.IGNORECASE,
)
# There is deliberately NO single-quote rule. An apostrophe inside a word
# ("Earth's", "don't", "et al.'s") is not a quote delimiter, but a naive
# r"'([^']+)'" treats it as one and captures everything up to the next
# apostrophe anywhere later in the text — which produced multi-hundred-character
# prose blobs. Every boundary-lookaround repair still leaks on ".'" and ")'",
# and measured over a real corpus the rule yielded no candidate that ever
# linked two facts, so it is removed rather than patched.

# Identifier-shaped tokens: the class that actually joins facts, and the class
# the other rules are structurally blind to (they cannot see a bare acronym,
# CamelCase or hyphenated name). Compounds must carry an uppercase character so
# "GRACE-FO"/"Gauss-Newton" match while "real-time"/"climate-driven" do not;
# there is deliberately no bare-uppercase alternative, because [A-Z]{2,} matches
# ALL-CAPS section markers (PAPER, LESSON, CONFIRMED) and would make them the
# highest-degree nodes in the graph.
_RE_IDENTIFIER = re.compile(
    r'\b('
    r'(?=[\w-]*[A-Z])[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+'  # GRACE-FO
    r'|[A-Z][A-Za-z]*[0-9]+[A-Za-z0-9]*'                          # SGP4, ERA5
    r'|[A-Z][a-z]+[A-Z][A-Za-z0-9]*'                              # LightRAG
    r'|[a-z]+[A-Z][A-Za-z0-9]*'                                   # arXiv
    r'|[a-z][a-z0-9]*(?:_[a-z0-9]+)+'                             # fact_store
    r')(?::([\w.\-]*[\w\-]))?'                                    # arXiv:2607.19083
)
# The lowercase-underscore branch has no uppercase requirement on purpose:
# snake_case in prose is essentially always an identifier. Lowercase HYPHEN
# compounds stay excluded — "real-time"/"climate-driven" are ordinary prose,
# and lexically inseparable from "llama-swap"; those names reach the graph
# through the tags path instead (a tag is deliberate metadata, not prose).

# Single capitalized words — the shape every rule above is structurally blind
# to, which left "Hermes" unreachable by the entity probe in 111 of the 113
# facts naming it. A sentence-initial capital is just orthography, so a word
# qualifies only where at least one occurrence is MID-sentence, and never from
# inside a span the multi-word rule already claimed ("Claude" inside
# "Claude Code" is not separately promoted from that occurrence).
_RE_CAP_SINGLE = re.compile(r'\b([A-Z][a-z]{2,})\b')

# Capitalized mid-sentence for reasons other than being a name: date prose,
# Python literals quoted in lessons, and generic title-case words. The last
# row is empirical — the only non-names in the top 40 by fact-degree when the
# rule was dry-run against the live 455-fact corpus (2026-08-14); everything
# above them was a genuine name. Extend from measurement, not speculation.
_SINGLE_NAME_STOP = frozenset("""
january february march april june july august september october november december
jan feb mar apr jun jul aug sep sept oct nov dec
monday tuesday wednesday thursday friday saturday sunday
mon tue wed thu fri sat sun
true false none
documents level methods applications multi
""".split())

# Characters that end a sentence or open a structural context (bullet, quote,
# bracket, table cell). A capital whose nearest preceding non-blank character
# is one of these — or which starts the text — is sentence-initial.
_SENTENCE_OPENERS = '.!?:;\n"\'`([{*>|-—'


def _is_mid_sentence(text: str, start: int) -> bool:
    i = start - 1
    while i >= 0 and text[i] in ' \t':
        i -= 1
    if i < 0:
        return False
    return text[i] not in _SENTENCE_OPENERS

# Entity validation bounds. 40 is the length above which a real corpus contained
# no entity linked to more than one fact — and joining facts is the entire
# purpose of an entity here. The longest genuine multi-word name the rules
# produce is well inside it.
_ENTITY_MIN_LEN = 2
_ENTITY_MAX_LEN = 40

# A proper name never begins or ends with a function word. A capitalized span
# that does is a sentence fragment ("The Gauss", "Uses Telemetry", "Profiles
# Using") — what _RE_CAPITALIZED emits when it splits a longer phrase at an
# interior lowercase word. Deliberately NOT unified with
# FactRetriever._FTS_STOPWORDS: that one filters query tokens, this one rejects
# name edges, and the two lists must be free to diverge.
_EDGE_WORDS = frozenset("""
a an the this that these those it its we they he she
is are was were be been do does did can could will would shall should may might must
of in on at to for from by with via per as into over under between during about
and or but so not no if when while then because which what how why where
use uses used using run runs ran add adds added see also new only all both more most now
""".split())

# The tail a contraction/possessive leaves behind when a quote span opened on
# its apostrophe ("Earth's" -> "s ...", "don't" -> "t ..."). The rules above can
# no longer produce these, but rows written before the fix can, and the short
# ones sit inside the length bound where nothing else catches them. Matched
# lowercase and only as a leading word, so "T Tauri" / "B Ring" are unaffected.
_CONTRACTION_HEADS = frozenset(("s", "t", "ll", "re", "ve", "d", "m"))


def _clamp_trust(value: float) -> float:
    return max(_TRUST_MIN, min(_TRUST_MAX, value))


def _is_entity_like(name: str) -> bool:
    """True if a candidate looks like a name rather than a fragment of prose."""
    if not (_ENTITY_MIN_LEN <= len(name) <= _ENTITY_MAX_LEN):
        return False
    words = name.split()
    if len(words) > 1 and (
        words[0].lower() in _EDGE_WORDS or words[-1].lower() in _EDGE_WORDS
    ):
        return False
    if len(words) > 1 and words[0] in _CONTRACTION_HEADS:
        return False
    return True


def _tag_entities(tags: str) -> list[str]:
    """Entity candidates from a comma-separated tags field.

    Tags are deliberate metadata — the one place a fact's author names its
    subjects directly, in whatever casing the corpus uses ("hermes",
    "llama-swap", "open-question"). None of the prose rules can see those
    shapes, so tags get their own path into the graph.
    """
    out: list[str] = []
    for tag in (tags or "").split(","):
        t = tag.strip()
        if t and _is_entity_like(t):
            out.append(t)
    return out


class MemoryStore:
    """SQLite-backed fact store with entity resolution and trust scoring."""

    # --- Process-wide shared connection registry -------------------------
    # SQLite permits only one writer at a time. Each MemoryStore instance used
    # to open its own connection guarded by its own RLock, so the several
    # providers that coexist in one process (the main agent plus every
    # delegate_task subagent) raced as independent WAL writers. Combined with
    # writes that were not rolled back on error, one connection could leave an
    # open write transaction that pinned the write lock and made every other
    # connection's write fail with "database is locked" for the full busy
    # timeout. All instances for the same database now share ONE connection and
    # ONE re-entrant lock, so access is fully serialized and cross-connection
    # contention is impossible. The shared connection is refcounted, so closing
    # one instance never tears the connection out from under a live sibling.
    _shared: dict = {}
    _shared_guard = threading.Lock()

    def __init__(
        self,
        db_path: "str | Path | None" = None,
        default_trust: float = 0.5,
        hrr_dim: int = 1024,
    ) -> None:
        if db_path is None:
            from hermes_constants import get_hermes_home
            db_path = str(get_hermes_home() / "memory_store.db")
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_trust = _clamp_trust(default_trust)
        self.hrr_dim = hrr_dim
        self._hrr_available = hrr._HAS_NUMPY

        # Acquire (or open) the process-wide shared connection for this DB.
        # resolve() (not just expanduser) so symlinked/relative paths to the
        # same file share ONE connection instead of silently reintroducing
        # the multi-writer contention this registry exists to prevent.
        try:
            self._key = str(self.db_path.resolve())
        except OSError:
            self._key = str(self.db_path)
        with MemoryStore._shared_guard:
            entry = MemoryStore._shared.get(self._key)
            if entry is None:
                conn = sqlite3.connect(
                    self._key,
                    check_same_thread=False,
                    timeout=10.0,
                    # Autocommit: every statement is its own transaction, so a
                    # write that raises mid-method can never leave a dangling
                    # transaction (and its write lock) open. The explicit
                    # commit() calls below become harmless no-ops.
                    isolation_level=None,
                )
                conn.row_factory = sqlite3.Row
                entry = {"conn": conn, "lock": threading.RLock(), "refs": 0, "ready": False}
                MemoryStore._shared[self._key] = entry
            entry["refs"] += 1
            self._entry = entry
            self._conn = entry["conn"]
            self._lock = entry["lock"]

        # Initialise the schema once per shared connection.
        with self._lock:
            if not self._entry["ready"]:
                self._init_db()
                self._entry["ready"] = True

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables, indexes, and triggers if they do not exist. Enable WAL mode."""
        # Use the shared WAL-fallback helper so memory_store.db degrades
        # gracefully on NFS/SMB/FUSE-mounted HERMES_HOME (same issue as
        # state.db / kanban.db — see hermes_state._WAL_INCOMPAT_MARKERS).
        from hermes_state import apply_wal_with_fallback
        apply_wal_with_fallback(self._conn, db_label="memory_store.db (holographic)")
        self._conn.executescript(_SCHEMA)
        # Migrate: add columns if missing (safe for existing databases)
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(facts)").fetchall()}
        if "hrr_vector" not in columns:
            self._conn.execute("ALTER TABLE facts ADD COLUMN hrr_vector BLOB")
        if "source_session" not in columns:
            self._conn.execute("ALTER TABLE facts ADD COLUMN source_session TEXT DEFAULT ''")
        hist_columns = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(fact_history)").fetchall()
        }
        if hist_columns and "changed_by_session" not in hist_columns:
            self._conn.execute(
                "ALTER TABLE fact_history ADD COLUMN changed_by_session TEXT DEFAULT ''"
            )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_fact(
        self,
        content: str,
        category: str = "general",
        tags: str = "",
        source_session: str = "",
    ) -> int:
        """Insert a fact and return its fact_id.

        Deduplicates by content (UNIQUE constraint). On duplicate, returns
        the existing fact_id without modifying the row. Extracts entities from
        the content and links them to the fact.

        source_session records which session wrote the fact, joinable against
        state.db / the trace archive — the provenance that made the 2026-08-14
        deletion recovery possible was reconstructed by hand; this makes it a
        column.
        """
        with self._lock:
            content = content.strip()
            if not content:
                raise ValueError("content must not be empty")

            try:
                cur = self._conn.execute(
                    """
                    INSERT INTO facts (content, category, tags, trust_score, source_session)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (content, category, tags, self.default_trust, source_session or ""),
                )
                self._conn.commit()
                fact_id: int = cur.lastrowid  # type: ignore[assignment]
            except sqlite3.IntegrityError:
                # Duplicate content — return existing id
                row = self._conn.execute(
                    "SELECT fact_id FROM facts WHERE content = ?", (content,)
                ).fetchone()
                return int(row["fact_id"])

            # Entity extraction and linking — content prose plus tags. A tag
            # is the author naming the fact's subject directly; the prose
            # rules cannot see lowercase names like "hermes" or "llama-swap".
            for name in self._extract_entities(content) + _tag_entities(tags):
                entity_id = self._resolve_entity(name)
                self._link_fact_entity(fact_id, entity_id)

            # Compute HRR vector after entity linking
            self._compute_hrr_vector(fact_id, content)
            self._rebuild_bank(category)

            return fact_id

    def search_facts(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Full-text search over facts using FTS5.

        Returns a list of fact dicts ordered by FTS5 rank, then trust_score
        descending. Also increments retrieval_count for matched facts.
        """
        with self._lock:
            query = query.strip()
            if not query:
                return []

            # FTS5 AND-joins tokens by default, which zeroes out recall on
            # natural-language queries. Reuse the retriever's sanitizer
            # (stopword drop + OR-join content tokens). Imported lazily to
            # avoid a store->retrieval import cycle.
            from plugins.memory.holographic.retrieval import FactRetriever

            match_query = FactRetriever._sanitize_fts_query(query)
            params: list = [match_query, min_trust]
            category_clause = ""
            if category is not None:
                category_clause = "AND f.category = ?"
                params.append(category)
            params.append(limit)

            sql = f"""
                SELECT f.fact_id, f.content, f.category, f.tags,
                       f.trust_score, f.retrieval_count, f.helpful_count,
                       f.created_at, f.updated_at
                FROM facts f
                JOIN facts_fts fts ON fts.rowid = f.fact_id
                WHERE facts_fts MATCH ?
                  AND f.trust_score >= ?
                  {category_clause}
                ORDER BY fts.rank, f.trust_score DESC
                LIMIT ?
            """

            rows = self._conn.execute(sql, params).fetchall()
            results = [self._row_to_dict(r) for r in rows]

            if results:
                ids = [r["fact_id"] for r in results]
                placeholders = ",".join("?" * len(ids))
                self._conn.execute(
                    f"UPDATE facts SET retrieval_count = retrieval_count + 1 WHERE fact_id IN ({placeholders})",
                    ids,
                )
                self._conn.commit()

            return results

    def _snapshot_fact(self, fact_id: int, op: str, changed_by: str = "") -> None:
        """Copy a fact's current row into fact_history before mutating it.

        Every update and delete leaves the prior version behind, so no write
        path can destroy the only copy of a fact. Recovery is a SELECT from
        fact_history, not archaeology in state.db message rows.

        The snapshot is the BEFORE image, so ``source_session`` in it is the
        session that CREATED the fact. ``changed_by`` is the session
        performing this mutation — the two differ whenever one job edits
        another's fact, which is the common case for the nightly jobs. Without
        the distinction a tombstone cannot name the job that deleted a row.

        Callers must run this inside the same transaction as the mutation (see
        ``_write_txn``); a snapshot committed for a write that then failed is
        a false audit record.
        """
        self._conn.execute(
            """
            INSERT INTO fact_history
                (fact_id, op, content, category, tags, trust_score,
                 source_session, changed_by_session, fact_created_at)
            SELECT fact_id, ?, content, category, tags, trust_score,
                   source_session, ?, created_at
            FROM facts WHERE fact_id = ?
            """,
            (op, changed_by or "", fact_id),
        )

    @contextmanager
    def _write_txn(self):
        """Commit a snapshot+mutation pair together, or roll both back.

        The connection runs in autocommit (``isolation_level=None``), so each
        statement was its own transaction: ``_snapshot_fact`` committed the
        history row the instant it ran, and when the mutation then raised —
        e.g. UNIQUE(content) on an update to text another fact already holds —
        the row stayed. ``rollback()`` cannot undo it; there is no open
        transaction to roll back. That is how fact_history came to assert an
        update that never happened.

        An explicit BEGIN is therefore required to bind the pair. It is closed
        on every path so the write lock is never left dangling, which is what
        autocommit was chosen to guarantee (see the connect() comment).
        """
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._conn.execute("COMMIT")
        except Exception:
            try:
                self._conn.execute("ROLLBACK")
            except Exception:
                logger.exception("fact_history rollback failed for a failed write")
            raise

    def update_fact(
        self,
        fact_id: int,
        content: str | None = None,
        trust_delta: float | None = None,
        tags: str | None = None,
        category: str | None = None,
        changed_by: str = "",
    ) -> bool:
        """Partially update a fact. Trust is clamped to [0, 1].

        Returns True if the row existed, False otherwise. The pre-update row
        is snapshotted into fact_history, stamped with *changed_by* so the
        mutating session is recoverable; the snapshot and the UPDATE commit
        together or not at all.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, trust_score, category FROM facts WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()
            if row is None:
                return False

            assignments: list[str] = ["updated_at = CURRENT_TIMESTAMP"]
            params: list = []

            if content is not None:
                assignments.append("content = ?")
                params.append(content.strip())
            if tags is not None:
                assignments.append("tags = ?")
                params.append(tags)
            if category is not None:
                assignments.append("category = ?")
                params.append(category)
            if trust_delta is not None:
                new_trust = _clamp_trust(row["trust_score"] + trust_delta)
                assignments.append("trust_score = ?")
                params.append(new_trust)

            params.append(fact_id)
            with self._write_txn():
                self._snapshot_fact(fact_id, "update", changed_by)
                self._conn.execute(
                    f"UPDATE facts SET {', '.join(assignments)} WHERE fact_id = ?",
                    params,
                )

            # Re-derive entity links and the HRR vector when the text they
            # encode changed. The content-change path drops every existing
            # link (stale prose entities must not linger), so tag-derived
            # links are restored from the row's CURRENT tags — a tags-only
            # change adds links without dropping any.
            if content is not None or tags is not None:
                row2 = self._conn.execute(
                    "SELECT content, tags FROM facts WHERE fact_id = ?",
                    (fact_id,),
                ).fetchone()
                if content is not None:
                    self._conn.execute(
                        "DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,)
                    )
                    for name in self._extract_entities(row2["content"]):
                        entity_id = self._resolve_entity(name)
                        self._link_fact_entity(fact_id, entity_id)
                for name in _tag_entities(row2["tags"]):
                    entity_id = self._resolve_entity(name)
                    self._link_fact_entity(fact_id, entity_id)
                self._conn.commit()
                self._compute_hrr_vector(fact_id, row2["content"])
            # Rebuild the destination bank — and the SOURCE bank too when the
            # fact changed category, or the departed bank goes on counting it.
            # remove_fact already rebuilds the category a fact leaves; a
            # migration is that same departure with an arrival attached.
            old_cat = row["category"]
            cat = category or old_cat
            self._rebuild_bank(cat)
            if cat != old_cat:
                self._rebuild_bank(old_cat)

            return True

    def get_fact(self, fact_id: int) -> dict | None:
        """Fetch one fact by id, or None if it does not exist.

        Exact lookup — no ranking, no trust floor, no retrieval-count bump.
        This is the verification path: ranked search cannot prove a fact is
        absent (a low-ranked or sub-min_trust row simply doesn't surface).
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, content, category, tags, trust_score,"
                " retrieval_count, helpful_count, created_at, updated_at"
                " FROM facts WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def remove_fact(
        self, fact_id: int, force: bool = False, changed_by: str = ""
    ) -> bool:
        """Delete a fact and its entity links. Returns True if the row existed.

        Facts in :data:`PROTECTED_CATEGORIES` are refused unless *force* is set.
        These are the durable lanes — a paper read, a lesson learned, a stated
        user preference. On 2026-08-14 an unattended consolidation run deleted
        six of them in one turn, two of them hours old, against a prompt that
        told it never to. Prose in a prompt is not an invariant; this is.

        Every delete that proceeds (including force) first snapshots the row
        into fact_history, so no deletion is ever the last copy.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, category FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                return False
            if not force and row["category"] in PROTECTED_CATEGORIES:
                return False

            with self._write_txn():
                self._snapshot_fact(fact_id, "delete", changed_by)
                self._conn.execute(
                    "DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,)
                )
                self._conn.execute("DELETE FROM facts WHERE fact_id = ?", (fact_id,))
            self._rebuild_bank(row["category"])
            return True

    def list_facts(
        self,
        category: str | None = None,
        min_trust: float = 0.0,
        limit: int = 50,
    ) -> list[dict]:
        """Browse facts ordered by trust_score descending, fact_id ascending.

        Optionally filter by category and minimum trust score. The fact_id
        tiebreak makes a LIMIT cut deterministic: without it, equal-trust
        facts were returned in arbitrary order, so which one fell off a
        truncated listing changed between calls.
        """
        with self._lock:
            params: list = [min_trust]
            category_clause = ""
            if category is not None:
                category_clause = "AND category = ?"
                params.append(category)
            params.append(limit)

            sql = f"""
                SELECT fact_id, content, category, tags, trust_score,
                       retrieval_count, helpful_count, created_at, updated_at
                FROM facts
                WHERE trust_score >= ?
                  {category_clause}
                ORDER BY trust_score DESC, fact_id ASC
                LIMIT ?
            """
            rows = self._conn.execute(sql, params).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def mark_retrieved(self, fact_ids: list[int]) -> None:
        """Increment retrieval_count for facts surfaced by any retrieval path.

        search_facts() bumps its own matches inline; this is for the OTHER
        surfacing paths (FactRetriever.search — which serves per-turn prefetch
        injection, the fact_store search action, and the MCP bridge). Before
        this existed, ambient prefetch recall was invisible: a fact could be
        injected into context every day and still read retrieval_count=0,
        which is how 296 of 455 facts came to look unread.
        """
        if not fact_ids:
            return
        with self._lock:
            placeholders = ",".join("?" * len(fact_ids))
            self._conn.execute(
                f"UPDATE facts SET retrieval_count = retrieval_count + 1"
                f" WHERE fact_id IN ({placeholders})",
                list(fact_ids),
            )
            self._conn.commit()

    def record_feedback(self, fact_id: int, helpful: bool, changed_by: str = "") -> dict:
        """Record user feedback and adjust trust asymmetrically.

        helpful=True  -> trust += 0.05, helpful_count += 1
        helpful=False -> trust -= 0.10

        Snapshots the pre-feedback row into fact_history (op='feedback',
        stamped with *changed_by*) — before this, a feedback write was the
        one mutation path invisible to both fact_history and the scheduler's
        per-run ledger, so a mass helpful=False downgrade would have left no
        audit trail (fid 737 on 2026-08-19 proved the blind spot).

        Returns a dict with fact_id, old_trust, new_trust, helpful_count.
        Raises KeyError if fact_id does not exist.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, trust_score, helpful_count FROM facts WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"fact_id {fact_id} not found")

            old_trust: float = row["trust_score"]
            delta = _HELPFUL_DELTA if helpful else _UNHELPFUL_DELTA
            new_trust = _clamp_trust(old_trust + delta)

            helpful_increment = 1 if helpful else 0
            with self._write_txn():
                self._snapshot_fact(fact_id, "feedback", changed_by)
                self._conn.execute(
                    """
                    UPDATE facts
                    SET trust_score    = ?,
                        helpful_count  = helpful_count + ?,
                        updated_at     = CURRENT_TIMESTAMP
                    WHERE fact_id = ?
                    """,
                    (new_trust, helpful_increment, fact_id),
                )

            return {
                "fact_id":      fact_id,
                "old_trust":    old_trust,
                "new_trust":    new_trust,
                "helpful_count": row["helpful_count"] + helpful_increment,
            }

    # ------------------------------------------------------------------
    # Entity helpers
    # ------------------------------------------------------------------

    def _extract_entities(self, text: str) -> list[str]:
        """Extract entity candidates from text using simple regex rules.

        Rules applied (in order):
        1. Capitalized multi-word phrases  e.g. "John Doe"
        2. Double-quoted terms             e.g. "Python"
        3. AKA patterns                    e.g. "Guido aka BDFL" -> two entities
        4. Identifier tokens               e.g. GRACE-FO, SGP4, arXiv:2607.19083
        5. Single capitalized words with a mid-sentence occurrence, e.g.
           "the Hermes gateway" -> Hermes (sentence-initial capitals are
           orthography, not names)

        Every candidate must satisfy _is_entity_like, so prose spans, error
        strings and sentence fragments never become entities.

        Returns a deduplicated list preserving first-seen order.
        """
        seen: set[str] = set()
        candidates: list[str] = []

        def _add(name: str) -> None:
            stripped = name.strip().strip('.,;:')
            if _is_entity_like(stripped) and stripped.lower() not in seen:
                seen.add(stripped.lower())
                candidates.append(stripped)

        multiword_spans = []
        for m in _RE_CAPITALIZED.finditer(text):
            _add(m.group(1))
            multiword_spans.append(m.span(1))

        for m in _RE_DOUBLE_QUOTE.finditer(text):
            _add(m.group(1))

        for m in _RE_AKA.finditer(text):
            _add(m.group(1))
            _add(m.group(2))

        # group(0), not group(1): keeps the optional ":suffix" (arXiv:2607.19083).
        for m in _RE_IDENTIFIER.finditer(text):
            _add(m.group(0))

        for m in _RE_CAP_SINGLE.finditer(text):
            word = m.group(1)
            lw = word.lower()
            if lw in _EDGE_WORDS or lw in _SINGLE_NAME_STOP:
                continue
            s, e = m.span(1)
            if any(ms <= s and e <= me for ms, me in multiword_spans):
                continue
            if _is_mid_sentence(text, s):
                _add(word)

        return candidates

    def _resolve_entity(self, name: str) -> int:
        """Find an existing entity by name or alias (case-insensitive) or create one.

        Returns the entity_id.
        """
        # Exact name match. '=' with an explicit NOCASE collation, not LIKE:
        # LIKE interprets '_' and '%' inside the CANDIDATE as wildcards, so a
        # name like "model_spec" silently resolved to whichever row the index
        # scan reached first — a different entity. _compute_hrr_vector then
        # re-reads that wrong name from the DB and encodes it into the fact
        # vector. NOCASE preserves the case-insensitivity LIKE gave for free;
        # a bare '=' would fork every entity by capitalisation.
        row = self._conn.execute(
            "SELECT entity_id FROM entities WHERE name = ? COLLATE NOCASE",
            (name,),
        ).fetchone()
        if row is not None:
            return int(row["entity_id"])

        # Search aliases — aliases stored as comma-separated; use LIKE with %
        # boundaries. ESCAPE is required here for the same reason: the
        # parameter is user data, not a pattern.
        alias_row = self._conn.execute(
            """
            SELECT entity_id FROM entities
            WHERE ',' || aliases || ',' LIKE '%,' || ? || ',%' ESCAPE '\\'
            """,
            (name.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_'),),
        ).fetchone()
        if alias_row is not None:
            return int(alias_row["entity_id"])

        # Create new entity
        cur = self._conn.execute(
            "INSERT INTO entities (name) VALUES (?)", (name,)
        )
        self._conn.commit()
        return int(cur.lastrowid)  # type: ignore[return-value]

    def _link_fact_entity(self, fact_id: int, entity_id: int) -> None:
        """Insert into fact_entities, silently ignore if the link already exists."""
        self._conn.execute(
            """
            INSERT OR IGNORE INTO fact_entities (fact_id, entity_id)
            VALUES (?, ?)
            """,
            (fact_id, entity_id),
        )
        self._conn.commit()

    def _compute_hrr_vector(self, fact_id: int, content: str) -> None:
        """Compute and store HRR vector for a fact. No-op if numpy unavailable."""
        with self._lock:
            # The flag was captured when THIS process first imported the
            # holographic module. Re-check before giving up: a process that
            # started while numpy was briefly unavailable would otherwise write
            # NULL vectors forever (see refresh_numpy_availability).
            if not self._hrr_available and hrr.refresh_numpy_availability():
                self._hrr_available = True
            if not self._hrr_available:
                # Do not skip quietly. A fact with no vector is invisible to
                # every semantic path (probe/related/reason and the HRR term in
                # search) while still looking perfectly healthy in the table —
                # the 2026-08-09 investigation burned hours because the only
                # symptom was a NULL column and a 5-minute repair timer quietly
                # papering over it. If this fires, numpy failed to import in
                # THIS process (the flag is captured once at module import), so
                # reinstalling numpy will not help a process that is already
                # running; it has to be restarted.
                logger.warning(
                    "HRR vector NOT computed for fact %s — numpy unavailable in "
                    "this process. The fact is stored but will not be found by "
                    "semantic retrieval until its vector is rebuilt. "
                    "numpy import error: %s",
                    fact_id,
                    getattr(hrr, "_NUMPY_IMPORT_ERROR", "") or "<none recorded>",
                )
                return

            # Get entities linked to this fact
            rows = self._conn.execute(
                """
                SELECT e.name FROM entities e
                JOIN fact_entities fe ON fe.entity_id = e.entity_id
                WHERE fe.fact_id = ?
                """,
                (fact_id,),
            ).fetchall()
            entities = [row["name"] for row in rows]

            vector = hrr.encode_fact(content, entities, self.hrr_dim)
            self._conn.execute(
                "UPDATE facts SET hrr_vector = ? WHERE fact_id = ?",
                (hrr.phases_to_bytes(vector), fact_id),
            )
            self._conn.commit()

    def _rebuild_bank(self, category: str) -> None:
        """Full rebuild of a category's memory bank from all its fact vectors."""
        with self._lock:
            if not self._hrr_available:
                return

            bank_name = f"cat:{category}"
            rows = self._conn.execute(
                "SELECT hrr_vector FROM facts WHERE category = ? AND hrr_vector IS NOT NULL",
                (category,),
            ).fetchall()

            if not rows:
                self._conn.execute("DELETE FROM memory_banks WHERE bank_name = ?", (bank_name,))
                self._conn.commit()
                return

            vectors = [hrr.bytes_to_phases(row["hrr_vector"], dim=self.hrr_dim) for row in rows]
            bank_vector = hrr.bundle(*vectors)
            fact_count = len(vectors)

            # SNR is recorded but NOT warned about: the bundled bank vector has
            # no reader. FactRetriever.probe() looks the row up and uses
            # only its existence to select a branch, then scores against the
            # individual fact vectors ("deliberately equivalent to the direct
            # scoring below" — see retrieval.py). A degrading bank SNR therefore
            # cannot degrade any retrieval, and warning that it might is what
            # sends the next reader off to raise hrr_dim or prune facts for
            # nothing. cat:lesson crossed dim/4 on 2026-08-28 and would have
            # logged this on every lesson write from then on.
            hrr.snr_estimate(self.hrr_dim, fact_count, warn=False)

            self._conn.execute(
                """
                INSERT INTO memory_banks (bank_name, vector, dim, fact_count, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(bank_name) DO UPDATE SET
                    vector = excluded.vector,
                    dim = excluded.dim,
                    fact_count = excluded.fact_count,
                    updated_at = excluded.updated_at
                """,
                (bank_name, hrr.phases_to_bytes(bank_vector), self.hrr_dim, fact_count),
            )
            self._conn.commit()

    def backfill_entity_links(self) -> dict:
        """Additively re-run entity extraction (content + tags) over every fact.

        Adds links that the current rules find and past rules missed. Never
        removes a link or an entity — hundreds of legacy entity rows are
        tag-derived and unregenerable, so subtractive re-extraction is
        forbidden (2026-07-29 lesson). Facts that gained links get their HRR
        vector recomputed and their category banks rebuilt, since the vector
        encodes the linked entities.

        Returns {"facts_scanned", "facts_changed", "links_added"}.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT fact_id, content, tags, category FROM facts"
            ).fetchall()

            links_added = 0
            changed: list = []
            for row in rows:
                names = (self._extract_entities(row["content"])
                         + _tag_entities(row["tags"]))
                if not names:
                    continue
                before = self._conn.execute(
                    "SELECT COUNT(*) FROM fact_entities WHERE fact_id = ?",
                    (row["fact_id"],),
                ).fetchone()[0]
                for name in names:
                    entity_id = self._resolve_entity(name)
                    self._link_fact_entity(row["fact_id"], entity_id)
                after = self._conn.execute(
                    "SELECT COUNT(*) FROM fact_entities WHERE fact_id = ?",
                    (row["fact_id"],),
                ).fetchone()[0]
                if after > before:
                    links_added += after - before
                    changed.append(row)

            categories: set[str] = set()
            for row in changed:
                self._compute_hrr_vector(row["fact_id"], row["content"])
                categories.add(row["category"])
            for category in categories:
                self._rebuild_bank(category)

            return {
                "facts_scanned": len(rows),
                "facts_changed": len(changed),
                "links_added": links_added,
            }

    def rebuild_all_vectors(self, dim: int | None = None) -> int:
        """Recompute all HRR vectors + banks from text. For recovery/migration.

        Returns the number of facts processed.
        """
        with self._lock:
            if not self._hrr_available:
                return 0

            if dim is not None:
                self.hrr_dim = dim

            rows = self._conn.execute(
                "SELECT fact_id, content, category FROM facts"
            ).fetchall()

            categories: set[str] = set()
            for row in rows:
                self._compute_hrr_vector(row["fact_id"], row["content"])
                categories.add(row["category"])

            for category in categories:
                self._rebuild_bank(category)

            return len(rows)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Convert a sqlite3.Row to a plain dict."""
        return dict(row)

    @classmethod
    def release_all_under(cls, directory: "str | Path") -> int:
        """Force-close every shared connection whose database lives under ``directory``.

        ``close()`` is refcount-driven, so a live holder (e.g. an agent's
        memory provider) keeps a profile's SQLite handle open indefinitely.
        That is exactly what a profile delete must break on Windows: the
        desktop's main ``serve`` process opens ``memory_store.db`` for every
        known profile, and ``rmtree`` of the profile directory fails with
        ``WinError 32`` while any of those handles is open (#88347). This
        closes the matching connections unconditionally — the directory is
        going away, so later use by a stale holder is expected to fail — and
        returns how many were closed. In a process that holds none (e.g. the
        CLI deleting from outside serve) this is a harmless no-op returning 0.
        """
        root = os.path.normcase(str(Path(directory).expanduser().resolve())) + os.sep
        with cls._shared_guard:
            # Snapshot the keys first so the registry stays stable while
            # connections are closed inside their per-database locks (closing
            # can run no user code, but this keeps the invariant obvious).
            doomed = [
                key
                for key in cls._shared
                if os.path.normcase(key).startswith(root)
            ]
            for key in doomed:
                entry = cls._shared.pop(key)
                try:
                    with entry["lock"]:
                        entry["conn"].close()
                except Exception:
                    # A connection that is already closed or broken must not
                    # abort releasing its siblings.
                    pass
        return len(doomed)

    def close(self) -> None:
        """Release this instance's reference to the shared connection.

        The underlying connection is closed only when the last MemoryStore
        referencing the same database is closed, so closing one instance can
        never break sibling instances that still hold it. Idempotent.
        """
        if getattr(self, "_entry", None) is None:
            return
        with MemoryStore._shared_guard:
            entry = self._entry
            if entry is None:
                return
            entry["refs"] -= 1
            if entry["refs"] <= 0:
                try:
                    entry["conn"].close()
                finally:
                    # Pop only OUR entry. After release_all_under() force-
                    # closed this entry (profile delete, #88347) a same-path
                    # store may have re-registered a FRESH entry under the
                    # same key; a stale holder's late close() must not evict
                    # it — that would silently reintroduce the multi-writer
                    # contention this registry exists to prevent.
                    if MemoryStore._shared.get(self._key) is entry:
                        MemoryStore._shared.pop(self._key, None)
            self._entry = None

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
