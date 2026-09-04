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
    from . import embeddings
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]
    import embeddings  # type: ignore[no-redef]

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
    source_session  TEXT DEFAULT '',
    valid_from      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    valid_until     TIMESTAMP
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

CREATE TABLE IF NOT EXISTS fact_markers (
    fact_id INTEGER NOT NULL,
    marker  TEXT NOT NULL,
    set_by  TEXT DEFAULT '',
    set_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (fact_id, marker)
);

CREATE INDEX IF NOT EXISTS idx_fact_markers_marker ON fact_markers(marker);

-- Dense retrieval vectors, one row per fact (2026-09-04). The FTS5 index and
-- the HRR vector are both computed locally; this is the only stored artefact
-- that comes from a model, so `model` and `dim` travel WITH the blob: a vector
-- written by a different embedding model is not comparable to one written by
-- this one, and a store that outlives a model swap has to be able to say which
-- rows are stale rather than silently mixing two vector spaces.
--
-- Deliberately a separate table rather than a column on facts: it is optional
-- (a store with none of these rows behaves exactly as it did before), it is
-- refreshed on a different schedule from the row it describes, and a 4 KB blob
-- on facts would be read by every SELECT f.* in the retrieval path.
CREATE TABLE IF NOT EXISTS fact_embeddings (
    fact_id    INTEGER PRIMARY KEY REFERENCES facts(fact_id),
    model      TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Trust adjustment constants
_HELPFUL_DELTA   =  0.05
_UNHELPFUL_DELTA = -0.10
_TRUST_MIN       =  0.0
_TRUST_MAX       =  1.0

# --- automatic supersession demotion -------------------------------------
# Ranking on the reranked path is `cross_encoder_score * (0.5 + 0.5*trust) *
# temporal_decay` (retrieval.py; the trust clamp and the decay term both landed
# 2026-08-29). The cross-encoder saturates — measured 0.9997-0.9988 across eight
# on-topic facts, a 0.0009 spread — so trust and age are still what separate
# facts inside one topic, and the clamp deliberately leaves a demoted 0.30 row
# ~13% below a current 0.50 one, far more than the ce spread it could win back.
# A fact that a later fact explicitly retracts therefore keeps outranking its
# own correction forever unless something demotes it. Found 2026-08-28 on a
# store where the most-revised topics were the worst affected: one query
# returned four retracted answers above both current ones, another returned
# first a verdict a later fact calls "INVERTED". Writing
# the retraction was never enough — nothing demoted what it retracted.
#
# ACTIVE VOICE ONLY. `superseded by fid N` names the NEWER fact, so matching it
# would demote exactly the wrong row; the passive form is deliberately absent.
# The `target < fact_id` guard in _demote_superseded is the backstop.
_SUPERSESSION_RE = re.compile(
    r"\b(?:supersede|supersedes|superseding|corrects?|correcting|"
    r"refutes?|invalidates?|obsoletes?)\s+fid[\s=:#]*(\d{1,7})\b",
    re.IGNORECASE,
)
# Floor, not a fixed delta: a fact may be retracted by several later facts, and
# three mentions should not drive a row to zero. 0.30 keeps it at the default
# search floor — still reachable, ranked below anything that corrected it.
_SUPERSESSION_FLOOR = 0.30

# --- fact validity windows (see invalidate_fact) --------------------------
# Demotion answers "which of these two do I rank first". It cannot answer "was
# this true on August 12", because a demoted fact is only ranked lower, never
# marked invalid, and the retraction itself lives in free-text prose that
# nothing but a human reader parses. `valid_from`/`valid_until` make the
# supersession relation a QUERYABLE interval instead: every fact is born valid
# at its own created_at, and the fact that retracts it closes the window at the
# RETRACTING fact's created_at, so successive versions of one claim tile the
# timeline as half-open intervals [valid_from, valid_until) that abut exactly
# and never overlap. Reading fid 1281 -> 1283 -> 1305 (three generations of one
# verdict on this store) now yields three adjacent windows rather than three
# rows at three trust scores.
#
# Deliberately NOT a second ranking signal. The read path is unchanged: search
# still orders by relevance * trust * decay, which already handles recency, and
# an expired fact stays reachable — it is the lineage of an answer, and hiding
# it would repeat the mistake the trust floor exists to avoid. The window is
# for the questions ranking cannot answer, and for the digest's consistency
# check (an expired fact still sitting above the search floor is a demotion
# that did not take).
#
# `valid_until` is closed by a fact id, never by a bare timestamp (see
# invalidate_fact): a window that closed for no nameable reason is exactly the
# unfalsifiable state this column exists to replace.

# --- near-duplicate guard (see _near_duplicate) ---------------------------
# Token-Jaccard at or above this against a same-category fact written in the
# last _NEAR_DUP_WINDOW_DAYS returns that row instead of inserting a second
# copy. 0.75 is set from the store, not from taste: replaying the guard over
# all 880 facts on 2026-08-28 fires on exactly ONE historical pair (867/1113 at
# 0.831, the same research question answered twice nine days apart) and the
# next in-scope pair down is 0.574. The empty band between them is where the
# threshold belongs — a false positive silently discards a real write, which is
# worse than the duplicate it would have prevented.
_NEAR_DUP_JACCARD = 0.75
_NEAR_DUP_WINDOW_DAYS = 30
# Below this many distinct tokens one word is worth more than 5% of the score,
# so Jaccard stops resolving. Only 10 of 880 stored facts are this short (p05
# is 32 tokens, median 95); they forgo the guard rather than risk it.
_NEAR_DUP_MIN_TOKENS = 20

# --- cross-job completion markers (see _reconcile_markers) ----------------
# `tags` is written with `tags = ?` — a FULL SQL REPLACE. That is correct for
# descriptive tags (a fact's subjects change when its content does), and it is
# catastrophic for the handful of tags that are not descriptions at all but
# CROSS-JOB COMPLETION STATE: one cron job writes the tag, a DIFFERENT job (or
# its prerun digest script) reads it to decide whether to act at all. Twenty-one
# nightly jobs share one tags column, four of them retype the whole string from
# memory, and three have provably destroyed another job's marker:
#
#   fid 919  lost `promote-candidate` AND `promoted` (daily-trace-mining,
#            2026-08-26 and 2026-08-27 — visible as two successive shrinking
#            tag strings in fact_history)
#   fid 685  lost `promote-candidate` the same way
#   fid 1268 lost `deep-dived` fifteen hours after topic-deep-dive wrote it,
#            to research-open-questions' answer rewrite
#            (`dream,unverified,…,deep-dived` -> `researched,answered,…,verified`)
#
# After three nightly dives only 2 facts store-wide still carried `deep-dived`.
# Each incident was patched in the losing job's PROMPT ("call action=get first
# and copy the tags"); prompt discipline is not an invariant, and at 21 jobs it
# will not hold. fact_markers is the invariant: markers live in their own table,
# and update_fact RE-APPLIES any marker the caller's string omitted, so a full
# replace becomes safe without changing a single caller.
#
# MEMBERSHIP TEST — a tag belongs here only if some job READS it to gate an
# action, not merely to describe. Verified against the live store 2026-08-30 by
# grepping every prompt in cron/jobs.json and every prerun script in scripts/:
#
#   promote-candidate  daily-review / weekly-trace-mining / retrieval-audit set
#                      it; dream-and-promote selects on it.
#   promoted           dream-and-promote sets it, and reads it on later nights
#                      (and in its 2026-08-30 reconcile step) to avoid
#                      re-promoting the same source fact.
#   deep-dived         topic-deep-dive sets it on its synthesis fact;
#                      scripts/deep-dive-topic.py DONE_TAGS excludes it.
#   deep-dive          the same script's OTHER done tag — the variant written on
#                      the SOURCE fact. Protecting only `deep-dived` would leave
#                      the exclusion half-working, so both are markers.
#   designed           experiment-design sets it; experiment-design-queue.py
#                      TERMINAL and deep-dive-topic.py TERMINAL_TAGS read it.
#   retired-experiment experiment-design sets it and its own prompt calls it
#                      PERMANENT; three prerun scripts read it.
#   experiment-run     declared in TERMINAL/TERMINAL_TAGS by both prerun
#                      scripts. Zero rows carry it today; it is listed so the
#                      FIRST write of it is protected rather than the second.
#   needs-experiment   research-open-questions sets it; three prerun scripts
#                      build their queues from it.
#   blocked-local      research-open-questions sets it; morning-briefing drops
#                      it from "actionable", experiment-design-queue.py skips it.
#   deep-review-sent   deep-review-prep sets it; deep-review-candidates.py
#                      SENT_TAG reads it "so they are never re-proposed".
#   deep-review-filed  the ingest half of that pair (deep-review-file.py).
#   retrieval-fail     retrieval-audit sets it, and its digest re-lists on it a
#                      week later — retrieval_count cannot hold that state
#                      because any read bumps the counter.
#
# DELIBERATELY NOT MARKERS. `research-queue`, `answered`, `verified`,
# `unverified`, `confirmed`, `partially-confirmed`, `researched`, `synthesized`,
# `open-question`, `memory-entry`, `highlight` are verdict or descriptive labels: they
# describe what a fact IS, they are rewritten as part of a legitimate state
# transition, and making them sticky would fight curation instead of protecting
# it. `attempted-once`/`stalled` are the closest call — research-open-questions
# does read them to pick its escalation path — but they are single-job aging
# counters with 1 and 0 live rows; they are left out until a loss is observed,
# and adding them is a one-line change here.
CROSS_JOB_MARKERS = frozenset({
    "promote-candidate",
    "promoted",
    "deep-dived",
    "deep-dive",
    "designed",
    "retired-experiment",
    "experiment-run",
    "needs-experiment",
    "blocked-local",
    "deep-review-sent",
    "deep-review-filed",
    "retrieval-fail",
})


def _split_tags(raw: "str | None") -> list[str]:
    """Split a tags field into stripped, non-empty tags, order preserved.

    Whitespace only — no other normalisation. Five rows in the live store hold
    JSON-array-shaped tags (`["paper", "GRACE", …]`, fids 816/817) and five use
    `", "` separators; neither shape contains a marker, and rewriting them here
    would silently reformat corpus rows on an unrelated write. Duplicates are
    NOT collapsed either: fid 685 carries `promote-candidate` twice after a
    manual restore, and deduplicating it as a side effect of some other job's
    update is exactly the kind of uninvited edit this module exists to stop.
    """
    return [t for t in (part.strip() for part in (raw or "").split(",")) if t]


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
        if "valid_from" not in columns:
            # No DEFAULT clause here, unlike the fresh-schema column above:
            # SQLite rejects a non-constant default on ADD COLUMN ("Cannot add
            # a column with non-constant default"), so CURRENT_TIMESTAMP is not
            # available on this path. created_at is the right backfill value
            # anyway — a fact was valid from the moment it was written, and
            # inventing "now" for existing rows would date every one of them to
            # whichever process happened to open the store first.
            self._conn.execute("ALTER TABLE facts ADD COLUMN valid_from TIMESTAMP")
        if "valid_until" not in columns:
            # NULL is the load-bearing value here (= still valid), so there is
            # nothing to backfill: invalidate_fact closes a window only when
            # given the fact that closed it, and for retractions written before
            # this column existed that pairing has to be recovered by an
            # operator pass over the corpus, not guessed at open time.
            self._conn.execute("ALTER TABLE facts ADD COLUMN valid_until TIMESTAMP")
        # UNCONDITIONALLY, not only when the column was just added. Adding a
        # column does not restart the processes already using this store, and
        # they keep INSERTing through the module they imported at startup —
        # which has no valid_from in its INSERT. Observed on 2026-09-02: the
        # migration ran at 00:17 and three facts written at 00:48 through an
        # MCP bridge started at 00:04 came back with valid_from NULL on an
        # already-migrated store. The gateway that runs the nightly cron is the
        # same shape of long-lived process, so "migrated once" is not the same
        # as "the invariant holds". This makes NOT NULL an invariant the store
        # re-establishes every time it is opened, rather than a property of one
        # lucky moment.
        self._backfill_valid_from()
        hist_columns = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(fact_history)").fetchall()
        }
        if hist_columns and "changed_by_session" not in hist_columns:
            self._conn.execute(
                "ALTER TABLE fact_history ADD COLUMN changed_by_session TEXT DEFAULT ''"
            )
        self._conn.commit()

    def _backfill_valid_from(self) -> None:
        """Stamp valid_from = created_at on every row that is missing it.

        Two sources of NULL: rows written before the column existed, and rows
        written after it existed by a process still holding the pre-column
        module (see the call site). Both mean the same thing — "valid from when
        it was created" — so both take the same repair.

        Returns immediately when nothing is NULL, which is the normal case on
        every open after the first. That probe is a short-circuiting scan with
        no index behind it; on a 939-row store it is microseconds, and paying it
        once per process start is what buys the invariant.

        This is the first migration that has had to WRITE every existing row,
        and facts_fts is an external-content FTS5 index: the facts_au trigger
        issues a `('delete', rowid, content, tags)` command for each row an
        UPDATE touches, and FTS5 raises "database disk image is malformed" when
        asked to delete an entry the index never held. Any row written before
        facts_fts existed is exactly such a row — the pre-FTS legacy shape in
        tests/plugins/memory/test_holographic_history.py is one — so the obvious
        one-line UPDATE fails on precisely the oldest stores.

        The index was already broken in that case: every future UPDATE to those
        rows would have raised the same error, and they were invisible to
        search. Rebuilding it is therefore a repair, not a side effect. It runs
        only after the direct path has actually failed — measured on a copy of
        the live 939-row store the UPDATE succeeds in 0.15 s and no rebuild
        happens.
        """
        probe = self._conn.execute(
            "SELECT 1 FROM facts WHERE valid_from IS NULL LIMIT 1"
        ).fetchone()
        if probe is None:
            return
        try:
            self._conn.execute(
                "UPDATE facts SET valid_from = created_at WHERE valid_from IS NULL"
            )
        except sqlite3.DatabaseError:
            logger.warning(
                "facts_fts holds no entry for at least one row of facts; "
                "rebuilding the index so the valid_from backfill can proceed"
            )
            self._conn.execute("INSERT INTO facts_fts(facts_fts) VALUES('rebuild')")
            self._conn.execute(
                "UPDATE facts SET valid_from = created_at WHERE valid_from IS NULL"
            )

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

        Deduplicates by content (UNIQUE constraint) and, since 2026-08-29, by
        token-Jaccard against recent same-category facts (see
        _near_duplicate). On either kind of duplicate, returns the existing
        fact_id without modifying the row. Extracts entities from the content
        and links them to the fact.

        source_session records which session wrote the fact, joinable against
        state.db / the trace archive — the provenance that made the 2026-08-14
        deletion recovery possible was reconstructed by hand; this makes it a
        column.
        """
        with self._lock:
            content = content.strip()
            if not content:
                raise ValueError("content must not be empty")

            # Near-duplicate: return the standing row, insert nothing. Placed
            # before the INSERT so it shares the exact-duplicate branch's
            # contract — including returning before _demote_superseded(), so a
            # restatement cannot demote the same target a second time.
            near = self._near_duplicate(content, category)
            if near is not None:
                return near

            try:
                cur = self._conn.execute(
                    """
                    INSERT INTO facts (content, category, tags, trust_score,
                                       source_session, valid_from)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
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

            # Markers a fact is born with — topic-deep-dive writes
            # tags="deep-dived,…" on the synthesis fact at ADD time, so
            # registering only on update would leave its very first marker
            # unprotected until something happened to touch the row.
            self._record_markers(fact_id, tags, source_session)

            # Compute HRR vector after entity linking
            self._compute_hrr_vector(fact_id, content)
            self._rebuild_bank(category)

            self._demote_superseded(fact_id, content)

        # OUTSIDE the lock, and outside it deliberately. self._lock is the
        # PROCESS-WIDE write lock every MemoryStore instance shares (see the
        # registry comment on the class) — the main agent plus every
        # delegate_task subagent. An HTTP call inside it stalls every other
        # memory write in the gateway for as long as llama-swap takes, which on
        # a cold load of jina-embed is seconds, not milliseconds.
        #
        # Placed after the lock rather than inside also means the two dedupe
        # returns above (near-duplicate at the top, UNIQUE(content) in the
        # except) skip it for free: a restatement never pays an embedding call.
        # The row is already durable — the connection is in autocommit, so the
        # INSERT committed the instant it ran and nothing here can un-write it.
        self._embed_fact(fact_id, content)
        return fact_id

    def _near_duplicate(self, content: str, category: str) -> int | None:
        """Return the fact_id this content merely restates, or None.

        UNIQUE(content) catches only byte-identical writes, so a promotion, a
        reworded prefix, or the same conclusion re-derived a week later inserted
        a competing row: measured over 871 facts on 2026-08-28, 7 pairs at
        Jaccard >= 0.55 and two effectively identical (fids 89/468 at 1.00,
        85/878 at 0.98; 919/1174 differ only by "LESSON:" vs
        "KEY PATTERN/LESSON:"). Copies split trust and retrieval_count between
        them and crowd the limit-5 prefetch window with one finding.

        Scoped to the same category and the last 30 days: the widest lane on
        this store is 234 rows (lesson), measured at 7.2 ms per call against a
        write path that already re-encodes an HRR vector and rebuilds a category
        bank. The scope also means a deliberate cross-lane restatement — a
        lesson promoted to a memory-entry, which the MEMORY.md renderer depends
        on — is correctly NOT a duplicate.

        POLARITY, AND WHY SUPERSESSION WRITES ARE EXEMPT: Jaccard cannot see
        polarity, so a long fact that corrects an earlier one by a single word
        ("does" -> "does not") scores ~0.98 and looks like a restatement. That
        is the highest-value write in the store, not a duplicate. Any content
        naming the fid it retracts is therefore skipped outright — measured on a
        real pair, the correction scored 0.861, was dropped, AND the target's
        trust stayed at 0.50 instead of 0.30, silently defeating
        _demote_superseded (fork da2f371005), which is the load-bearing half of
        the supersession fix. Losing the correction and the demotion together is
        strictly worse than keeping one near-duplicate row.

        Never raises: a guard that loses fact writes is worse than the
        duplicates it exists to stop.
        """
        # Before any similarity work: a write that names the fid it supersedes
        # is by definition intended to change state, however much of the earlier
        # wording it reuses.
        if _SUPERSESSION_RE.search(content):
            return None
        try:
            # Lazy, and via the retriever, so the guard and the Jaccard term in
            # ranked search can never disagree about what a token is (same
            # reason search_facts imports it here rather than duplicating the
            # sanitizer).
            from plugins.memory.holographic.retrieval import FactRetriever

            tokens = FactRetriever._tokenize(content)
            if len(tokens) < _NEAR_DUP_MIN_TOKENS:
                return None
            rows = self._conn.execute(
                """
                SELECT fact_id, content FROM facts
                 WHERE category = ?
                   AND created_at >= datetime('now', ?)
                """,
                (category, f"-{_NEAR_DUP_WINDOW_DAYS} days"),
            ).fetchall()
            best_id: int | None = None
            best = 0.0
            for row in rows:
                sim = FactRetriever._jaccard_similarity(
                    tokens, FactRetriever._tokenize(row["content"])
                )
                if sim > best:
                    best_id, best = int(row["fact_id"]), sim
            if best_id is not None and best >= _NEAR_DUP_JACCARD:
                # WARNING, not debug: this drops a write on the floor. It has to
                # be greppable in agent.log when a session insists it stored
                # something the store does not have.
                logger.warning(
                    "near-duplicate write suppressed: Jaccard %.3f against "
                    "fid %s (category=%s) — returning the existing row, "
                    "nothing inserted",
                    best, best_id, category,
                )
                return best_id
        except Exception:
            logger.debug("near-duplicate check failed", exc_info=True)
        return None

    def _demote_superseded(self, fact_id: int, content: str) -> list[int]:
        """Demote any strictly-older fact this one explicitly retracts.

        Called only on the genuine-insert path — the duplicate-content branch
        returns before this, so re-writing the same fact cannot demote its
        target twice.

        Deliberately here rather than in the scheduler's per-run ledger: the
        ledger sees only what the nightly cron writes, while every writer on the
        box — cron, an interactive session, and scripts/hermes_memory_mcp.py's
        ``remember`` (which imports this class directly and loads no Hermes
        plugins) — goes through add_fact. One place covers all three.

        Never raises. A missed demotion leaves a stale row ranked high, which is
        the bug this fixes; an exception escaping here would lose the fact write
        itself, which is strictly worse. Returns the fids actually demoted.
        """
        demoted: list[int] = []
        try:
            targets = {int(m) for m in _SUPERSESSION_RE.findall(content)}
        except Exception:                                    # pragma: no cover
            return demoted
        for target in sorted(targets):
            # Strictly older only. A fact cannot retract one written after it,
            # so a forward reference is a parse artefact (or the passive voice
            # slipping through) and must never cost the newer row its trust.
            if not 0 < target < fact_id:
                continue
            # Close the retracted fact's validity window BEFORE the trust loop,
            # and OUTSIDE it. The loop breaks on its first iteration whenever
            # the target already sits at the floor — the normal state for a
            # fact retracted twice — so a stamp placed inside it would be
            # skipped for exactly the rows with the most interesting history.
            # Trust and validity are answering different questions here and
            # must not share a control flow.
            try:
                self.invalidate_fact(target, superseded_by=fact_id)
            except Exception:
                logger.warning(
                    "validity-window close failed for fid %s (retracted by %s)",
                    target, fact_id, exc_info=True,
                )
            try:
                # Bounded: at most two steps of _UNHELPFUL_DELTA per retraction,
                # so one write can never sink a row further than 0.50 -> 0.30.
                for _ in range(2):
                    row = self._conn.execute(
                        "SELECT trust_score FROM facts WHERE fact_id = ?", (target,)
                    ).fetchone()
                    # Epsilon, not a bare <=. Repeated -0.10 steps land on
                    # 0.30000000000000004, which compares GREATER than 0.30, so
                    # a bare test lets the next retraction punch through the
                    # floor to 0.20 and hide the row from default search.
                    if row is None or row["trust_score"] <= _SUPERSESSION_FLOOR + 1e-9:
                        break
                    self.record_feedback(
                        target, helpful=False,
                        changed_by=f"auto-supersession: retracted by fid {fact_id}",
                    )
                    demoted.append(target)
            except Exception:
                logger.warning(
                    "supersession demotion failed for fid %s (retracted by %s)",
                    target, fact_id, exc_info=True,
                )
        return demoted

    def invalidate_fact(self, fact_id: int, superseded_by: int) -> bool:
        """Close *fact_id*'s validity window at the moment *superseded_by* was written.

        Returns True only when this call actually moved the column — a window
        already closed, an unknown fact on either side, or a superseding id
        that is not strictly newer all return False without writing.

        FIRST CLOSE WINS. `valid_until` records when a claim STOPPED being the
        current answer, and that is the first retraction, not the last: a fact
        retracted again a month later did not become invalid twice. The
        `valid_until IS NULL` predicate is the whole idempotence story, which is
        also what makes the backfill script safe to re-run.

        The close time is read from the superseding fact's own created_at rather
        than from the clock, so the live path and the historical backfill
        produce identical values and the two windows abut exactly:
        older.valid_until == newer.valid_from (they are half-open, so the
        endpoints touching is correct and not an overlap). Calling with a bare
        timestamp is deliberately impossible — see the validity-window comment
        at the top of this module.

        No fact_history snapshot: the snapshot columns are content/category/
        tags/trust_score, none of which move here, so the row it wrote would
        assert a change it cannot show. The audit trail for a supersession
        close is the two `feedback` snapshots _demote_superseded writes, each
        stamped "retracted by fid N", plus the retracting fact's own prose.
        """
        if not 0 < fact_id < superseded_by:
            return False
        with self._lock:
            cur = self._conn.execute(
                """
                UPDATE facts
                   SET valid_until = (SELECT created_at FROM facts WHERE fact_id = :newer)
                 WHERE fact_id = :older
                   AND valid_until IS NULL
                   AND EXISTS (SELECT 1 FROM facts WHERE fact_id = :newer)
                """,
                {"older": fact_id, "newer": superseded_by},
            )
            self._conn.commit()
            if cur.rowcount:
                logger.info(
                    "fid %s validity window closed by fid %s", fact_id, superseded_by
                )
            return bool(cur.rowcount)

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

            embed_content: "str | None" = None
            assignments: list[str] = ["updated_at = CURRENT_TIMESTAMP"]
            params: list = []

            # A tags write is a full replace; re-apply this fact's markers
            # before it lands, and record any the caller introduced. Returns
            # the caller's own string byte-for-byte when there is nothing to
            # restore, so a fact with no markers is unaffected.
            marker_inserts: list[tuple[int, str, str]] = []
            if tags is not None:
                tags = self._reconcile_markers(fact_id, tags, changed_by, marker_inserts)

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
                # Same transaction as the UPDATE: a marker recorded for a write
                # that then failed (UNIQUE(content), say) would claim state the
                # rendered tags do not show — the two must not diverge.
                for row_args in marker_inserts:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO fact_markers (fact_id, marker, set_by)"
                        " VALUES (?, ?, ?)",
                        row_args,
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
                # The dense vector encodes CONTENT only, so a tags-only edit
                # leaves it correct — unlike the HRR vector above, which
                # re-derives from entity links that tags CAN change. Deferred
                # to after the lock for the same reason as in add_fact; read
                # back from the row, not from the argument, so it matches what
                # was actually stored.
                if content is not None:
                    embed_content = row2["content"]
            # Rebuild the destination bank — and the SOURCE bank too when the
            # fact changed category, or the departed bank goes on counting it.
            # remove_fact already rebuilds the category a fact leaves; a
            # migration is that same departure with an arrival attached.
            old_cat = row["category"]
            cat = category or old_cat
            self._rebuild_bank(cat)
            if cat != old_cat:
                self._rebuild_bank(old_cat)

        # Outside the lock — see the same call at the end of add_fact.
        if embed_content is not None:
            self._embed_fact(fact_id, embed_content)
        return True

    # ------------------------------------------------------------------
    # Cross-job completion markers
    # ------------------------------------------------------------------
    # fact_markers is the source of truth; facts.tags is a RENDERED VIEW of it.
    # Keeping the rendering means every existing reader — the four prerun
    # digest scripts that match `tags LIKE '%needs-experiment%'`, the FTS index,
    # every prompt that tells a model to read a fact's tags — keeps working
    # unchanged. See the CROSS_JOB_MARKERS comment for why this table exists.

    def _markers_of(self, fact_id: int) -> set[str]:
        rows = self._conn.execute(
            "SELECT marker FROM fact_markers WHERE fact_id = ?", (fact_id,)
        ).fetchall()
        return {str(r["marker"]) for r in rows}

    def _record_markers(self, fact_id: int, tags: str, set_by: str = "") -> None:
        """Register any CROSS_JOB_MARKERS present in *tags*.

        The write path stays the tags string, deliberately: every cron prompt
        already says `tags="<existing>,promoted"`, and requiring them all to
        learn a new call is the same prompt-discipline bet that failed three
        times. A marker the caller ADDS is simply recorded.

        Never raises. Losing the fact write to a bookkeeping failure would be
        strictly worse than the missed marker (same contract as
        _demote_superseded).
        """
        try:
            for tag in _split_tags(tags):
                if tag in CROSS_JOB_MARKERS:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO fact_markers (fact_id, marker, set_by)"
                        " VALUES (?, ?, ?)",
                        (fact_id, tag, set_by or ""),
                    )
        except Exception:                                    # pragma: no cover
            logger.warning("marker registration failed for fid %s", fact_id,
                           exc_info=True)

    def _reconcile_markers(
        self,
        fact_id: int,
        tags: str,
        changed_by: str,
        out_inserts: list,
    ) -> str:
        """Return *tags* with this fact's missing markers re-appended.

        THE PROTECTION. A job that retypes the tag string from memory, or
        rewrites it wholesale as part of a state transition, cannot destroy
        another job's completion marker: whatever it omits is put back.

        RE-APPLY IS UNCONDITIONAL — it does not try to tell a "retype" from a
        "curation". That heuristic was considered and rejected on the evidence:
        fid 1268's loss came from research-open-questions turning an answered
        question's tags from `dream,unverified,…` into `researched,answered,…`,
        which is a genuine, correct curation by every signal a heuristic could
        read — and it still had no business dropping topic-deep-dive's
        `deep-dived`. A guess that is wrong loses the marker silently, which is
        the bug. So the rule is flat: the tags string cannot retire a marker.

        HOW A MARKER IS RETIRED: clear_marker(). It deletes the fact_markers row
        FIRST and only then rewrites tags, so there is nothing left for this
        method to re-apply — removal is always reachable, in one call, in any
        order. (`promoted` after an incumbent review demotes the MEMORY.md
        entry, and `deep-dived` when a topic is re-opened, are the real cases.)

        Returns the caller's string UNCHANGED — byte for byte, no reordering,
        no dedup, no whitespace normalisation — whenever nothing is missing.
        A store with an empty fact_markers therefore behaves exactly as it did
        before this table existed, which is also the pre-backfill state.

        Never raises: on any failure the caller's string is used as-is, i.e.
        today's behaviour. A tags write must not be lost to this.
        """
        try:
            present = {t for t in _split_tags(tags) if t in CROSS_JOB_MARKERS}
            stored = self._markers_of(fact_id)

            for marker in sorted(present - stored):
                out_inserts.append((fact_id, marker, changed_by or ""))

            missing = sorted(stored - present)
            if not missing:
                return tags

            # WARNING, not debug: this is a job trying to delete state it does
            # not own. It has to be greppable in agent.log — a marker silently
            # restored every night means a prompt is still wrong even though
            # the damage is now contained.
            logger.warning(
                "fact_markers: re-applied %s to fid %s — the tags write from "
                "%r omitted them (full-replace protection)",
                ",".join(missing), fact_id, changed_by or "<unknown session>",
            )
            base = tags.rstrip()
            if base.endswith(","):
                base = base[:-1].rstrip()
            return (base + "," if base else "") + ",".join(missing)
        except Exception:                                    # pragma: no cover
            logger.warning("marker reconciliation failed for fid %s", fact_id,
                           exc_info=True)
            return tags

    def get_markers(self, fact_id: int) -> list[str]:
        """Every cross-job marker this fact carries, sorted. [] if none."""
        with self._lock:
            return sorted(self._markers_of(fact_id))

    def set_marker(self, fact_id: int, marker: str, set_by: str = "") -> bool:
        """Record *marker* on a fact and render it into the fact's tags.

        The explicit form of what `tags="<existing>,promoted"` already does —
        for new callers, and for anything that has no reason to hold the whole
        tag string. Returns False if the fact does not exist.

        A marker outside CROSS_JOB_MARKERS is refused rather than quietly
        stored: an unrecognised name would be recorded here, re-applied
        forever, and read by nothing.
        """
        marker = (marker or "").strip()
        if marker not in CROSS_JOB_MARKERS:
            raise ValueError(
                f"{marker!r} is not a cross-job marker; add it to "
                f"CROSS_JOB_MARKERS (with the job that reads it) first"
            )
        with self._lock:
            row = self._conn.execute(
                "SELECT tags FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                return False
            self._conn.execute(
                "INSERT OR IGNORE INTO fact_markers (fact_id, marker, set_by)"
                " VALUES (?, ?, ?)",
                (fact_id, marker, set_by or ""),
            )
            self._conn.commit()
            # Render. Routed through update_fact so the tags change is
            # snapshotted into fact_history and the tag-derived entity links
            # are refreshed, exactly as a caller-written tags string would be.
            current = row["tags"] or ""
            if marker not in _split_tags(current):
                joined = f"{current},{marker}" if current.strip() else marker
                self.update_fact(
                    fact_id, tags=joined,
                    changed_by=set_by or f"set_marker:{marker}",
                )
            return True

    def clear_marker(self, fact_id: int, marker: str, changed_by: str = "") -> bool:
        """Retire *marker*: delete the row, then strip it from the tags.

        The ONLY way a marker comes off, since _reconcile_markers restores
        anything a tags write omits. Order is load-bearing — the fact_markers
        row goes first, so the rewrite below finds nothing to re-apply and
        removal cannot deadlock against the protection. Doing it the other way
        round (rewrite, then delete) would also work; doing only the rewrite
        would not, which is precisely the point.

        Returns True if the fact carried the marker in either place.
        """
        marker = (marker or "").strip()
        with self._lock:
            row = self._conn.execute(
                "SELECT tags FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                return False
            cur = self._conn.execute(
                "DELETE FROM fact_markers WHERE fact_id = ? AND marker = ?",
                (fact_id, marker),
            )
            self._conn.commit()
            had_row = cur.rowcount > 0

            parts = _split_tags(row["tags"])
            if marker in parts:
                self.update_fact(
                    fact_id, tags=",".join(t for t in parts if t != marker),
                    changed_by=changed_by or f"clear_marker:{marker}",
                )
                return True
            return had_row

    def facts_with_marker(self, marker: str, limit: int = 200) -> list[dict]:
        """Facts carrying *marker*, oldest first, from fact_markers.

        Reads the source-of-truth table, not `tags LIKE '%…%'`: a substring
        match over a shared tag vocabulary is wrong in both directions — it
        cannot tell `deep-dive` from `deep-dived`, and `needs-experiment` from
        `retired-experiment` (scripts/deep-dive-topic.py carries that scar in a
        comment). Returns [] before the backfill has run; the prerun scripts
        keep using tags until they are migrated, which is why the rendering
        exists.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score,"
                "       f.retrieval_count, f.helpful_count, f.created_at,"
                "       f.updated_at, m.set_by, m.set_at"
                "  FROM fact_markers m JOIN facts f USING (fact_id)"
                " WHERE m.marker = ?"
                " ORDER BY f.fact_id ASC LIMIT ?",
                (marker, limit),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_fact(self, fact_id: int) -> dict | None:
        """Fetch one fact by id, or None if it does not exist.

        Exact lookup — no ranking, no trust floor, no retrieval-count bump.
        This is the verification path: ranked search cannot prove a fact is
        absent (a low-ranked or sub-min_trust row simply doesn't surface).
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, content, category, tags, trust_score,"
                " retrieval_count, helpful_count, created_at, updated_at,"
                " valid_from, valid_until"
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
                # Markers go with the row. fact_id is AUTOINCREMENT so an id is
                # never reused and orphans could not be mis-attributed, but a
                # marker outliving its fact would still be counted by
                # facts_with_marker(); the snapshot in fact_history carries the
                # tags string, so this is not the last copy.
                self._conn.execute(
                    "DELETE FROM fact_markers WHERE fact_id = ?", (fact_id,)
                )
                # Same reasoning as fact_markers: fact_id is AUTOINCREMENT so a
                # surviving vector could never be mis-attributed to a later
                # fact, but it would go on being scored by the dense scan and
                # returning a fact_id that JOINs to nothing.
                self._conn.execute(
                    "DELETE FROM fact_embeddings WHERE fact_id = ?", (fact_id,)
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
                       retrieval_count, helpful_count, created_at, updated_at,
                       valid_from, valid_until
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

    # ------------------------------------------------------------------
    # Dense embedding lane
    # ------------------------------------------------------------------
    # ACTIVATION IS THE DATA, NOT A FLAG. Every method below is inert while
    # fact_embeddings is empty, and populating it (scripts/backfill-fact-
    # embeddings.py) is what switches the lane on. That is deliberate:
    #
    #   * a fresh checkout, and every test that builds a MemoryStore over a
    #     tmp_path, has an empty table and therefore makes NO network call —
    #     this class had never issued one before 2026-09-04 and a config-flag
    #     design would have made "does add_fact touch the network?" depend on
    #     which config file the test process happened to find;
    #   * turning the lane on is a deliberate, auditable act (a backfill run)
    #     rather than a line in a YAML file that `hermes config migrate` is
    #     known to rewrite;
    #   * turning it OFF is `DELETE FROM fact_embeddings`, which is also
    #     exactly what you would do to force a re-embed after a model swap.

    def _embeddings_active(self) -> bool:
        """Is the dense lane switched on for this store?

        The True result is cached forever — a lane cannot un-activate itself,
        and re-probing on the hot write path for a fact we are about to embed
        anyway is wasted work. A False result is NOT cached: the backfill that
        activates the lane runs in a different process, and a long-lived
        gateway must notice it without a restart.
        """
        # Both caches live on the shared registry entry, not on self: one
        # gateway process holds a MemoryStore per provider (the main agent plus
        # every delegate_task subagent) and they all point at one connection.
        # Per-instance caches would mean N copies of the same 4 MB matrix and N
        # independent activation probes.
        if self._entry.get("embed_on"):
            return True
        if not embeddings.available():
            return False
        try:
            row = self._conn.execute(
                "SELECT 1 FROM fact_embeddings LIMIT 1"
            ).fetchone()
        except sqlite3.DatabaseError:
            return False
        if row is None:
            return False
        self._entry["embed_on"] = True
        return True

    def _embed_fact(self, fact_id: int, content: str) -> None:
        """Write-path hook. Never raises, never blocks a fact from existing.

        An embedding failure leaves the row absent, which is precisely the
        state the nightly heal job looks for — the same self-heal shape as
        hermes-numpy-ensure. The alternative, failing the write, would let a
        cold llama-swap load cost the metabolism a fact.
        """
        if not self._embeddings_active():
            return
        try:
            self.ensure_embedding(fact_id, content=content, force=True)
        except Exception:
            logger.debug("embedding write failed for fact %s", fact_id, exc_info=True)

    def ensure_embedding(
        self,
        fact_id: int,
        content: "str | None" = None,
        force: bool = False,
    ) -> bool:
        """Compute and store this fact's dense vector. True if a row was written.

        This is the ONLY sanctioned way to populate fact_embeddings — the same
        rule that keeps raw SQL out of the fact-write path, for the same reason:
        a hand-written INSERT would skip the normalisation in
        embeddings.to_blob() and produce vectors that the cosine scan silently
        mis-ranks instead of rejecting.

        With *force* false, a row that already exists for the CURRENT model and
        dim is left alone, so a backfill can be re-run over a partly populated
        store without re-embedding what is already correct.
        """
        if not embeddings.available():
            return False
        model = embeddings.resolve_model()
        with self._lock:
            row = self._conn.execute(
                "SELECT content FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                return False
            text = content if content is not None else row["content"]
            if not force:
                existing = self._conn.execute(
                    "SELECT model, dim FROM fact_embeddings WHERE fact_id = ?",
                    (fact_id,),
                ).fetchone()
                if existing is not None and existing["model"] == model:
                    return False

        # OUTSIDE the lock. This is an HTTP call that can take seconds against a
        # cold llama-swap, and the lock it would otherwise hold is the process-
        # wide write lock every other provider in the gateway shares.
        vector = embeddings.embed_one(text, model=model)
        if not vector:
            return False
        blob = embeddings.to_blob(vector)

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO fact_embeddings (fact_id, model, dim, vector, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(fact_id) DO UPDATE SET
                    model = excluded.model,
                    dim = excluded.dim,
                    vector = excluded.vector,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (fact_id, model, len(vector), blob),
            )
            self._conn.commit()
            self._entry["embed_on"] = True
        return True

    def missing_embeddings(self, limit: int = 0) -> "list[tuple[int, str]]":
        """(fact_id, content) for facts with no vector for the CURRENT model.

        Rows embedded by a DIFFERENT model count as missing: they are in another
        vector space, so scoring them against this model's query vector is worse
        than not scoring them at all. That makes a model swap self-healing —
        change HERMES_EMBED_MODEL and the nightly heal job re-embeds the corpus.
        """
        model = embeddings.resolve_model()
        sql = """
            SELECT f.fact_id, f.content
            FROM facts f
            LEFT JOIN fact_embeddings e ON e.fact_id = f.fact_id
            WHERE e.fact_id IS NULL OR e.model != ?
            ORDER BY f.fact_id
        """
        params: list = [model]
        if limit and limit > 0:
            sql += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [(int(r["fact_id"]), r["content"]) for r in rows]

    def prune_orphan_embeddings(self) -> int:
        """Delete vectors whose fact no longer exists. Returns the count.

        remove_fact deletes the vector in the same transaction as the row, so
        this store never creates an orphan itself. A process running code that
        predates the table does: the gateway that was up when the migration
        landed keeps deleting facts through the module it imported at startup,
        which knows nothing about fact_embeddings. The dense scan already drops
        an id it cannot resolve (retrieval._dense_candidates), so an orphan
        costs a wasted slot in the top-k rather than a wrong answer — this is
        hygiene for the nightly heal, not a correctness fix.
        """
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM fact_embeddings WHERE fact_id NOT IN"
                " (SELECT fact_id FROM facts)"
            )
            self._conn.commit()
            return int(cur.rowcount or 0)

    def backfill_embeddings(
        self,
        limit: int = 0,
        batch: int = 0,
        progress=None,
    ) -> dict:
        """Embed every fact that has no current vector. Returns a summary dict.

        Batched, unlike ensure_embedding: measured against the live llama-swap
        entry, 32 inputs per request runs ~2200 tok/s and does the whole
        1008-fact corpus in about two minutes, while one request per fact pays
        the round trip a thousand times. Batch 64 was no faster per token and
        gave every failure a longer tail, so embeddings.BATCH is 32.

        Shaped like backfill_entity_links(): a repair pass that is safe to
        re-run, reports what it did, and never raises on a transport failure —
        a half-finished backfill is a valid state that the next run completes.
        """
        summary = {"pending": 0, "embedded": 0, "failed": 0, "batches": 0}
        if not embeddings.available():
            summary["error"] = "numpy unavailable"
            return summary
        model = embeddings.resolve_model()
        pending = self.missing_embeddings(limit=limit)
        summary["pending"] = len(pending)
        if not pending:
            return summary

        size = batch or embeddings.BATCH
        for start in range(0, len(pending), size):
            chunk = pending[start : start + size]
            vectors = embeddings.embed([content for _, content in chunk], model=model)
            summary["batches"] += 1
            if vectors is None:
                summary["failed"] += len(chunk)
                if progress:
                    progress(summary)
                # Keep going: a single failed batch is usually a cold-start
                # timeout on the first request, and abandoning the run would
                # leave the corpus split across two vector states for a day.
                continue
            rows = [
                (fact_id, model, len(vec), embeddings.to_blob(vec))
                for (fact_id, _), vec in zip(chunk, vectors)
            ]
            with self._lock:
                self._conn.executemany(
                    """
                    INSERT INTO fact_embeddings (fact_id, model, dim, vector, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(fact_id) DO UPDATE SET
                        model = excluded.model,
                        dim = excluded.dim,
                        vector = excluded.vector,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    rows,
                )
                self._conn.commit()
                self._entry["embed_on"] = True
            summary["embedded"] += len(rows)
            if progress:
                progress(summary)
        return summary

    def embedding_matrix(self):
        """(fact_ids, matrix) for the whole corpus, or None when unavailable.

        Brute force is the right answer at this scale and will stay right for a
        long time: 1008 facts x 1024 float32 is 4 MB and one matmul, against
        which any vector database would be pure operational surface. The cache
        is what keeps it cheap per search — rebuilding it is a 4 MB read, and
        prefetch calls search() every turn.

        Cache validity is (model, row count, max updated_at). The MODEL is in
        the key because the matrix is filtered by it: swap HERMES_EMBED_MODEL
        without re-embedding and the count and timestamp are both unchanged,
        so a key without it would hand back the previous model's matrix to be
        scored against a query vector from the new one — two vector spaces
        silently dotted together, which reads as "the dense lane got worse"
        rather than as an error. The count and timestamp catch inserts and the
        nightly heal's refreshes; they would miss two processes rewriting the
        same row within one clock second, a state no writer here can produce
        (ensure_embedding is the only writer and the shared lock serialises it).
        """
        if not embeddings.available():
            return None
        numpy = hrr._np()
        model = embeddings.resolve_model()
        with self._lock:
            try:
                stamp = self._conn.execute(
                    "SELECT COUNT(*), MAX(updated_at) FROM fact_embeddings"
                ).fetchone()
            except sqlite3.DatabaseError:
                return None
            key = (model, int(stamp[0]), stamp[1])
            if key[1] == 0:
                return None
            cached = self._entry.get("embed_cache")
            if cached is not None and cached[0] == key:
                return cached[1], cached[2]
            rows = self._conn.execute(
                "SELECT fact_id, dim, vector FROM fact_embeddings WHERE model = ?"
                " ORDER BY fact_id",
                (model,),
            ).fetchall()

        ids: list[int] = []
        vectors: list = []
        dim = None
        for row in rows:
            if dim is None:
                dim = int(row["dim"])
            if int(row["dim"]) != dim:
                # Mixed dims inside one model name means the served model was
                # re-quantised or re-aliased under the same name. Skipping the
                # odd ones out keeps the matmul well-formed; the heal job's
                # model check will not catch this, so say so once.
                logger.warning(
                    "fact_embeddings holds dim %s and dim %s under model %r;"
                    " skipping the minority. Re-run the embedding backfill.",
                    dim,
                    row["dim"],
                    model,
                )
                continue
            vec = embeddings.from_blob(row["vector"], dim)
            if vec is None:
                continue
            ids.append(int(row["fact_id"]))
            vectors.append(vec)

        if not vectors:
            return None
        matrix = numpy.vstack(vectors)
        id_array = numpy.asarray(ids, dtype=numpy.int64)
        with self._lock:
            self._entry["embed_cache"] = (key, id_array, matrix)
        return id_array, matrix

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
