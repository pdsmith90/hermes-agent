"""Tests for the entity-coverage extension: single capitalized words,
lowercase snake_case identifiers, tag-derived entities, and the additive
backfill.

Before this, "Hermes" was unreachable by the entity probe in 111 of the 113
facts naming it: every extraction rule was structurally blind to a single
capitalized word, lowercase identifiers, and the tags field.
"""

import sqlite3

import pytest

from plugins.memory.holographic.store import MemoryStore, _tag_entities


@pytest.fixture(autouse=True)
def _clean_shared_registry():
    for entry in list(MemoryStore._shared.values()):
        try:
            entry["conn"].close()
        except sqlite3.Error:
            pass
    MemoryStore._shared.clear()
    yield
    for entry in list(MemoryStore._shared.values()):
        try:
            entry["conn"].close()
        except sqlite3.Error:
            pass
    MemoryStore._shared.clear()


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "memory_store.db")
    yield s
    s.close()


def _linked_entities(store, fact_id):
    rows = store._conn.execute(
        """
        SELECT e.name FROM entities e
        JOIN fact_entities fe ON fe.entity_id = e.entity_id
        WHERE fe.fact_id = ?
        """,
        (fact_id,),
    ).fetchall()
    return {r["name"] for r in rows}


class TestSingleCapitalizedWords:
    def test_mid_sentence_name_is_extracted(self, store):
        names = store._extract_entities(
            "The nightly run restarts the Hermes gateway before dawn."
        )
        assert "Hermes" in names

    def test_sentence_initial_only_is_not_extracted(self, store):
        # Every occurrence sentence-initial: just orthography.
        names = store._extract_entities(
            "Deleted six facts in one turn. Deleted them against the prompt."
        )
        assert "Deleted" not in names

    def test_sentence_initial_name_rescued_by_later_occurrence(self, store):
        names = store._extract_entities(
            "Hermes failed at 03:10. The fix restarts Hermes cleanly."
        )
        assert "Hermes" in names

    def test_months_days_and_literals_are_stopped(self, store):
        names = store._extract_entities(
            "The run on Friday in August returned True for every case."
        )
        assert "Friday" not in names
        assert "August" not in names
        assert "True" not in names

    def test_word_inside_multiword_span_not_promoted(self, store):
        names = store._extract_entities(
            "We reviewed the Claude Code transcripts again."
        )
        assert "Claude Code" in names
        assert "Claude" not in names
        assert "Code" not in names

    def test_edge_words_never_promoted(self, store):
        names = store._extract_entities("It said the This keyword is odd.")
        assert "This" not in names


class TestSnakeCaseIdentifiers:
    def test_lowercase_snake_case_extracted(self, store):
        names = store._extract_entities(
            "the fact_store tool writes through memory_store directly"
        )
        assert "fact_store" in names
        assert "memory_store" in names

    def test_hyphenated_lowercase_still_excluded(self, store):
        # Prose compounds are lexically inseparable from lowercase-hyphen
        # names; those reach the graph via tags instead.
        names = store._extract_entities(
            "a real-time check of the climate-driven signal"
        )
        assert "real-time" not in names
        assert "climate-driven" not in names


class TestTagEntities:
    def test_tag_parse(self):
        assert _tag_entities("hermes, llama-swap , open-question") == [
            "hermes", "llama-swap", "open-question"
        ]
        assert _tag_entities("") == []
        assert _tag_entities("a") == []  # below min length

    def test_add_fact_links_tag_entities(self, store):
        fid = store.add_fact(
            "LESSON: the swap config must keep ttl above the job gap.",
            category="lesson",
            tags="hermes,llama-swap",
        )
        linked = _linked_entities(store, fid)
        assert "hermes" in linked
        assert "llama-swap" in linked

    def test_update_tags_adds_links(self, store):
        fid = store.add_fact("bare fact with no names", category="general")
        store.update_fact(fid, tags="zotero")
        assert "zotero" in _linked_entities(store, fid)

    def test_content_update_preserves_tag_links(self, store):
        # The content path drops all links before re-extracting; tag-derived
        # links must be restored from the row's current tags.
        fid = store.add_fact("first version", category="general", tags="hermes")
        store.update_fact(fid, content="second version entirely")
        assert "hermes" in _linked_entities(store, fid)


class TestBackfill:
    def test_backfill_adds_missing_links_only(self, store):
        # Simulate a pre-coverage row: insert directly, bypassing extraction.
        store._conn.execute(
            "INSERT INTO facts (content, category, tags) VALUES (?, ?, ?)",
            ("the Hermes gateway calls fact_store nightly", "lesson", "cron"),
        )
        fid = store._conn.execute(
            "SELECT fact_id FROM facts WHERE content LIKE 'the Hermes%'"
        ).fetchone()["fact_id"]
        assert _linked_entities(store, fid) == set()

        result = store.backfill_entity_links()
        assert result["links_added"] >= 3
        linked = _linked_entities(store, fid)
        assert {"Hermes", "fact_store", "cron"} <= linked

    def test_backfill_is_idempotent(self, store):
        store.add_fact("the Hermes gateway calls fact_store nightly",
                       category="lesson", tags="cron")
        first = store.backfill_entity_links()
        second = store.backfill_entity_links()
        assert second["links_added"] == 0
        assert second["facts_changed"] == 0
        assert first["facts_scanned"] == second["facts_scanned"]

    def test_backfill_never_removes_links(self, store):
        fid = store.add_fact("GRACE-FO telemetry via LightRAG",
                             category="paper", tags="legacy-tag")
        before = _linked_entities(store, fid)
        store.backfill_entity_links()
        assert before <= _linked_entities(store, fid)
