"""Tests for the dense candidate lane (fact_embeddings) and the abstention gate.

The 2026-09-02 retrieval eval found the store's real limit was candidate
GENERATION, not ranking: every arm above raw BM25 reorders the same FTS5 pool,
and that pool held the gold fact for 36% of paraphrase-shaped questions. A
question that shares no words with the stored fact cannot be answered by a
keyword index, however well the survivors are ranked. The dense lane is a
second candidate generator — jina-embed vectors, brute-force cosine, UNIONed
with the FTS pool — and these tests pin the properties that make it safe to put
on the write path of a store that had never made a network call before:

  * it is INERT until the table has rows, so a fresh store — every tmp_path
    store in this suite — reaches the network exactly never, whatever the
    environment says;
  * an embedding failure can never fail a fact write;
  * the process-wide write lock is NOT held across the HTTP call;
  * dense_url="" is authoritative OFF (the rerank_url="" trap, not repeated);
  * when the lane is off, scoring is arithmetically the pre-lane blend.

The abstention gate (Track 2) rides on the same search() and is tested here
too, because its one load-bearing rule — the floor sits on the raw
cross-encoder score, never on fact["score"] — is a property of search()'s
metadata, not of either door that renders it.

No test here touches llama-swap. The embedding backend is a deterministic fake
that maps synonym groups to axes, so "reverted" and "rollback" embed to the
same vector while sharing no token — the exact situation FTS cannot handle.
"""

import json
import os
import sqlite3
import threading
import time

import pytest

pytest.importorskip("numpy")

from plugins.memory.holographic import embeddings
from plugins.memory.holographic.retrieval import (
    FactRetriever,
    _query_shape,
    no_confident_match,
)
from plugins.memory.holographic.store import MemoryStore


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


# A tiny embedding model: each axis is a synonym group. A text's vector is the
# normalised sum of the axes its words hit; a text hitting nothing gets a
# vector on a junk axis so it is never NaN and never accidentally close.
_AXES = [
    {"rollback", "reverted", "revert", "undo", "rolled"},
    {"gravity", "mascon", "geoid"},
    {"reranker", "cross-encoder", "rerank"},
    {"telegram", "bot", "chat"},
]
_DIM = len(_AXES) + 1


def _fake_vector(text):
    import numpy as np

    words = set(text.lower().replace(",", " ").replace(".", " ").split())
    vec = np.zeros(_DIM, dtype=np.float32)
    for i, group in enumerate(_AXES):
        if words & group:
            vec[i] = 1.0
    if not vec.any():
        vec[-1] = 1.0
    return (vec / np.linalg.norm(vec)).tolist()


@pytest.fixture
def fake_backend(monkeypatch):
    """Replace the HTTP embedder with the synonym model; count the calls."""
    calls = []

    def fake_embed(texts, *, url=None, model="", timeout=None):
        if embeddings.resolve_url(url) == "":
            return None
        calls.append(list(texts))
        return [_fake_vector(t) for t in texts]

    monkeypatch.setattr(embeddings, "embed", fake_embed)
    return calls


def _count(store, table):
    return store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


# ---------------------------------------------------------------------------


class TestLaneIsInertUntilActivated:
    def test_a_fresh_store_never_calls_the_embedder(self, store, fake_backend):
        store.add_fact("LESSON: the deploy was rolled back at noon", category="lesson")
        store.update_fact(1, content="LESSON: the deploy was rolled back at one")
        r = FactRetriever(store, rerank_url="")
        r.search("why was the release reverted")
        assert fake_backend == []
        assert store.embedding_matrix() is None
        assert not store._embeddings_active()

    def test_activation_is_the_data_not_a_flag(self, store, fake_backend):
        fid = store.add_fact("LESSON: the deploy was rolled back at noon", category="lesson")
        assert store.ensure_embedding(fid) is True
        assert store._embeddings_active()
        # From here on the write path embeds by itself.
        store.add_fact("LESSON: mascon gravity solutions need a floor", category="lesson")
        assert store.missing_embeddings() == []
        ids, matrix = store.embedding_matrix()
        assert ids.tolist() == [1, 2]
        assert matrix.shape == (2, _DIM)

    def test_activation_survives_a_reopen(self, tmp_path, fake_backend):
        path = tmp_path / "memory_store.db"
        first = MemoryStore(path)
        fid = first.add_fact("LESSON: the deploy was rolled back", category="lesson")
        first.ensure_embedding(fid)
        first.close()
        MemoryStore._shared.clear()
        second = MemoryStore(path)
        try:
            assert second._embeddings_active()
            second.add_fact("LESSON: the bot went quiet", category="lesson")
            assert second.missing_embeddings() == []
        finally:
            second.close()


class TestSchemaMigration:
    def test_table_self_creates_on_a_preexisting_database(self, tmp_path):
        """Same shape as the fact_markers migration test: drop the table from a
        real store rather than hand-roll a legacy schema, because a facts table
        without its FTS index fails on the first UPDATE for unrelated reasons."""
        path = tmp_path / "memory_store.db"
        s = MemoryStore(path)
        s.add_fact("LESSON: predates the dense lane", category="lesson")
        s._conn.execute("DROP TABLE fact_embeddings")
        s._conn.commit()
        s.close()
        MemoryStore._shared.clear()
        reopened = MemoryStore(path)
        try:
            tables = {
                r[0]
                for r in reopened._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            assert "fact_embeddings" in tables
            assert reopened.missing_embeddings() == [(1, "LESSON: predates the dense lane")]
        finally:
            reopened.close()


class TestWritePathNeverFailsTheWrite:
    def test_an_embedder_that_raises_still_leaves_a_durable_fact(
        self, store, fake_backend, monkeypatch
    ):
        fid = store.add_fact("LESSON: seed", category="lesson")
        store.ensure_embedding(fid)

        def boom(*a, **k):
            raise RuntimeError("llama-swap is on fire")

        monkeypatch.setattr(embeddings, "embed", boom)
        new = store.add_fact("LESSON: written during the fire", category="lesson")
        assert store.get_fact(new)["content"] == "LESSON: written during the fire"
        assert store.missing_embeddings() == [(new, "LESSON: written during the fire")]

    def test_an_embedder_that_returns_none_is_the_same(self, store, fake_backend, monkeypatch):
        fid = store.add_fact("LESSON: seed", category="lesson")
        store.ensure_embedding(fid)
        monkeypatch.setattr(embeddings, "embed", lambda *a, **k: None)
        new = store.add_fact("LESSON: timed out", category="lesson")
        assert store.get_fact(new) is not None
        assert [f for f, _ in store.missing_embeddings()] == [new]

    def test_the_write_lock_is_released_across_the_embedding_call(
        self, store, fake_backend, monkeypatch
    ):
        """The lock is process-wide: every MemoryStore in a gateway shares it.
        An HTTP call inside it would stall every other memory write for as long
        as llama-swap takes, which on a cold load is seconds."""
        fid = store.add_fact("LESSON: seed", category="lesson")
        store.ensure_embedding(fid)
        started = threading.Event()
        real = embeddings.embed

        def slow(*a, **k):
            started.set()
            time.sleep(0.6)
            return real(*a, **k)

        monkeypatch.setattr(embeddings, "embed", slow)
        waited = []

        def writer():
            started.wait(5)
            t = time.time()
            store.update_fact(fid, trust_delta=0.01)  # takes the lock, embeds nothing
            waited.append(time.time() - t)

        th = threading.Thread(target=writer)
        th.start()
        store.add_fact("LESSON: slow to embed", category="lesson")
        th.join(5)
        assert waited and waited[0] < 0.3


class TestVectorFollowsContent:
    def _blob(self, store, fid):
        return store._conn.execute(
            "SELECT vector FROM fact_embeddings WHERE fact_id = ?", (fid,)
        ).fetchone()[0]

    def test_content_update_refreshes_and_tags_update_does_not(self, store, fake_backend):
        fid = store.add_fact("LESSON: the deploy was rolled back", category="lesson")
        store.ensure_embedding(fid)
        before = self._blob(store, fid)
        store.update_fact(fid, tags="ops,deploy")
        assert self._blob(store, fid) == before
        store.update_fact(fid, content="LESSON: the reranker went down instead")
        assert self._blob(store, fid) != before

    def test_remove_takes_the_vector_with_it(self, store, fake_backend):
        fid = store.add_fact("LESSON: the deploy was rolled back", category="lesson")
        store.ensure_embedding(fid)
        assert store.remove_fact(fid, force=True)
        assert _count(store, "fact_embeddings") == 0

    def test_prune_clears_vectors_left_by_pre_table_code(self, store, fake_backend):
        fid = store.add_fact("LESSON: the deploy was rolled back", category="lesson")
        store.ensure_embedding(fid)
        # Delete the way a process on old code would: the row only.
        store._conn.execute("DELETE FROM facts WHERE fact_id = ?", (fid,))
        store._conn.commit()
        assert _count(store, "fact_embeddings") == 1
        assert store.prune_orphan_embeddings() == 1
        assert _count(store, "fact_embeddings") == 0

    def test_ensure_embedding_is_idempotent_without_force(self, store, fake_backend):
        fid = store.add_fact("LESSON: the deploy was rolled back", category="lesson")
        assert store.ensure_embedding(fid) is True
        n = len(fake_backend)
        assert store.ensure_embedding(fid) is False
        assert len(fake_backend) == n


class TestBackfill:
    def test_backfills_only_what_is_missing_and_reports_it(self, store, fake_backend):
        for i in range(5):
            store.add_fact(f"LESSON: fact number {i}", category="lesson")
        first = store.backfill_embeddings(batch=2)
        assert (first["pending"], first["embedded"], first["batches"]) == (5, 5, 3)
        second = store.backfill_embeddings()
        assert (second["pending"], second["embedded"]) == (0, 0)

    def test_a_failed_batch_is_counted_and_does_not_abort_the_run(
        self, store, fake_backend, monkeypatch
    ):
        for i in range(4):
            store.add_fact(f"LESSON: fact number {i}", category="lesson")
        real = embeddings.embed
        seen = []

        def flaky(texts, **k):
            seen.append(len(texts))
            return None if len(seen) == 1 else real(texts, **k)

        monkeypatch.setattr(embeddings, "embed", flaky)
        report = store.backfill_embeddings(batch=2)
        assert report["failed"] == 2 and report["embedded"] == 2
        assert len(store.missing_embeddings()) == 2

    def test_a_model_swap_makes_every_row_missing_again(self, store, fake_backend, monkeypatch):
        fid = store.add_fact("LESSON: the deploy was rolled back", category="lesson")
        store.ensure_embedding(fid)
        assert store.missing_embeddings() == []
        monkeypatch.setenv("HERMES_EMBED_MODEL", "some-other-embedder")
        assert [f for f, _ in store.missing_embeddings()] == [fid]

    def test_a_model_swap_invalidates_the_matrix_cache(self, store, fake_backend, monkeypatch):
        """Count and max(updated_at) are both unchanged by a bare model swap, so
        a cache key without the model would score the OLD model's vectors
        against a query embedded by the NEW one."""
        fid = store.add_fact("LESSON: the deploy was rolled back", category="lesson")
        store.ensure_embedding(fid)
        ids, _ = store.embedding_matrix()
        assert ids.tolist() == [fid]
        monkeypatch.setenv("HERMES_EMBED_MODEL", "some-other-embedder")
        assert store.embedding_matrix() is None


# ---------------------------------------------------------------------------


class TestSearchUnion:
    @pytest.fixture
    def populated(self, store, fake_backend):
        store.add_fact("LESSON: the deploy was rolled back after the canary failed",
                       category="lesson")
        store.add_fact("LESSON: mascon gravity fields need a positivity floor",
                       category="lesson")
        store.add_fact("LESSON: the reranker cold start costs the first turn",
                       category="lesson")
        store.add_fact("LESSON: the telegram bot token was duplicated in the env file",
                       category="lesson")
        assert store.backfill_embeddings()["embedded"] == 4
        return store

    def test_a_paraphrase_sharing_no_token_with_the_fact_is_found(self, populated):
        r = FactRetriever(populated, rerank_url="", hrr_weight=0.0)
        # "reverted" shares no token with "rolled back"; FTS alone finds nothing.
        assert r._fts_candidates("why was the release reverted", None, 0.3, 30) == []
        results, meta = r.search("why was the release reverted", with_meta=True)
        assert results and "rolled back" in results[0]["content"]
        assert meta["shape"] == "semantic" and meta["dense_candidates"] > 0

    def test_dense_off_is_authoritative_even_with_the_env_exported(
        self, populated, monkeypatch
    ):
        monkeypatch.setenv("HERMES_EMBED_URL", "http://127.0.0.1:1/v1/embeddings")
        off = FactRetriever(populated, rerank_url="", dense_url="")
        assert off.dense_url == ""
        assert off.search("why was the release reverted") == []
        on = FactRetriever(populated, rerank_url="")
        assert on.dense_url == "http://127.0.0.1:1/v1/embeddings"

    def test_fts_hits_keep_their_rank_and_dense_hits_get_none(self, populated):
        r = FactRetriever(populated, rerank_url="", hrr_weight=0.0)
        results = r.search("gravity mascon floor")
        top = results[0]
        assert "mascon" in top["content"]
        assert top["fts_rank"] > 0

    def test_category_and_trust_filters_apply_to_dense_candidates(self, populated):
        r = FactRetriever(populated, rerank_url="", hrr_weight=0.0)
        assert r.search("why was the release reverted", category="paper") == []
        assert r.search("why was the release reverted", min_trust=0.9) == []

    def test_an_embedder_failure_degrades_to_fts_only(self, populated, monkeypatch):
        r = FactRetriever(populated, rerank_url="", hrr_weight=0.0)
        monkeypatch.setattr(embeddings, "embed", lambda *a, **k: None)
        assert r.search("why was the release reverted") == []
        results, meta = r.search("reranker cold start", with_meta=True)
        assert results and meta["dense_candidates"] == 0

    def test_scoring_is_the_old_blend_whenever_dense_produced_nothing(
        self, populated, monkeypatch
    ):
        """dense_share is applied only when the lane scored something. Without
        this, a failed embed would silently rescale every weight by (1-share)
        on exactly the path that is supposed to be the pre-lane one."""
        r_on = FactRetriever(populated, rerank_url="")
        monkeypatch.setattr(embeddings, "embed", lambda *a, **k: None)
        r_off = FactRetriever(populated, rerank_url="", dense_url="")
        q = "reranker cold start"
        assert [(f["fact_id"], f["score"]) for f in r_on.search(q)] == [
            (f["fact_id"], f["score"]) for f in r_off.search(q)
        ]


class TestQueryShape:
    @pytest.mark.parametrize(
        "query",
        [
            'what does "database is locked" mean here',
            "what did fid 940 say",
            "where is ingest_shortest_first.sh",
            "which llama.cpp build serves head-dim-256 models",
            "why did the GPU OOM",
            "how does FactRetriever pick candidates",
        ],
    )
    def test_literal_shapes_route_lexical(self, query):
        assert _query_shape(query) == "lexical"

    @pytest.mark.parametrize(
        "query",
        [
            "Wasn't there a day when the daily review reported nothing had run?",
            "the author's method for the gravity of a curved block",
            "can you tell an AI agent is going wrong while it is still running",
            "how did I rearrange the way the laptop reaches the file server",
        ],
    )
    def test_prose_with_apostrophes_and_short_acronyms_routes_semantic(self, query):
        assert _query_shape(query) == "semantic"


# ---------------------------------------------------------------------------


class TestAbstention:
    def test_the_verdict_fires_only_below_the_floor_on_the_reranked_scale(self):
        base = {"n_results": 5, "reranked": True, "top_ce": 0.2}
        assert no_confident_match(base, floor=0.5) == {"top_ce": 0.2, "floor": 0.5}
        assert no_confident_match({**base, "top_ce": 0.9}, floor=0.5) is None
        assert no_confident_match({**base, "reranked": False}, floor=0.5) is None
        assert no_confident_match({**base, "n_results": 0}, floor=0.5) is None
        assert no_confident_match(base, floor=0.0) is None

    def test_search_meta_reports_the_raw_cross_encoder_score(
        self, store, fake_backend, monkeypatch
    ):
        store.add_fact("LESSON: the deploy was rolled back", category="lesson")
        r = FactRetriever(store, rerank_url="http://stub")
        monkeypatch.setattr(r, "_rerank_scores", lambda q, docs: [0.37] * len(docs))
        # A second row so the rerank stage (len > 1) actually runs.
        store.add_fact("LESSON: the deploy was rolled forward", category="lesson")
        results, meta = r.search("deploy rolled", with_meta=True)
        assert meta["reranked"] is True
        assert meta["top_ce"] == pytest.approx(0.37)
        # top_score is an RRF score since 2026-09-05 — with every ce equal the
        # two rankings agree, so the winner is rank 1 in both. The point of the
        # assertion is unchanged: top_score carries policy, top_ce must not.
        from plugins.memory.holographic.retrieval import _RRF_K

        assert meta["top_score"] == pytest.approx(2.0 / (_RRF_K + 1.0))
        assert meta["top_ce"] != pytest.approx(meta["top_score"])
        assert all(
            "_ce" not in f and "_decay" not in f and "_ce_final" not in f
            for f in results
        )

    def test_top_ce_is_the_best_returned_row_not_rank_zero(
        self, store, fake_backend, monkeypatch
    ):
        """The gate must not be a function of the ranking policy.

        Rank 0 is chosen by RRF, which can seat a blend-favoured row the
        cross-encoder scored near zero above rows it scored ~1.0. Reading rank
        0's ce alone made the calibrated floor move whenever the ranking moved
        — that is what forced the 2026-09-05 recalibration. top_ce is the best
        ce among the rows the caller actually gets, so a result set containing
        a confident answer never reads as "no confident match" merely because
        something else sorted above it.
        """
        store.add_fact("LESSON: the deploy was rolled back after the outage")
        store.add_fact("LESSON: deploy rollback runbook, second note")
        r = FactRetriever(store, rerank_url="http://stub")
        # Whichever row lands at rank 0, one of the two scored 0.99.
        seen = {}

        def _stub(query, docs):
            seen["docs"] = docs
            return [0.01, 0.99][: len(docs)]

        monkeypatch.setattr(r, "_rerank_scores", _stub)
        results, meta = r.search("deploy rollback", with_meta=True)
        assert len(results) == 2
        assert meta["top_ce"] == pytest.approx(0.99)
        assert no_confident_match(meta, floor=0.5) is None

    def test_the_fact_store_door_adds_sibling_keys_and_keeps_every_row(
        self, tmp_path, fake_backend, monkeypatch
    ):
        from plugins.memory.holographic import HolographicMemoryProvider

        provider = HolographicMemoryProvider(
            config={"db_path": str(tmp_path / "m.db"), "hrr_dim": 64}
        )
        provider.initialize("test-session")
        try:
            provider._store.add_fact("LESSON: the deploy was rolled back", category="lesson")
            provider._store.add_fact("LESSON: the deploy was rolled forward", category="lesson")
            monkeypatch.setattr(
                provider._retriever, "_rerank_scores", lambda q, d: [0.1] * len(d)
            )
            provider._retriever.rerank_url = "http://stub"
            monkeypatch.setattr(
                "plugins.memory.holographic.retrieval.ABSTAIN_FLOOR", 0.5
            )
            payload = json.loads(
                provider.handle_tool_call(
                    "fact_store", {"action": "search", "query": "deploy rolled"}
                )
            )
            assert payload["count"] == 2 and len(payload["results"]) == 2
            assert payload["no_confident_match"] is True
            assert payload["floor"] == 0.5 and payload["top_ce"] == pytest.approx(0.1)
        finally:
            provider.shutdown()


class TestEmbeddingsModule:
    def test_resolve_url_has_three_states(self, monkeypatch):
        monkeypatch.setenv("HERMES_EMBED_URL", "http://env/v1/embeddings")
        assert embeddings.resolve_url(None) == "http://env/v1/embeddings"
        assert embeddings.resolve_url("") == ""
        assert embeddings.resolve_url("http://x") == "http://x"
        monkeypatch.delenv("HERMES_EMBED_URL")
        assert embeddings.resolve_url(None) == embeddings.DEFAULT_URL

    def test_embed_with_url_off_returns_none_without_a_request(self, monkeypatch):
        import urllib.request

        def no_network(*a, **k):
            raise AssertionError("network touched")

        monkeypatch.setattr(urllib.request, "urlopen", no_network)
        assert embeddings.embed(["x"], url="") is None
        assert embeddings.embed([]) == []

    def test_blob_roundtrip_normalises_and_rejects_the_wrong_dim(self):
        import numpy as np

        blob = embeddings.to_blob([3.0, 4.0])
        vec = embeddings.from_blob(blob, 2)
        assert np.allclose(vec, [0.6, 0.8])
        assert embeddings.from_blob(blob, 3) is None
        assert embeddings.from_blob(b"", 2) is None


class TestSupersededByPointer:
    """The FORWARD pointer from a retracted fact to what replaced it.

    valid_until has recorded WHEN a claim stopped being current since
    2026-09-02; superseded_by records WHAT replaced it (2026-09-05). Only the
    reverse direction existed before, in the retracting fact's own prose.
    """

    def test_retraction_records_the_retracting_fid(self, store):
        old = store.add_fact("LESSON: the reranker runs on the CPU.")
        new = store.add_fact(
            f"LESSON: the reranker runs on the GPU — supersedes fid {old}."
        )
        row = store.get_fact(old)
        assert row["superseded_by"] == new
        assert row["valid_until"] is not None
        assert store.get_fact(new)["superseded_by"] is None

    def test_first_close_wins_for_the_pointer_too(self, store):
        """A fact retracted twice names the retraction that ENDED it.

        valid_until and superseded_by move under one predicate because they are
        two halves of one event; letting the pointer drift to the latest
        mention would contradict the timestamp sitting next to it.
        """
        old = store.add_fact("LESSON: the reranker runs on the CPU.")
        first = store.add_fact(f"LESSON: actually the GPU — corrects fid {old}.")
        second = store.add_fact(f"LESSON: the GPU, confirmed — corrects fid {old}.")
        assert store.get_fact(old)["superseded_by"] == first != second

    def test_forward_reference_never_sets_the_pointer(self, store):
        """The passive voice names the NEWER fact, so it must not close a window."""
        first = store.add_fact("LESSON: the reranker runs on the CPU.")
        second = store.add_fact(
            f"LESSON: the reranker runs on the GPU (superseded by fid {first})."
        )
        assert store.get_fact(second)["superseded_by"] is None
        assert store.get_fact(first)["superseded_by"] is None

    def test_pointer_is_recovered_for_windows_closed_before_the_column(self, store):
        """The historical repair, which every store open re-runs.

        Simulated by clearing the pointer while leaving valid_until set —
        exactly the state a store migrated from before 2026-09-05 is in.
        """
        old = store.add_fact("LESSON: the reranker runs on the CPU.")
        new = store.add_fact(
            f"LESSON: the reranker runs on the GPU — supersedes fid {old}."
        )
        store._conn.execute(
            "UPDATE facts SET superseded_by = NULL WHERE fact_id = ?", (old,)
        )
        store._conn.commit()
        assert store.get_fact(old)["superseded_by"] is None

        store._backfill_superseded_by()
        assert store.get_fact(old)["superseded_by"] == new

    def test_repair_leaves_an_ambiguous_retractor_null(self, store):
        """A wrong pointer is worse than a missing one.

        Two facts sharing the retractor's created_at and both naming the target
        cannot be told apart, so neither is chosen.
        """
        old = store.add_fact("LESSON: the reranker runs on the CPU.")
        a = store.add_fact(f"LESSON: the GPU actually — corrects fid {old}.")
        b = store.add_fact(f"LESSON: the GPU indeed — corrects fid {old}.")
        store._conn.execute(
            "UPDATE facts SET created_at = (SELECT created_at FROM facts"
            "   WHERE fact_id = ?) WHERE fact_id = ?", (a, b)
        )
        store._conn.execute(
            "UPDATE facts SET superseded_by = NULL, valid_until ="
            " (SELECT created_at FROM facts WHERE fact_id = ?) WHERE fact_id = ?",
            (a, old),
        )
        store._conn.commit()
        store._backfill_superseded_by()
        assert store.get_fact(old)["superseded_by"] is None

        # ...and prove that NULL is the ambiguity and not an inert repair:
        # remove the second claimant and the same call now resolves.
        store._conn.execute(
            "UPDATE facts SET content = 'LESSON: unrelated note' WHERE fact_id = ?",
            (b,),
        )
        store._conn.commit()
        store._backfill_superseded_by()
        assert store.get_fact(old)["superseded_by"] == a

    def test_the_pointer_reaches_search_results(self, store, fake_backend):
        old = store.add_fact("LESSON: the deploy rollback runs on the CPU.")
        store.add_fact(
            f"LESSON: the deploy rollback runs on the GPU — supersedes fid {old}."
        )
        r = FactRetriever(store, rerank_url="", dense_url="")
        rows = {f["fact_id"]: f for f in r.search("deploy rollback", min_trust=0.0)}
        assert rows[old]["superseded_by"] is not None
