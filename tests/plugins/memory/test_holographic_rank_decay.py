"""Ranking regression tests for the reranked path (2026-08-29).

Three defects shipped together here, and every one of them was invisible to the
existing suite because that suite only exercises the additive blend — the path
production never takes, since HERMES_RERANK_URL is set for both consumers:

  1. ``search()`` computed the temporal decay and then threw it away when the
     cross-encoder stage rewrote ``fact["score"]``, so
     ``temporal_decay_half_life: 60`` was dead config in production.
  2. The trust multiplier ran raw (0.30-1.00) against a cross-encoder that
     saturates at 0.9997-0.9988 inside one topic, so trust alone decided the
     order: a retracted trust-0.70 synthesis outranked the trust-0.50 fact that
     answered the question, despite the reranker scoring it 4% lower.
  3. scripts/hermes_memory_mcp.py built its FactRetriever with the default
     half_life of 0 while the Hermes plugin passed 60.

So EVERY test here stubs the rerank stage ON. A ranking test that passes with
the reranker off is not testing the code that runs.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("numpy")  # retrieval imports numpy indirectly

from plugins.memory.holographic import holographic as hrr
from plugins.memory.holographic.retrieval import FactRetriever
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


def _age(store, fact_id, days):
    """Backdate a row. created_at/updated_at are SQLite CURRENT_TIMESTAMP."""
    ts = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    store._conn.execute(
        "UPDATE facts SET created_at = ?, updated_at = ? WHERE fact_id = ?",
        (ts, ts, fact_id),
    )
    store._conn.commit()


def _set_trust(store, fact_id, trust):
    store._conn.execute(
        "UPDATE facts SET trust_score = ? WHERE fact_id = ?", (trust, fact_id)
    )
    store._conn.commit()


def _stub_rerank(retriever, by_marker):
    """Turn the cross-encoder stage ON with deterministic scores.

    by_marker maps a substring of a fact's content to its ce score, so the stub
    survives the blend re-sort that decides what order documents arrive in.
    """
    retriever.rerank_url = "http://stub.invalid/v1/rerank"

    def _scores(query, documents):
        out = []
        for doc in documents:
            for marker, score in by_marker.items():
                if marker in doc:
                    out.append(score)
                    break
            else:                                        # pragma: no cover
                raise AssertionError(f"unstubbed document: {doc[:60]!r}")
        return out

    retriever._rerank_scores = _scores  # type: ignore[method-assign]


class TestDecaySurvivesTheRerank:
    """Leg (a): the decay factor must reach the reranked score."""

    def test_older_fact_loses_to_newer_at_equal_trust(self, store):
        old = store.add_fact(
            "MARKER-OLD: the document-store reranker runs on the CPU, measured June."
        )
        new = store.add_fact(
            "MARKER-NEW: the document-store reranker runs on the GPU, measured August."
        )
        _age(store, old, 120)
        _age(store, new, 1)

        r = FactRetriever(store, temporal_decay_half_life=60)
        # The saturated regime: the OLD fact even wins on raw relevance.
        _stub_rerank(r, {"MARKER-OLD": 0.9995, "MARKER-NEW": 0.9990})

        ranked = [f["fact_id"] for f in r.search("document-store reranker cpu gpu")]
        assert ranked[:2] == [new, old], (
            "temporal decay does not reach the reranked score — this is the "
            "2026-08-29 bug where stage 3 overwrote it"
        )

    def test_half_life_zero_disables_decay(self, store):
        old = store.add_fact(
            "MARKER-OLD: the document-store reranker runs on the CPU, measured June."
        )
        new = store.add_fact(
            "MARKER-NEW: the document-store reranker runs on the GPU, measured August."
        )
        _age(store, old, 120)
        _age(store, new, 1)

        r = FactRetriever(store, temporal_decay_half_life=0)
        _stub_rerank(r, {"MARKER-OLD": 0.9995, "MARKER-NEW": 0.9990})

        ranked = [f["fact_id"] for f in r.search("document-store reranker cpu gpu")]
        assert ranked[:2] == [old, new], "decay applied though half_life is 0"

    def test_decay_factor_is_not_leaked_to_callers(self, store):
        store.add_fact("MARKER-OLD: the document-store reranker runs on the CPU.")
        r = FactRetriever(store, temporal_decay_half_life=60)
        _stub_rerank(r, {"MARKER-OLD": 0.99})
        results = r.search("document-store reranker cpu")
        assert results
        for fact in results:
            assert "_decay" not in fact
            assert "hrr_vector" not in fact


class TestTrustClampOnTheRerankedPath:
    """Leg (c): trust may at most halve a score, it may not decide the order."""

    def _clamp_fixture(self, store):
        fid = store.add_fact("MARKER-ONE: document-store reranker sizing on the GPU.")
        # Stage 3 only runs on a pool of more than one, so the decoy is load-
        # bearing: with a single candidate this would silently measure the
        # blend path instead.
        store.add_fact("MARKER-TWO: document-store reranker sizing, second note on GPU.")
        _set_trust(store, fid, 0.70)
        _age(store, fid, 60)  # exactly one half-life
        return fid

    def test_score_is_ce_times_clamped_trust_times_decay(self, store):
        """The formula itself, on the path where it is still the score.

        Since the 2026-09-05 rank-fusion change the cross-encoder's product no
        longer becomes `score` on the production path — it becomes one of the
        two RANKINGS that RRF fuses (`ce_final` in search()). It is still
        computed exactly as before, and rerank_fusion=False is the arm that
        exposes it directly, so that is where the arithmetic is pinned. The
        ORDERING consequences of the clamp are covered by the two tests below,
        which run on the production path.
        """
        fid = self._clamp_fixture(store)
        r = FactRetriever(store, temporal_decay_half_life=60, rerank_fusion=False)
        _stub_rerank(r, {"MARKER-ONE": 0.90, "MARKER-TWO": 0.10})

        results = {f["fact_id"]: f for f in r.search("document-store reranker gpu")}
        assert results[fid]["score"] == pytest.approx(
            0.90 * (0.5 + 0.5 * 0.70) * 0.5, rel=1e-3
        )

    def test_fused_score_is_an_rrf_score_not_the_ce_product(self, store):
        """Guards the scale change, which callers can see.

        A consumer that thresholds on `score` would silently misbehave if this
        ever reverted: the fused score is a sum of two reciprocal ranks,
        nowhere near the ce product.

        This fixture is also the two lists DISAGREEING, which is the case worth
        pinning. MARKER-ONE is ce rank 1 (0.90 against 0.10) but blend rank 2,
        because one half-life of decay takes its raw trust term to 0.70 * 0.5 =
        0.35 against the decoy's 0.50 * 1.0. So its fused score is
        1/(K+2) + 1/(K+1), and it still comes out on top — the cross-encoder
        wins a rank-1-versus-rank-2 disagreement on the ce tiebreak.
        """
        from plugins.memory.holographic.retrieval import _RRF_K

        fid = self._clamp_fixture(store)
        r = FactRetriever(store, temporal_decay_half_life=60)
        _stub_rerank(r, {"MARKER-ONE": 0.90, "MARKER-TWO": 0.10})

        ordered = r.search("document-store reranker gpu")
        results = {f["fact_id"]: f for f in ordered}
        assert results[fid]["score"] == pytest.approx(
            1.0 / (_RRF_K + 2.0) + 1.0 / (_RRF_K + 1.0)
        )
        assert ordered[0]["fact_id"] == fid
        # and the internal stashes never reach the caller
        assert "_ce_final" not in results[fid]
        assert "_ce" not in results[fid]

    def test_stale_high_trust_loses_to_fresh_default_trust(self, store):
        """The live F1 shape, with the measured numbers from the audit.

        fid 986 (trust 0.70, 10 days old, ce 0.9599, text reads "RETIRED")
        outranked fid 1305 (trust 0.50, fresh, ce 0.9990) — the reranker DID
        detect the 4% deficit and the raw trust term swamped it.
        """
        stale = store.add_fact(
            "MARKER-STALE: llama-swap config-reload playbook, the document-store "
            "reranker lane is RETIRED."
        )
        current = store.add_fact(
            "MARKER-CURRENT: the GPU reranker DOES fit, measured with "
            "config reload on llama-swap."
        )
        _set_trust(store, stale, 0.70)
        _set_trust(store, current, 0.50)
        _age(store, stale, 10)
        _age(store, current, 1)

        r = FactRetriever(store, temporal_decay_half_life=60)
        _stub_rerank(r, {"MARKER-STALE": 0.9599, "MARKER-CURRENT": 0.9990})

        ranked = [f["fact_id"] for f in r.search("document-store reranker gpu llama-swap")]
        assert ranked[0] == current, (
            "a stale trust-0.70 fact still outranks the trust-0.50 fact that "
            "answers the question — the trust clamp regressed"
        )

    def test_a_demoted_fact_still_loses_to_a_current_one(self, store):
        """The clamp must not neuter the supersession lane.

        A retracted row sits at _SUPERSESSION_FLOOR (0.30) and, because
        record_feedback stamps updated_at, looks BRAND NEW to the decay term.
        Trust alone has to be enough there.
        """
        demoted = store.add_fact("MARKER-DEMOTED: the GPU reranker does not fit.")
        current = store.add_fact("MARKER-CURRENT: the GPU reranker does fit now.")
        _set_trust(store, demoted, 0.30)
        _age(store, demoted, 0)
        _age(store, current, 0)

        r = FactRetriever(store, temporal_decay_half_life=60)
        # Saturated and tied in the reranker's eyes, as the live store measures.
        _stub_rerank(r, {"MARKER-DEMOTED": 0.9997, "MARKER-CURRENT": 0.9988})

        ranked = [f["fact_id"] for f in r.search("gpu reranker fit")]
        assert ranked[0] == current


class TestBlendPathUnchanged:
    def test_blend_still_uses_the_raw_trust_multiplier(self, store):
        """The clamp is rerank-path-ONLY.

        The blend's relevance term has real dynamic range, so trust cannot swamp
        it there, and the tuned fts/jaccard/hrr weights assume the raw
        multiplier. Recomputed here the same way
        test_hoisted_query_vector_matches_per_candidate does, because the whole
        point is that this arithmetic did not change.
        """
        fid = store.add_fact("MARKER-ONE: document-store reranker sizing on the GPU.")
        _set_trust(store, fid, 0.70)
        query = "document-store reranker gpu"

        r = FactRetriever(store, temporal_decay_half_life=0, rerank_url="")
        r.rerank_url = ""  # ignore an inherited HERMES_RERANK_URL
        [candidate] = r._fts_candidates(query, None, 0.3, 30)
        query_tokens = r._tokenize(query)
        all_tokens = r._tokenize(candidate["content"]) | r._tokenize(
            candidate.get("tags", "")
        )
        role_content = hrr.encode_atom("__hrr_role_content__", r.hrr_dim)
        query_vec = hrr.bind(hrr.encode_text(query, r.hrr_dim), role_content)
        fact_vec = hrr.bytes_to_phases(candidate["hrr_vector"], dim=r.hrr_dim)
        hrr_sim = (hrr.similarity(query_vec, fact_vec) + 1.0) / 2.0
        relevance = (
            r.fts_weight * candidate.get("fts_rank", 0.0)
            + r.jaccard_weight * r._jaccard_similarity(query_tokens, all_tokens)
            + r.hrr_weight * hrr_sim
        )

        [result] = r.search(query)
        assert result["fact_id"] == fid
        assert result["score"] == pytest.approx(relevance * 0.70)
        # The clamp would have produced relevance * 0.85.
        assert result["score"] != pytest.approx(relevance * 0.85)
