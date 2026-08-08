"""Tests for FactRetriever FTS5 query sanitization.

These tests cover the fix where raw natural-language queries passed to
FTS5 MATCH were AND-joined by default, dropping recall to zero on any
multi-word prose query. The sanitizer drops stopwords and OR-joins the
remaining content tokens as phrase literals.
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")  # retrieval module imports numpy indirectly

from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


# ---------------------------------------------------------------------------
# _sanitize_fts_query — unit tests (no DB required)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query,expected_tokens",
    [
        # stopwords dropped
        ("what happened with the deployment rollback", {"happened", "deployment", "rollback"}),
        # single content word passes through
        ("compaction", {"compaction"}),
        # all stopwords → falls back to raw
        ("the and of", None),  # None = sentinel for fallback-to-raw
        # empty string → empty output
        ("", ""),
        # FTS5 operator characters stripped
        ("context: length-probe", {"context", "lengthprobe"}),
        # trailing punctuation stripped by tokenizer
        ("hello, world!", {"hello", "world"}),
    ],
)
def test_sanitize_fts_query_extracts_content_tokens(query, expected_tokens):
    result = FactRetriever._sanitize_fts_query(query)

    if expected_tokens == "":
        assert result == ""
        return

    if expected_tokens is None:
        # Pathological case: all stopwords — should fall back to raw query
        assert result == query
        return

    # OR-joined phrase literals: `"tok1" OR "tok2" OR ...`
    # Extract the tokens between quotes, order-independent.
    import re
    matches = re.findall(r'"([^"]+)"', result)
    assert set(matches) == expected_tokens, f"got {result!r}"


# ---------------------------------------------------------------------------
# Integration test — actually run _fts_candidates against an in-memory DB
# ---------------------------------------------------------------------------

@pytest.fixture
def retriever_with_facts(tmp_path):
    """MemoryStore seeded with a few facts for retrieval tests."""
    db_path = tmp_path / "test_facts.db"
    store = MemoryStore(str(db_path))
    store.add_fact(
        content="The Thursday deployment rollback failed because of stale migration state.",
        category="project",
    )
    store.add_fact(
        content="Compaction settings tuned to 0.85 threshold.",
        category="tool",
    )
    store.add_fact(
        content="Venice.ai advertises availableContextTokens inside model_spec.",
        category="tool",
    )
    retriever = FactRetriever(store=store)
    yield retriever
    store.close()


def test_prefetch_recovers_prose_query(retriever_with_facts):
    """A natural-language query should now match the relevant fact.

    Before the sanitizer fix, 'what happened with the deployment rollback'
    returned zero hits because FTS5 required every token to co-occur.
    """
    results = retriever_with_facts.search(
        "what happened with the deployment rollback"
    )
    assert len(results) >= 1
    # The top hit should be the deployment rollback fact
    assert "deployment rollback" in results[0]["content"].lower()




# ---------------------------------------------------------------------------
# Loop-invariant encode hoists (perf) — search/probe/related must encode
# constant vectors ONCE per call, not once per candidate/row.
# encode_text/encode_atom are deterministic (SHA-256 counter blocks), so the
# hoisted vectors are bit-identical to the per-iteration values they replace.
# ---------------------------------------------------------------------------

from plugins.memory.holographic import holographic as hrr


@pytest.fixture
def hoisted_retriever(tmp_path):
    """30 facts with HRR vectors, default dim (smaller dims trip an
    inhomogeneous-shape edge in the fact encoder).

    NOTE: a real tmp_path db, NOT ":memory:" — MemoryStore resolves the
    path and shares one process-wide connection per file, so ":memory:"
    becomes a literal ./:memory: file that leaks state across runs (and
    the NULL-vector test below would permanently corrupt it)."""
    store = MemoryStore(str(tmp_path / "hoist_store.db"))
    for i in range(30):
        store.add_fact(
            content=f"deploy target {i} setting alpha beta gamma option {i % 7}",
            category="fact" if i % 2 else "preference",
            tags=f"entity_{i % 5} deploy",
        )
    retriever = FactRetriever(store=store)
    yield retriever
    store.close()


def _counting_spy(monkeypatch, attr):
    calls = []
    real = getattr(hrr, attr)

    def wrapper(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(hrr, attr, wrapper)
    return calls


def test_encode_functions_are_deterministic():
    """Soundness premise of the hoists: same input -> identical vector."""
    import numpy as np

    assert np.array_equal(hrr.encode_text("deploy target", 1024),
                          hrr.encode_text("deploy target", 1024))
    assert np.array_equal(hrr.encode_atom("__hrr_role_content__", 1024),
                          hrr.encode_atom("__hrr_role_content__", 1024))


def test_search_encodes_query_vector_once(hoisted_retriever, monkeypatch):
    calls = _counting_spy(monkeypatch, "encode_text")
    results = hoisted_retriever.search("deploy target setting")
    assert results  # the HRR path actually engaged
    assert len(calls) == 1, (
        f"query vector encoded {len(calls)}x in one search() — "
        "loop-invariant hoist regressed"
    )


def test_search_results_bit_identical_to_unhoisted(hoisted_retriever):
    """Parity: the lazy hoist must not change search() results.

    FORK DIVERGENCE (see the note in FactRetriever.search): upstream's version
    of this test builds its reference with an UNBOUND ``encode_text(query)``,
    because upstream's hoist is a pure performance change and bit-identity to
    the pre-hoist loop is exactly what it should assert. This fork's
    9abe585d2 deliberately changed that semantics — a fact's content component
    is ``bind(encode_text(content), ROLE_CONTENT)``, so comparing an unbound
    query against it is quasi-orthogonal by construction and made the HRR term
    near-constant noise. Asserting the unbound reference here would re-assert
    the bug that commit fixed.

    So the reference below is rebuilt with the BOUND query vector. The test's
    actual job is unchanged and still enforced: encoding the query once,
    lazily, must give exactly what encoding it per-candidate gives.
    """
    r = hoisted_retriever
    query = "deploy target setting"
    new_results = r.search(query)

    # --- pre-fix reference ---
    candidates = r._fts_candidates(query, None, 0.3, 10 * 3)
    query_tokens = r._tokenize(query)
    scored = []
    for fact in candidates:
        content_tokens = r._tokenize(fact["content"])
        tag_tokens = r._tokenize(fact.get("tags", ""))
        all_tokens = content_tokens | tag_tokens
        jaccard = r._jaccard_similarity(query_tokens, all_tokens)
        fts_score = fact.get("fts_rank", 0.0)
        if r.hrr_weight > 0 and fact.get("hrr_vector"):
            fact_vec = hrr.bytes_to_phases(fact["hrr_vector"], dim=r.hrr_dim)
            # Per-candidate, and BOUND to ROLE_CONTENT to match search().
            role_content = hrr.encode_atom("__hrr_role_content__", r.hrr_dim)
            query_vec = hrr.bind(hrr.encode_text(query, r.hrr_dim), role_content)
            hrr_sim = (hrr.similarity(query_vec, fact_vec) + 1.0) / 2.0
        else:
            hrr_sim = 0.5
        relevance = (r.fts_weight * fts_score
                     + r.jaccard_weight * jaccard
                     + r.hrr_weight * hrr_sim)
        fact["score"] = relevance * fact["trust_score"]
        scored.append(fact)
    scored.sort(key=lambda x: x["score"], reverse=True)
    old_results = scored[:10]
    for fact in old_results:
        fact.pop("hrr_vector", None)

    assert new_results == old_results


def test_related_encodes_role_atoms_once(hoisted_retriever, monkeypatch):
    calls = _counting_spy(monkeypatch, "encode_atom")
    results = hoisted_retriever.related("entity_1")
    assert results
    role_calls = [a for a in calls
                  if a and str(a[0]).startswith("__hrr_role_")]
    assert len(role_calls) == 2, (
        f"role atoms encoded {len(role_calls)}x in one related() — "
        "expected exactly 2 (role_entity + role_content, hoisted)"
    )


def test_probe_encodes_role_atom_once(hoisted_retriever, monkeypatch):
    """FORK DIVERGENCE: expected count is 0, not upstream's 1.

    Upstream hoists a role_content atom out of probe()'s per-fact loop, so it
    asserts exactly one encode (the anti-regression being "back to once per
    row"). This fork's 9abe585d2 replaced probe()'s scoring with a direct
    bundle-membership test against probe_key, which removed the only consumer
    of that atom — so the correct count here is now zero, and the hoist was
    dropped rather than left dead.

    Kept as == 0 rather than <= 1 on purpose: a count of 1 means someone
    reintroduced the residual-vs-content_vec comparison this fork removed.
    """
    calls = _counting_spy(monkeypatch, "encode_atom")
    results = hoisted_retriever.probe("entity_1")
    assert results
    role_content_calls = [a for a in calls
                          if a and a[0] == "__hrr_role_content__"]
    assert len(role_content_calls) == 0, (
        f"role_content atom encoded {len(role_content_calls)}x in one "
        "probe() — the bundle-membership fix should need it zero times"
    )


def test_search_without_vectors_never_encodes(hoisted_retriever, monkeypatch):
    """Migrated DBs can have FTS candidates with NULL hrr_vector
    (MemoryStore._init_db adds the column without backfilling existing
    facts). The lazy hoist must not encode a query vector nothing will
    use — pre-fix main encoded only beneath fact.get('hrr_vector')."""
    store = hoisted_retriever.store
    store._conn.execute("UPDATE facts SET hrr_vector = NULL")
    store._conn.commit()
    calls = _counting_spy(monkeypatch, "encode_text")
    results = hoisted_retriever.search("deploy target setting")
    assert results  # candidates exist; neutral hrr_sim=0.5 path
    assert calls == [], (
        f"encode_text called {len(calls)}x with zero vector candidates — "
        "lazy hoist regressed to eager"
    )
