"""Per-action argument validation for the ``fact_store`` tool.

``FACT_STORE_SCHEMA`` declares ``required: ["action"]`` and nothing else: nine
actions share one flat property bag, so which argument pairs with which action
lives only in prose. Enforcement used to be a bare ``except KeyError`` that
reported the missing key and nothing more — "Missing required argument:
'entity'" tells a model neither which action it got wrong nor what to send
instead, so the same malformed shapes recurred nightly in cron.

The three shapes below are real failing calls observed in unattended runs, in
frequency order. Two different models produced them, so this is a contract
defect rather than a model quirk:

  {"action": "update",  "content": ..., "trust_delta": ...}   # no fact_id
  {"action": "search",  "entity": ...,  "limit": ...}         # query/entity mixup
  {"action": "related", "fact_id": ..., "limit": ...}         # entity/fact_id mixup

Each error must name the action, name the missing argument, echo what was
supplied, and point at the action that *would* have worked.
"""

import json

import pytest

from plugins.memory.holographic import HolographicMemoryProvider


@pytest.fixture
def provider(tmp_path):
    p = HolographicMemoryProvider(
        config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 64}
    )
    p.initialize(session_id="test-session")
    yield p
    p.shutdown()


def _err(provider, args):
    result = json.loads(provider._handle_fact_store(args))
    assert "error" in result, f"expected an error for {args}, got {result}"
    return result["error"]


# -- the three real-world malformed shapes ---------------------------------


def test_update_without_fact_id_explains_how_to_get_one(provider):
    """The most frequent shape: a revision with no id to revise."""
    err = _err(
        provider,
        {"action": "update", "content": "revised", "trust_delta": 0.1, "category": "paper"},
    )
    assert "update" in err
    assert "fact_id" in err
    # Must say where a fact_id comes from — there is no update-by-content path,
    # which is precisely why the model kept improvising this call.
    assert "search" in err or "probe" in err


def test_search_with_entity_points_at_probe(provider):
    """search takes free text; entity lookups belong to probe/related."""
    err = _err(provider, {"action": "search", "entity": "Postgres", "limit": 5})
    assert "search" in err
    assert "query" in err
    assert "probe" in err, "should redirect entity lookups to probe/related"
    assert "entity" in err, "should acknowledge the argument that was supplied"


def test_related_with_fact_id_explains_entity_is_a_name(provider):
    """A document-ingestion failure observed verbatim in an unattended run."""
    err = _err(provider, {"action": "related", "fact_id": 697, "limit": 5})
    assert "related" in err
    assert "entity" in err
    assert "fact_id" in err, "should name the wrong argument the model supplied"


def test_probe_without_entity(provider):
    err = _err(provider, {"action": "probe", "limit": 5})
    assert "probe" in err and "entity" in err


def test_add_without_content(provider):
    err = _err(provider, {"action": "add", "category": "paper", "tags": "x"})
    assert "add" in err and "content" in err


def test_remove_without_fact_id(provider):
    err = _err(provider, {"action": "remove"})
    assert "remove" in err and "fact_id" in err


@pytest.mark.parametrize("entities", [None, [], ""])
def test_reason_requires_a_non_empty_entity_list(provider, entities):
    args = {"action": "reason"}
    if entities is not None:
        args["entities"] = entities
    err = _err(provider, args)
    assert "reason" in err and "entities" in err


def test_error_names_the_action_not_just_the_key(provider):
    """Regression guard on the old message: a bare key name is not enough."""
    err = _err(provider, {"action": "related", "fact_id": 697})
    assert err != "Missing required argument: 'entity'"


# -- no regression on valid calls ------------------------------------------


def test_valid_calls_still_work(provider):
    added = json.loads(
        provider._handle_fact_store(
            {"action": "add", "content": "Postgres uses MVCC for isolation.",
             "category": "paper", "tags": "database,concurrency"}
        )
    )
    fact_id = added["fact_id"]
    assert added["status"] == "added"

    for args in (
        {"action": "search", "query": "isolation"},
        {"action": "probe", "entity": "Postgres"},
        {"action": "related", "entity": "Postgres"},
        {"action": "reason", "entities": ["Postgres"]},
        {"action": "contradict"},
        {"action": "list"},
        {"action": "update", "fact_id": fact_id, "trust_delta": 0.1},
        {"action": "remove", "fact_id": fact_id},
    ):
        result = json.loads(provider._handle_fact_store(args))
        assert "error" not in result, f"{args} regressed: {result}"


def test_actions_with_no_required_args_are_not_blocked(provider):
    for action in ("contradict", "list"):
        result = json.loads(provider._handle_fact_store({"action": action}))
        assert "error" not in result


def test_unknown_action_still_reported(provider):
    err = _err(provider, {"action": "frobnicate"})
    assert "frobnicate" in err


def test_category_schema_does_not_contradict_what_add_accepts(provider):
    """'add' stores category verbatim, so the schema must not claim otherwise.

    The enum listed user_pref/project/tool/general, but callers routinely
    invent labels (paper, researched, lesson, hypothesis, ...) and 'add' stores
    them as given — leaving the schema declaring most stored rows invalid.
    """
    from plugins.memory.holographic import FACT_STORE_SCHEMA

    category = FACT_STORE_SCHEMA["parameters"]["properties"]["category"]
    assert "enum" not in category, (
        "category is free-form at the storage layer; an enum here re-creates "
        "the schema/behaviour drift this pins"
    )

    for name in ("paper", "researched", "open-question", "user_pref"):
        result = json.loads(
            provider._handle_fact_store(
                {"action": "add", "content": f"fact about {name}", "category": name}
            )
        )
        assert result["status"] == "added"
        stored = json.loads(
            provider._handle_fact_store({"action": "list", "category": name})
        )
        assert stored["count"] == 1, f"category {name!r} did not round-trip"


# -- action=get: exact lookup by id (the verification path) ----------------
#
# Ranked search cannot prove a fact is ABSENT — a low-ranked or sub-min_trust
# row simply doesn't surface, which made the morning briefing's write
# spot-checks report false "claimed write did not land" flags (2026-08-04:
# fids 747/755 flagged missing while sitting in the table). 'get' is the
# side-effect-free existence check those spot-checks need.


def test_get_without_fact_id_is_rejected_with_lookup_hint(provider):
    err = _err(provider, {"action": "get"})
    assert "get" in err and "fact_id" in err


def test_get_returns_the_exact_fact(provider):
    added = json.loads(provider._handle_fact_store(
        {"action": "add", "content": "Postgres runs on port 5432",
         "category": "project", "tags": "infra"}
    ))
    fid = added["fact_id"]
    result = json.loads(provider._handle_fact_store({"action": "get", "fact_id": fid}))
    assert result["found"] is True
    assert result["fact"]["fact_id"] == fid
    assert result["fact"]["content"] == "Postgres runs on port 5432"
    assert result["fact"]["category"] == "project"


def test_get_missing_fact_reports_found_false_not_error(provider):
    """Absence is a result the caller acts on, not a tool failure."""
    result = json.loads(provider._handle_fact_store({"action": "get", "fact_id": 999999}))
    assert "error" not in result
    assert result["found"] is False
    assert result["fact"] is None


def test_get_ignores_the_search_trust_floor(provider):
    """A decayed fact is invisible to default search but must still 'get'."""
    added = json.loads(provider._handle_fact_store(
        {"action": "add", "content": "decayed zombie fact", "category": "hypothesis"}
    ))
    fid = added["fact_id"]
    provider._handle_fact_store({"action": "update", "fact_id": fid, "trust_delta": -0.4})
    result = json.loads(provider._handle_fact_store({"action": "get", "fact_id": fid}))
    assert result["found"] is True
    assert result["fact"]["trust_score"] < 0.3
