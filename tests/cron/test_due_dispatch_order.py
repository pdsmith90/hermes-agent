"""Due-scan dispatch ordering: scheduled order, not jobs.json file order.

``_get_due_jobs_locked`` accumulates due jobs by iterating ``jobs.json`` and
appending, so the returned list used to carry *file* order.  On a healthy tick
that is invisible — one job comes due at a time.  It matters on a catch-up
tick, when many jobs come due at once (gateway down, host asleep or in
gaming-mode, or a long overrun backing the night up): with
``cron.max_parallel_jobs: 1`` the scheduler serialises the whole due set onto a
single-worker pool, so the order of this list *is* execution order.  File order
is arbitrary, so a 03:50 job sitting above a 03:15 job in jobs.json ran first,
inverting the intended night sequence.

These tests pin the ordering contract:
  - a mixed backlog is returned in scheduled order regardless of file order
  - ordering holds across BOTH due branches (within grace and past grace)
  - a legacy naive timestamp mixes with aware ones without a comparison crash
  - ordering compares parsed instants, not ISO strings (mixed UTC offsets)
  - an unplaceable ``next_run_at`` sorts last and keeps its relative file order
"""

import pytest
from datetime import datetime, timedelta, timezone

from cron.jobs import (
    get_due_jobs,
    save_jobs,
    _compute_grace_seconds,
    _due_dispatch_order_key,
)

# Pinned "now": deliberately mid-morning so a backed-up night sits behind it.
FIXED_NOW = datetime(2026, 6, 22, 4, 30, 0, tzinfo=timezone.utc)

DAILY_GRACE = _compute_grace_seconds({"kind": "cron", "expr": "5 2 * * *"})


@pytest.fixture()
def cron_store(tmp_path, monkeypatch):
    """Redirect cron storage to a temp dir and pin the clock.

    Re-points the module constants (the documented process-wide compatibility
    surface honoured by ``_current_cron_store``) *and* HERMES_HOME, so the
    past-grace branch's ``record_catch_up_occurrence()`` writes into tmp rather
    than the caller's real store.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: FIXED_NOW)
    return tmp_path


def _daily(jid, expr, next_run_at):
    """A recurring daily cron job parked at an explicit ``next_run_at``.

    ``next_run_at`` must be a genuine occurrence of ``expr`` or the due scan's
    stale-schedule guard re-anchors the record instead of firing it.
    """
    return {
        "id": jid,
        "name": jid,
        "prompt": "x",
        "schedule": {"kind": "cron", "expr": expr, "display": expr},
        "repeat": {"times": None, "completed": 3},
        "enabled": True,
        "state": "scheduled",
        "next_run_at": next_run_at,
        "last_run_at": None,
        "deliver": "none",
        "workdir": None,
        "fire_claim": None,
    }


def _at(hh, mm, day=22, offset_hours=0):
    """An ISO timestamp on 2026-06-<day> at hh:mm in the given UTC offset."""
    tz = timezone(timedelta(hours=offset_hours))
    return datetime(2026, 6, day, hh, mm, 0, tzinfo=tz).isoformat()


class TestDueDispatchOrder:
    def test_backlog_returned_in_scheduled_order_not_file_order(self, cron_store):
        """The real inversion: dream (03:50) sits above consolidate (03:15)."""
        # File order deliberately scrambled, mirroring the production jobs.json
        # where dream-and-promote is index 0 and consolidate-synthesize index 1.
        save_jobs([
            _daily("dream", "50 3 * * *", _at(3, 50)),
            _daily("consolidate", "15 3 * * *", _at(3, 15)),
            _daily("review", "5 2 * * *", _at(2, 5)),
            _daily("ingest", "20 2 * * *", _at(2, 20)),
            _daily("arxiv", "50 2 * * *", _at(2, 50)),
        ])

        due = get_due_jobs()

        assert [d["id"] for d in due] == [
            "review",       # 02:05
            "ingest",       # 02:20
            "arxiv",        # 02:50
            "consolidate",  # 03:15
            "dream",        # 03:50
        ]

    def test_order_holds_across_grace_and_past_grace_branches(self, cron_store):
        """Both due branches feed one list; the sort must span them.

        At 04:30 with a 2h grace, 03:15 is a within-grace catch-up while 02:05
        is past grace (fast-forwarded, then fired once).  File order puts the
        past-grace job second so a branch-local sort would not fix it.
        """
        early = _at(2, 5)    # past grace  -> fast-forward branch
        late = _at(3, 15)    # within grace -> ordinary catch-up
        save_jobs([_daily("late", "15 3 * * *", late), _daily("early", "5 2 * * *", early)])

        # Guard the premise: these really are different branches.
        assert (FIXED_NOW - datetime.fromisoformat(early)).total_seconds() > DAILY_GRACE
        assert (FIXED_NOW - datetime.fromisoformat(late)).total_seconds() < DAILY_GRACE

        due = get_due_jobs()

        assert [d["id"] for d in due] == ["early", "late"]

    def test_mixed_naive_and_aware_timestamps_sort_without_crashing(self, cron_store):
        """A legacy naive ``next_run_at`` alongside an aware one must not crash.

        Naive timestamps are explicitly supported (``_ensure_aware`` interprets
        them as system-local wall time) and, unlike a foreign UTC offset, they
        are NOT re-anchored upstream — ``_timezone_offset_mismatch`` returns
        False for them by design — so a mixed due set genuinely reaches the
        sort.  Comparing them with bare ``datetime.fromisoformat`` would raise
        "can't compare offset-naive and offset-aware datetimes" and take the
        whole tick down; the key normalises both first.
        """
        naive = "2026-06-22T02:05:00"          # no offset at all
        aware = _at(3, 15)                     # 03:15+00:00

        save_jobs([
            _daily("consolidate", "15 3 * * *", aware),
            _daily("review", "5 2 * * *", naive),
        ])

        due = get_due_jobs()

        assert [d["id"] for d in due] == ["review", "consolidate"]

    def test_orders_by_instant_not_iso_string(self):
        """Mixed UTC offsets: lexical order on the raw strings is wrong.

        Key-level rather than end-to-end on purpose.  A *cron* record carrying
        a non-local offset never reaches the sort: the due scan's stale-schedule
        guard compares the ``_ensure_aware``-normalised instant against the
        expression, sees a mismatch, and re-anchors the record without firing
        it.  The ordering property is still the key's to hold — interval jobs
        skip that cron-only guard, and it costs nothing to be right here.
        """
        # 2026-06-22T02:05:00+05:00 == 2026-06-21T21:05Z  (earlier instant)
        shifted = _at(2, 5, offset_hours=5)
        # 2026-06-21T22:00:00+00:00 == 2026-06-21T22:00Z  (later instant)
        utc = _at(22, 0, day=21)

        assert utc < shifted, "premise: lexical order is the reverse of real order"

        key_shifted = _due_dispatch_order_key(_daily("s", "5 2 * * *", shifted), FIXED_NOW)
        key_utc = _due_dispatch_order_key(_daily("u", "0 22 * * *", utc), FIXED_NOW)

        assert key_shifted < key_utc, "must order by instant, not by ISO string"

    def test_unplaceable_next_run_at_sorts_last_and_ties(self):
        """A record we cannot place must never jump the queue.

        Exercised against the real key rather than through ``get_due_jobs``:
        the scan repairs an unparseable ``next_run_at`` upstream by re-arming
        the record into the future, so this bucket is unreachable end-to-end.
        It is the sort's fail-safe, and it is still worth pinning.
        """
        placeable = _due_dispatch_order_key(
            _daily("ok", "5 2 * * *", _at(2, 5)), FIXED_NOW
        )
        assert placeable[0] == 0

        previous = None
        for bad in (None, "", "not-a-timestamp", "2026-13-45T99:99:99", 12345, {}):
            job = dict(_daily("bad", "5 2 * * *", _at(2, 5)), next_run_at=bad)
            key = _due_dispatch_order_key(job, FIXED_NOW)
            assert key[0] == 1, f"{bad!r} should be unplaceable"
            assert placeable < key, f"{bad!r} must sort after a placeable record"
            # Ties, so a stable sort preserves file order among unplaceables.
            if previous is not None:
                assert key == previous
            previous = key
