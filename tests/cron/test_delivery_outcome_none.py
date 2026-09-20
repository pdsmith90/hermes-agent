"""``deliver: none`` resolves to zero delivery targets (like ``local``), so a run on such a job
must not be recorded as ``delivered`` (2026-09-20: every file-only job read as delivered)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron.scheduler import _classify_delivery_outcome


def _classify(lane, **kw):
    args = dict(delivery_error=None, should_deliver=True, unresolved_origin=False,
                normalized_deliver=lane, incident_acked=False, success=True)
    args.update(kw)
    return _classify_delivery_outcome(**args)


def test_none_lane_is_suppressed_not_delivered():
    assert _classify("none") == "suppressed"


def test_local_lane_is_suppressed():
    assert _classify("local") == "suppressed"


def test_origin_and_platform_lanes_are_delivered():
    assert _classify("origin") == "delivered"
    assert _classify("telegram") == "delivered"


def test_failure_on_none_lane_with_acked_incident_reads_suppressed_acked():
    assert _classify("none", incident_acked=True, success=False) == "suppressed_acked"
