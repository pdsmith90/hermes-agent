"""Behavior contract for the cron.allow_agent_scheduling config gate.

``_resolve_cron_disabled_toolsets`` decides which toolsets a cron-spawned
agent must never receive. Historically ``cronjob`` was hard-denied there as
loop-prevention policy. The ``cron.allow_agent_scheduling`` gate (config.yaml,
default off) makes that denial opt-out-able:

  - gate off / absent: byte-exact current behavior — ``cronjob`` denied.
  - gate on: ``cronjob`` dropped from the base denylist; ``messaging`` and
    ``clarify`` (interactivity constraints) are ALWAYS denied regardless of
    the gate.
  - user-level ``agent.disabled_toolsets`` still layers on top, so a user who
    denies ``cronjob`` globally keeps it denied even with the gate on
    (per-job enabled_toolsets can never widen past the config denylist).

Fork divergence from upstream's version of this contract:

  - ``memory`` is NOT in the base denylist. This began as a fork divergence:
    the upstream entry was reverted here because memory_provider_tools_enabled
    treats it as an explicit disable of the EXTERNAL provider tools too, which
    silently removed fact_store from every cron agent (2026-08-09). As of the
    2026-08-28 upstream sync this is no longer a divergence — upstream reached
    the same position ("cron agents get memory like any other agent run"). See
    the NOTE in ``_resolve_cron_disabled_toolsets`` and TestCronDisabledToolsets
    in test_scheduler.py.
  - ``code_execution`` is appended whenever approvals.cron_mode resolves to
    "deny" (the shipped default), so the exact-list assertions pin it. The
    fixture below pins the approval mode so these tests don't float on the
    machine's live config.
"""

from unittest.mock import patch

import pytest

from cron.scheduler import _resolve_cron_disabled_toolsets


# The toolsets that must be denied in cron context no matter what the
# agent-scheduling gate says: messaging/clarify are interactive-only.
# (``memory`` is absent on both sides since the 2026-08-28 upstream sync —
# upstream converged on this fork's position; see the module docstring.)
ALWAYS_DISABLED = ["messaging", "clarify"]

# The base denylist with the gate off, under cron_mode=deny (pinned below).
GATE_OFF_BASE = ["cronjob", "messaging", "clarify", "code_execution"]


@pytest.fixture(autouse=True)
def _pin_cron_approval_mode_deny():
    """Pin approvals.cron_mode=deny so exact-list assertions are stable."""
    with patch("tools.approval._get_cron_approval_mode", return_value="deny"):
        yield


class TestGateOffDefault:
    def test_empty_config_denies_cronjob(self):
        assert _resolve_cron_disabled_toolsets({}) == GATE_OFF_BASE

    def test_none_config_denies_cronjob(self):
        assert _resolve_cron_disabled_toolsets(None) == GATE_OFF_BASE

    def test_cron_section_present_but_gate_absent(self):
        cfg = {"cron": {"preflight": True}}
        assert _resolve_cron_disabled_toolsets(cfg) == GATE_OFF_BASE

    def test_explicit_false_matches_default(self):
        cfg = {"cron": {"allow_agent_scheduling": False}}
        assert _resolve_cron_disabled_toolsets(cfg) == \
            _resolve_cron_disabled_toolsets({})

    @pytest.mark.parametrize("falsy", [False, None, "", 0])
    def test_falsy_values_keep_gate_off(self, falsy):
        cfg = {"cron": {"allow_agent_scheduling": falsy}}
        disabled = _resolve_cron_disabled_toolsets(cfg)
        assert "cronjob" in disabled


class TestGateOn:
    def test_cronjob_dropped_from_denylist(self):
        cfg = {"cron": {"allow_agent_scheduling": True}}
        disabled = _resolve_cron_disabled_toolsets(cfg)
        assert "cronjob" not in disabled

    def test_interactivity_denials_survive_the_gate(self):
        cfg = {"cron": {"allow_agent_scheduling": True}}
        disabled = _resolve_cron_disabled_toolsets(cfg)
        for name in ALWAYS_DISABLED:
            assert name in disabled

    def test_memory_not_denied(self):
        # Cron agents run with memory enabled like any other agent run
        # (skip_memory=False); the toolset must not be policy-denied.
        for cfg in ({}, {"cron": {"allow_agent_scheduling": True}}):
            assert "memory" not in _resolve_cron_disabled_toolsets(cfg)

    def test_user_denylist_wins_over_gate(self):
        # A user who denies cronjob in agent.disabled_toolsets keeps it
        # denied even with the gate on — the gate only removes the built-in
        # policy denial, never the user's own config denylist.
        cfg = {
            "cron": {"allow_agent_scheduling": True},
            "agent": {"disabled_toolsets": ["cronjob"]},
        }
        assert "cronjob" in _resolve_cron_disabled_toolsets(cfg)

    def test_unrelated_user_denylist_layers_without_reviving_cronjob(self):
        cfg = {
            "cron": {"allow_agent_scheduling": True},
            "agent": {"disabled_toolsets": ["browser"]},
        }
        disabled = _resolve_cron_disabled_toolsets(cfg)
        assert "browser" in disabled
        assert "cronjob" not in disabled


class TestUserLayerUnchanged:
    def test_user_denylist_still_layers_when_gate_off(self):
        cfg = {"agent": {"disabled_toolsets": ["browser", "cronjob"]}}
        disabled = _resolve_cron_disabled_toolsets(cfg)
        assert "browser" in disabled
        # No duplicate when the user names an already-denied toolset.
        assert disabled.count("cronjob") == 1

    def test_user_can_still_deny_memory_for_cron(self):
        # Memory is no longer policy-denied, but a user-level denylist
        # entry still applies to cron runs.
        cfg = {"agent": {"disabled_toolsets": ["memory"]}}
        assert "memory" in _resolve_cron_disabled_toolsets(cfg)

    def test_blank_and_whitespace_entries_ignored(self):
        cfg = {
            "cron": {"allow_agent_scheduling": True},
            "agent": {"disabled_toolsets": ["", "  ", "browser"]},
        }
        disabled = _resolve_cron_disabled_toolsets(cfg)
        assert "browser" in disabled
        assert "" not in disabled
