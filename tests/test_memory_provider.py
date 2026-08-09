

class TestCronMemoryProviderToolsSurvive:
    """Cron must not disable the "memory" toolset — it also gates fact_store.

    REGRESSION 2026-08-09. Upstream 03dc4aad5 added "memory" to
    _resolve_cron_disabled_toolsets to hide the unbacked built-in memory() from
    cron. But memory_provider_tools_enabled treats that same name as an
    explicit disable of the EXTERNAL provider tools, and fact_store /
    fact_feedback are the only memory tools a cron job has. Every fact-writer
    then ran to completion writing nothing, one improvised raw SQL that
    corrupted the fact_id sequence, and one fabricated the fact_ids in its
    report.

    The gate is NOT the place to fix this: "an explicit memory disable wins" is
    a user-facing contract pinned by tests/agent/test_memory_provider.py.
    """

    def test_cron_denylist_does_not_contain_memory(self):
        from cron.scheduler import _resolve_cron_disabled_toolsets
        assert "memory" not in _resolve_cron_disabled_toolsets({}), (
            "adding 'memory' here silently removes fact_store from every cron agent"
        )

    def test_real_cron_toolsets_expose_provider_tools(self):
        """Guard the actual resolver, not a hand-written list."""
        from cron.scheduler import _resolve_cron_disabled_toolsets
        from agent.memory_manager import memory_provider_tools_enabled
        assert memory_provider_tools_enabled(
            None, _resolve_cron_disabled_toolsets({})
        ) is True

    def test_explicit_user_disable_still_wins(self):
        """The upstream contract must remain intact: disabling memory disables providers."""
        from agent.memory_manager import memory_provider_tools_enabled
        assert memory_provider_tools_enabled(None, ["memory"]) is False
