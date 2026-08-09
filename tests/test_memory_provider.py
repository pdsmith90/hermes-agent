

class TestCronMemoryProviderToolsSurvive:
    """External provider tools must survive cron's built-in-memory disable.

    REGRESSION 2026-08-09. ``_resolve_cron_disabled_toolsets`` disables the
    "memory" toolset in cron so the model is not offered an unbacked
    ``memory()`` under skip_memory (upstream 03dc4aad5). But "memory" was also
    the kill switch in ``memory_provider_tools_enabled``, so that one entry
    stripped fact_store/fact_feedback — the ONLY memory tools a cron job has.
    Every fact-writer then ran to completion writing nothing, and one fabricated
    the fact_ids in its report rather than reporting the failure.
    """

    def test_cron_disable_of_builtin_memory_keeps_provider_tools(self):
        from agent.memory_manager import memory_provider_tools_enabled
        cron_disabled = ["cronjob", "messaging", "clarify", "memory"]
        assert memory_provider_tools_enabled(None, cron_disabled) is True, (
            "hiding the built-in memory tool must not strip fact_store"
        )

    def test_explicit_provider_optout_still_disables(self):
        from agent.memory_manager import (
            memory_provider_tools_enabled,
            MEMORY_PROVIDER_TOOLSET,
        )
        assert memory_provider_tools_enabled(
            None, ["memory", MEMORY_PROVIDER_TOOLSET]
        ) is False

    def test_real_cron_toolsets_expose_provider_tools(self):
        """Guard the actual resolver, not just a hand-written list."""
        from cron.scheduler import _resolve_cron_disabled_toolsets
        from agent.memory_manager import memory_provider_tools_enabled
        disabled = _resolve_cron_disabled_toolsets({})
        assert "memory" in disabled, "upstream intent (hide built-in) changed"
        assert memory_provider_tools_enabled(None, disabled) is True
