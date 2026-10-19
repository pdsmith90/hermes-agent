"""Fork: cron and subagent agents load the external memory provider despite ``skip_memory=True``.

Cron agents run with ``skip_memory=True`` so they never touch the built-in MEMORY.md store (a
rendered view on this fork), yet ``fact_store``/``fact_feedback`` come from the external provider,
so ``agent_init._init_memory`` still loads it for the ``cron`` and ``subagent`` platforms. Every
other ``skip_memory`` agent (background review fork, ``/refine`` and side agents, curator) keeps
upstream's gate and builds no provider.
"""
import pytest


@pytest.mark.parametrize("platform, skip_memory, expected", [
    ("cron", True, ["fake"]),
    ("subagent", True, ["fake"]),
    ("cli", False, ["fake"]),
    ("cli", True, []),
    ("telegram", True, []),
    ("curator", True, []),
])
def test_provider_load_follows_the_platform_when_memory_is_skipped(
        tmp_path, monkeypatch, platform, skip_memory, expected):
    from agent import agent_init, safe_worker_policy
    import plugins.memory
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(safe_worker_policy, "safe_worker_enabled", lambda: False)
    loaded = []

    class _Provider:
        name = "fake"

        def is_available(self):
            return True

    def load_provider(name):
        loaded.append(name)
        return _Provider()

    monkeypatch.setattr(plugins.memory, "load_memory_provider", load_provider)
    monkeypatch.setattr(agent_init, "is_core_memory_provider", lambda name: False)
    agent = object.__new__(AIAgent)
    agent.session_id = "provider-gate"
    agent.valid_tool_names = set()
    agent.enabled_toolsets = agent.disabled_toolsets = None

    agent_init._init_memory(agent, {"memory": {"provider": "fake"}}, skip_memory, platform)

    assert loaded == expected
