"""Regression test for the ``HermesMCPOAuthProvider.async_auth_flow`` bidirectional
generator bridge.

PR #11383 introduced a subclass method that wrapped the SDK's ``auth_flow`` with::

    async for item in super().async_auth_flow(request):
        yield item

``httpx``'s auth_flow contract is a **bidirectional** async generator — the
driving code (``httpx._client._send_handling_auth``) does::

    next_request = await auth_flow.asend(response)

to feed HTTP responses back into the generator. The naive ``async for ...``
wrapper discards those ``.asend(response)`` values and resumes the inner
generator with ``None``, so the SDK's ``response = yield request`` branch in
``mcp/client/auth/oauth2.py`` sees ``response = None`` and crashes at
``if response.status_code == 401`` with
``AttributeError: 'NoneType' object has no attribute 'status_code'``.

This broke every OAuth MCP server on the first HTTP response regardless of
status code. The reason nothing caught it in CI: zero existing tests drive
the full ``.asend()`` round-trip — the integration tests in
``test_mcp_oauth_integration.py`` stop at ``_initialize()`` and disk-watching.

These tests drive the wrapper through a manual ``.asend()`` sequence to prove
the bridge forwards responses correctly into the inner SDK generator.
"""
from __future__ import annotations

import pytest


pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required")


@pytest.mark.asyncio
async def test_hermes_provider_forwards_asend_values(tmp_path, monkeypatch):
    """The wrapper MUST forward ``.asend(response)`` into the inner generator.

    This is the primary regression test. With the broken wrapper, the inner
    SDK generator sees ``response = None`` and raises ``AttributeError`` at
    ``oauth2.py:505``. With the correct bridge, a 200 response finishes the
    flow cleanly (``StopAsyncIteration``).
    """
    # The SDK's httpx flavour (httpx2 on mcp >= 2.0): the provider is an
    # Auth subclass from that module and its auth_flow only accepts its own
    # Request/Response types.
    from tools.mcp_tool import sdk_httpx
    httpx = sdk_httpx()
    from mcp.shared.auth import OAuthClientMetadata, OAuthToken
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests

    assert _HERMES_PROVIDER_CLS is not None, "SDK OAuth types must be available"

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    # Seed a valid-looking token so the SDK's _initialize loads something and
    # can_refresh_token() is True (though we don't exercise refresh here — we
    # go straight through the 200 path).
    storage = HermesTokenStorage("srv")
    await storage.set_tokens(
        OAuthToken(
            access_token="old_access",
            token_type="Bearer",
            expires_in=3600,
            refresh_token="old_refresh",
        )
    )
    # Also seed client_info so the SDK doesn't attempt registration.
    from mcp.shared.auth import OAuthClientInformationFull

    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="test-client",
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
        )
    )

    metadata = OAuthClientMetadata(
        redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
        client_name="Hermes Agent",
    )
    provider = _HERMES_PROVIDER_CLS(
        server_name="srv",
        server_url="https://example.com/mcp",
        client_metadata=metadata,
        storage=storage,
        redirect_handler=_noop_redirect,
        callback_handler=_noop_callback,
    )

    req = httpx.Request("POST", "https://example.com/mcp")
    flow = provider.async_auth_flow(req)

    # First anext() drives the wrapper + inner generator until the inner
    # yields the outbound request (at oauth2.py:503 ``response = yield request``).
    outbound = await flow.__anext__()
    assert outbound is not None, "wrapper must yield the outbound request"
    assert outbound.url.host == "example.com"

    # Simulate httpx returning a 200 response.
    fake_response = httpx.Response(200, request=outbound)

    # The broken wrapper would crash here with AttributeError: 'NoneType'
    # object has no attribute 'status_code', because the SDK's inner generator
    # resumes with response=None and dereferences .status_code at line 505.
    #
    # The correct wrapper forwards the response, the SDK takes the non-401
    # non-403 exit, and the generator ends cleanly (StopAsyncIteration).
    with pytest.raises(StopAsyncIteration):
        await flow.asend(fake_response)


@pytest.mark.asyncio
async def test_hermes_provider_forwards_401_triggers_refresh(tmp_path, monkeypatch):
    """A 401 response MUST flow into the inner generator and trigger the
    SDK's 401 recovery branch.

    With the broken wrapper, the inner generator sees ``response = None``
    and the 401 check short-circuits into AttributeError. With the correct
    bridge, the 401 is routed into the SDK's ``response.status_code == 401``
    branch which begins discovery (yielding a metadata-discovery request).
    """
    # The SDK's httpx flavour (httpx2 on mcp >= 2.0): the provider is an
    # Auth subclass from that module and its auth_flow only accepts its own
    # Request/Response types.
    from tools.mcp_tool import sdk_httpx
    httpx = sdk_httpx()
    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests

    assert _HERMES_PROVIDER_CLS is not None

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    storage = HermesTokenStorage("srv")
    await storage.set_tokens(
        OAuthToken(
            access_token="old_access",
            token_type="Bearer",
            expires_in=3600,
            refresh_token="old_refresh",
        )
    )
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="test-client",
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
        )
    )

    metadata = OAuthClientMetadata(
        redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
        client_name="Hermes Agent",
    )
    provider = _HERMES_PROVIDER_CLS(
        server_name="srv",
        server_url="https://example.com/mcp",
        client_metadata=metadata,
        storage=storage,
        redirect_handler=_noop_redirect,
        callback_handler=_noop_callback,
    )

    req = httpx.Request("POST", "https://example.com/mcp")
    flow = provider.async_auth_flow(req)

    # Drive to the first yield (outbound MCP request).
    outbound = await flow.__anext__()

    # Reply with a 401 including a minimal WWW-Authenticate so the SDK's
    # 401 branch can parse resource metadata from it. We just need something
    # the SDK accepts before it tries to yield the metadata-discovery request.
    fake_401 = httpx.Response(
        401,
        request=outbound,
        headers={"www-authenticate": 'Bearer resource_metadata="https://example.com/.well-known/oauth-protected-resource"'},
    )

    # The correct bridge forwards the 401 into the SDK; the SDK then yields
    # its NEXT request (a metadata-discovery GET). We assert we get a request
    # back — any request. The broken bridge would have crashed with
    # AttributeError before we ever reach this point.
    next_request = await flow.asend(fake_401)
    assert isinstance(next_request, httpx.Request), (
        "wrapper must forward .asend() so the SDK's 401 branch can yield the "
        "next request in the discovery flow"
    )

    # Clean up the generator — we don't need to complete the full dance.
    await flow.aclose()


async def _build_provider(tmp_path):
    """Seed storage + build a provider the way the tests above do."""
    from mcp.shared.auth import (
        OAuthClientInformationFull,
        OAuthClientMetadata,
        OAuthToken,
    )
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS

    storage = HermesTokenStorage("srv")
    await storage.set_tokens(
        OAuthToken(
            access_token="old_access",
            token_type="Bearer",
            expires_in=3600,
            refresh_token="old_refresh",
        )
    )
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="test-client",
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
        )
    )
    return _HERMES_PROVIDER_CLS(
        server_name="srv",
        server_url="https://example.com/mcp",
        client_metadata=OAuthClientMetadata(
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            client_name="Hermes Agent",
        ),
        storage=storage,
        redirect_handler=_noop_redirect,
        callback_handler=_noop_callback,
    )


@pytest.mark.asyncio
async def test_early_close_releases_sdk_auth_lock(tmp_path, monkeypatch):
    """Closing the wrapper mid-flow MUST release the SDK's ``context.lock``.

    The SDK holds ``async with self.context.lock`` across its ``yield``s
    (oauth2.py:493). httpx closes an auth_flow that never ran to completion,
    which raises GeneratorExit at the wrapper's ``yield outgoing``. If the
    wrapper does not close the inner generator in its own task, the inner one
    is finalized later by the GC on a *different* task and anyio's
    ``Lock.release()`` raises "The current task is not holding this lock" —
    leaving the lock held for the life of the process.
    """
    import httpx

    from tools.mcp_oauth_manager import reset_manager_for_tests

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()
    provider = await _build_provider(tmp_path)

    flow = provider.async_auth_flow(httpx.Request("POST", "https://example.com/mcp"))
    await flow.__anext__()
    assert provider.context.lock.locked(), (
        "precondition: the SDK holds context.lock while suspended at its yield"
    )

    await flow.aclose()

    assert not provider.context.lock.locked(), (
        "closing the wrapper must unwind the inner generator's `async with "
        "self.context.lock`; a leaked lock deadlocks every later auth flow"
    )


@pytest.mark.asyncio
async def test_flow_after_early_close_does_not_deadlock(tmp_path, monkeypatch):
    """The symptom the leaked lock produces: the NEXT connect hangs.

    Observed 2026-08-27 — every cron session failed to connect to the
    `consensus` MCP server with ``CancelledError`` after exactly the 60s
    ``connect_timeout``, because the first flow of the gateway process leaked
    the lock and every subsequent flow blocked on acquiring it.
    """
    import asyncio

    import httpx

    from tools.mcp_oauth_manager import reset_manager_for_tests

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()
    provider = await _build_provider(tmp_path)

    first = provider.async_auth_flow(httpx.Request("POST", "https://example.com/mcp"))
    await first.__anext__()
    await first.aclose()

    second = provider.async_auth_flow(httpx.Request("POST", "https://example.com/mcp"))
    try:
        outbound = await asyncio.wait_for(second.__anext__(), timeout=5)
    except asyncio.TimeoutError:
        pytest.fail(
            "second auth flow blocked on context.lock — the first flow leaked it"
        )
    assert isinstance(outbound, httpx.Request)
    await second.aclose()


async def _noop_redirect(_url: str) -> None:
    """Redirect handler that does nothing (won't be invoked in these tests)."""
    return None


async def _noop_callback() -> tuple[str, str | None]:
    """Callback handler that won't be invoked in these tests."""
    raise AssertionError(
        "callback handler should not be invoked in bidirectional-generator tests"
    )
