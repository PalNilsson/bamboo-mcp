"""Tests for the Reconnect path in ``interfaces/streamlit/chat.py``.

The sidebar's Reconnect button called ``st.cache_resource.clear()`` and nothing
else. That drops the cache's reference to the live :class:`MCPClientSync`
without closing it, and nothing else in the tree ever held one — so every click
orphaned a client together with its event-loop thread, its transport and, on
stdio, a server subprocess. One such orphan held a wedged anyio cancel scope
and spun a full CPU core for 17 hours on ``aipanda033``.

Streamlit is stubbed out via ``sys.modules`` so the module can be imported
headless, following the pattern in ``tests/test_normalise_latex.py``. The
``@st.cache_resource`` decorator is stubbed as a pass-through, because a
``MagicMock`` would replace the decorated function with a mock and there would
be nothing left to test.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest


def _import_chat() -> Any:
    """Import ``interfaces/streamlit/chat.py`` without starting Streamlit.

    Stubs are injected only for the duration of the import and removed from
    ``sys.modules`` afterwards, so other test modules still get the real
    ``interfaces.shared`` packages.

    Returns:
        The imported chat module.
    """
    import importlib.util
    from pathlib import Path

    def _cache_resource(func: Callable[..., Any]) -> Callable[..., Any]:
        """Stand in for ``st.cache_resource`` by returning the function itself.

        Args:
            func: The decorated function.

        Returns:
            ``func`` unchanged.
        """
        return func

    _cache_resource.clear = MagicMock()  # type: ignore[attr-defined]

    st_mock = MagicMock()
    st_mock.session_state = {}
    st_mock.cache_resource = _cache_resource

    stub_names = [
        "streamlit",
        "streamlit.components",
        "streamlit.components.v1",
        "plotly",
        "plotly.graph_objects",
        "plotly.express",
        "interfaces",
        "interfaces.shared",
        "interfaces.shared.mcp_client",
        "interfaces.shared.deeplink",
        "interfaces.shared.superuser_guard",
    ]
    stubs: dict[str, Any] = {
        "streamlit": st_mock,
        "streamlit.components": MagicMock(),
        "streamlit.components.v1": MagicMock(),
        "plotly": MagicMock(),
        "plotly.graph_objects": MagicMock(),
        "plotly.express": MagicMock(),
        "interfaces": types.ModuleType("interfaces"),
        "interfaces.shared": types.ModuleType("interfaces.shared"),
        "interfaces.shared.mcp_client": MagicMock(),
        "interfaces.shared.deeplink": MagicMock(),
        "interfaces.shared.superuser_guard": MagicMock(),
    }

    added: list[str] = []
    saved: dict[str, Any] = {}
    for name in stub_names:
        if name in sys.modules:
            saved[name] = sys.modules[name]
        else:
            added.append(name)
        sys.modules[name] = stubs[name]

    try:
        spec = importlib.util.spec_from_file_location(
            "streamlit_chat_reconnect",
            Path(__file__).parent.parent / "interfaces" / "streamlit" / "chat.py",
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for name in added:
            sys.modules.pop(name, None)
        for name, original in saved.items():
            sys.modules[name] = original
        sys.modules.pop("streamlit_chat_reconnect", None)


@pytest.fixture(scope="module")
def chat() -> Any:
    """Import the chat module once for this test module.

    Returns:
        The imported chat module.
    """
    return _import_chat()


@pytest.fixture(autouse=True)
def _clean_registry(chat: Any) -> Iterator[None]:
    """Empty the live-client registry around every test.

    Args:
        chat: The imported chat module.

    Yields:
        None.
    """
    chat._LIVE_CLIENTS.clear()
    yield
    chat._LIVE_CLIENTS.clear()


class TestClientRegistry:
    """Tests that every client created is recorded so it can be closed."""

    def test_a_created_client_is_registered(self, chat: Any) -> None:
        """``_get_mcp_client`` records what it builds.

        Without this the client is reachable only through the Streamlit cache,
        which hands out no way to close it.

        Args:
            chat: The imported chat module.
        """
        client = chat._get_mcp_client(
            transport="http",
            http_url="http://server.example/mcp",
            bearer_token="",
            stdio_command="/usr/bin/python3",
            trace_file="/tmp/trace.jsonl",
        )

        assert chat._LIVE_CLIENTS == [client]

    def test_a_stdio_client_is_registered_too(self, chat: Any) -> None:
        """The stdio branch builds its config separately; it must register too.

        Args:
            chat: The imported chat module.
        """
        client = chat._get_mcp_client(
            transport="stdio",
            http_url="",
            bearer_token="",
            stdio_command="/usr/bin/python3",
            trace_file="/tmp/trace.jsonl",
        )

        assert chat._LIVE_CLIENTS == [client]

    def test_every_registered_client_is_closed(self, chat: Any) -> None:
        """All of them, not just the most recent.

        Args:
            chat: The imported chat module.
        """
        first, second = MagicMock(), MagicMock()
        chat._register_client(first)
        chat._register_client(second)

        chat._close_live_clients()

        first.close.assert_called_once_with()
        second.close.assert_called_once_with()

    def test_the_registry_is_drained(self, chat: Any) -> None:
        """A closed client must not be closed again on the next reconnect.

        Args:
            chat: The imported chat module.
        """
        client = MagicMock()
        chat._register_client(client)

        chat._close_live_clients()
        chat._close_live_clients()

        assert chat._LIVE_CLIENTS == []
        client.close.assert_called_once_with()

    def test_a_failing_close_does_not_stop_the_others(self, chat: Any) -> None:
        """One wedged client must not leave the rest orphaned.

        Args:
            chat: The imported chat module.
        """
        bad, good = MagicMock(), MagicMock()
        bad.close.side_effect = RuntimeError("transport already gone")
        chat._register_client(bad)
        chat._register_client(good)

        chat._close_live_clients()

        good.close.assert_called_once_with()

    def test_a_failing_close_does_not_propagate(self, chat: Any) -> None:
        """Reconnect is what the user reaches for when things are broken.

        Args:
            chat: The imported chat module.
        """
        bad = MagicMock()
        bad.close.side_effect = RuntimeError("transport already gone")
        chat._register_client(bad)

        chat._close_live_clients()

    def test_closing_nothing_is_harmless(self, chat: Any) -> None:
        """A reconnect before any client exists must not raise.

        Args:
            chat: The imported chat module.
        """
        chat._close_live_clients()

        assert chat._LIVE_CLIENTS == []


class TestReconnect:
    """Tests for the sidebar Reconnect routine."""

    def test_the_client_is_closed_before_the_cache_is_cleared(self, chat: Any) -> None:
        """Order matters: after the clear there is no reference left to close.

        Args:
            chat: The imported chat module.
        """
        order: list[str] = []
        client = MagicMock()
        client.close.side_effect = lambda: order.append("close")
        chat._register_client(client)
        chat.st.cache_resource.clear = lambda: order.append("clear")

        chat._reconnect()

        assert order == ["close", "clear"], order

    def test_the_cache_is_still_cleared(self, chat: Any) -> None:
        """Closing is an addition to the old behaviour, not a replacement.

        Args:
            chat: The imported chat module.
        """
        cleared: list[str] = []
        chat.st.cache_resource.clear = lambda: cleared.append("clear")

        chat._reconnect()

        assert cleared == ["clear"]

    def test_a_failing_close_still_clears_the_cache(self, chat: Any) -> None:
        """A client that cannot be closed must not block the reconnect.

        Args:
            chat: The imported chat module.
        """
        cleared: list[str] = []
        bad = MagicMock()
        bad.close.side_effect = RuntimeError("transport already gone")
        chat._register_client(bad)
        chat.st.cache_resource.clear = lambda: cleared.append("clear")

        chat._reconnect()

        assert cleared == ["clear"]

    def test_the_stale_server_metadata_is_dropped(self, chat: Any) -> None:
        """Pre-existing behaviour: session keys describing the old server go.

        Args:
            chat: The imported chat module.
        """
        chat.st.cache_resource.clear = lambda: None
        chat.st.session_state.update(
            {
                "server_ok": True,
                "tool_names": ["atlas.core_dump_analysis"],
                "display_name": "ATLAS",
                "llm_info": "claude-sonnet-4-6",
                "last_spans": [{"name": "x"}],
                "last_evidence": {"a": 1},
                "last_raw": "raw",
                "last_tool": "atlas.jobs_query",
                "messages": ["keep me"],
            }
        )

        chat._reconnect()

        for key in (
            "server_ok",
            "tool_names",
            "display_name",
            "llm_info",
            "last_spans",
            "last_evidence",
            "last_raw",
            "last_tool",
        ):
            assert key not in chat.st.session_state, f"{key} survived the reconnect"
        assert chat.st.session_state["messages"] == ["keep me"], "chat history was dropped"

    def test_the_script_is_rerun(self, chat: Any) -> None:
        """Pre-existing behaviour: the run ends with a rerun.

        Args:
            chat: The imported chat module.
        """
        chat.st.cache_resource.clear = lambda: None
        chat.st.rerun = MagicMock()

        chat._reconnect()

        chat.st.rerun.assert_called_once_with()
