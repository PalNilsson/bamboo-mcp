# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Authors
# - Paul Nilsson, paul.nilsson@cern.ch, 2026

"""Tests for :mod:`bamboo.session_scope`.

Covers the four properties the rest of the system relies on: buckets are
isolated per session, an unscoped caller keeps the previous process-global
behaviour, the registry is bounded, and a session id can be handed across a
thread boundary explicitly.
"""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Any
from unittest.mock import patch

import pytest

from bamboo import session_scope as ss


@pytest.fixture(autouse=True)
def _clean_registry() -> Any:
    """Discard all session state before and after each test.

    Yields:
        None.
    """
    ss.reset_all()
    yield
    ss.reset_all()


class TestSessionBinding:
    """Binding, reading, and restoring the active session id."""

    def test_unscoped_reads_return_none(self) -> None:
        """With no scope bound, the active id is None, not the default id."""
        assert ss.current_session_id() is None

    def test_scope_binds_and_restores(self) -> None:
        """The context manager binds an id and restores the previous value."""
        with ss.session_scope("mcp:a"):
            assert ss.current_session_id() == "mcp:a"
        assert ss.current_session_id() is None

    def test_scopes_nest(self) -> None:
        """An inner scope shadows the outer one and restores it on exit."""
        with ss.session_scope("mcp:outer"):
            with ss.session_scope("mcp:inner"):
                assert ss.current_session_id() == "mcp:inner"
            assert ss.current_session_id() == "mcp:outer"

    def test_reset_tolerates_a_foreign_token(self) -> None:
        """Resetting a token from another context is swallowed, not raised."""
        token = ss.set_session_id("mcp:a")
        result: list[BaseException | None] = []

        def _reset_elsewhere() -> None:
            try:
                ss.reset_session_id(token)
                result.append(None)
            except BaseException as exc:  # pragma: no cover - must not happen
                result.append(exc)

        thread = threading.Thread(target=_reset_elsewhere)
        thread.start()
        thread.join()

        assert result == [None]
        ss.reset_session_id(token)


class TestBucketIsolation:
    """Buckets must not be visible across sessions."""

    def test_unscoped_uses_the_default_bucket(self) -> None:
        """An unscoped write lands in the default session."""
        ss.bucket("evidence")["k"] = 1
        assert ss.bucket("evidence", session_id=ss.DEFAULT_SESSION_ID) == {"k": 1}

    def test_two_sessions_do_not_see_each_other(self) -> None:
        """Writes in one session are invisible in another."""
        with ss.session_scope("mcp:a"):
            ss.bucket("evidence")["job"] = 111
        with ss.session_scope("mcp:b"):
            assert ss.bucket("evidence") == {}
            ss.bucket("evidence")["job"] = 222
        with ss.session_scope("mcp:a"):
            assert ss.bucket("evidence")["job"] == 111

    def test_named_buckets_are_separate(self) -> None:
        """Bucket names partition a session's state."""
        with ss.session_scope("mcp:a"):
            ss.bucket(ss.EVIDENCE_BUCKET)["x"] = 1
            assert ss.bucket(ss.PROMPTLOG_BUCKET) == {}

    def test_bucket_returns_the_same_object(self) -> None:
        """Repeated calls return one dict, so in-place mutation is safe."""
        with ss.session_scope("mcp:a"):
            first = ss.bucket("evidence")
            first["x"] = 1
            assert ss.bucket("evidence") is first

    def test_clear_session_drops_only_that_session(self) -> None:
        """Clearing one session leaves the others intact."""
        with ss.session_scope("mcp:a"):
            ss.bucket("evidence")["x"] = 1
        with ss.session_scope("mcp:b"):
            ss.bucket("evidence")["y"] = 2

        ss.clear_session("mcp:a")

        assert "mcp:a" not in ss.active_sessions()
        with ss.session_scope("mcp:b"):
            assert ss.bucket("evidence") == {"y": 2}


class TestEviction:
    """The registry must stay bounded on a long-running server."""

    def test_lru_cap_evicts_the_oldest(self) -> None:
        """Beyond the cap, least-recently-used sessions are dropped."""
        with patch.dict(os.environ, {"BAMBOO_SESSION_BUCKETS": "2"}):
            for sid in ("s1", "s2", "s3"):
                ss.bucket("evidence", session_id=sid)["x"] = sid

            remaining = ss.active_sessions()

        assert "s1" not in remaining
        assert "s2" in remaining
        assert "s3" in remaining

    def test_access_refreshes_lru_position(self) -> None:
        """Touching a session saves it from being the eviction victim."""
        with patch.dict(os.environ, {"BAMBOO_SESSION_BUCKETS": "2"}):
            ss.bucket("evidence", session_id="s1")
            ss.bucket("evidence", session_id="s2")
            ss.bucket("evidence", session_id="s1")  # refresh s1
            ss.bucket("evidence", session_id="s3")

            remaining = ss.active_sessions()

        assert "s1" in remaining
        assert "s2" not in remaining

    def test_ttl_drops_idle_sessions(self) -> None:
        """A bucket untouched for longer than the TTL is discarded."""
        with patch.dict(os.environ, {"BAMBOO_SESSION_TTL_S": "0.05"}):
            ss.bucket("evidence", session_id="stale")["x"] = 1
            # A later access on a different session triggers the prune.
            import time

            time.sleep(0.1)
            ss.bucket("evidence", session_id="fresh")

            remaining = ss.active_sessions()

        assert "stale" not in remaining
        assert "fresh" in remaining

    def test_default_session_is_exempt(self) -> None:
        """The default bucket survives both TTL and cap pressure."""
        with patch.dict(
            os.environ,
            {"BAMBOO_SESSION_BUCKETS": "1", "BAMBOO_SESSION_TTL_S": "0.05"},
        ):
            ss.bucket("evidence")["x"] = 1  # default session
            import time

            time.sleep(0.1)
            for sid in ("s1", "s2", "s3"):
                ss.bucket("evidence", session_id=sid)

            assert ss.DEFAULT_SESSION_ID in ss.active_sessions()
            assert ss.bucket("evidence") == {"x": 1}

    def test_unparsable_env_falls_back(self) -> None:
        """A typo in a tuning knob must not raise or disable the cap."""
        with patch.dict(
            os.environ,
            {"BAMBOO_SESSION_BUCKETS": "many", "BAMBOO_SESSION_TTL_S": "-3"},
        ):
            assert ss.max_sessions() == ss.DEFAULT_MAX_SESSIONS
            assert ss.session_ttl_s() == ss.DEFAULT_SESSION_TTL_S


class TestThreadHandoff:
    """Context variables do not cross threads; the explicit id must."""

    def test_thread_writes_into_the_right_session(self) -> None:
        """A worker thread given the id explicitly reaches the same bucket."""
        with ss.session_scope("mcp:a"):
            captured = ss.current_session_id()

            def _worker() -> None:
                # Inside the thread the context variable is unset.
                assert ss.current_session_id() is None
                ss.bucket("promptlog", session_id=captured)["last_doc"] = ("i", "d")

            thread = threading.Thread(target=_worker)
            thread.start()
            thread.join()

            assert ss.bucket("promptlog")["last_doc"] == ("i", "d")

    def test_to_thread_without_the_id_would_miss(self) -> None:
        """Documents why the id is passed explicitly rather than inferred."""

        async def _run() -> str | None:
            with ss.session_scope("mcp:a"):
                return await asyncio.to_thread(ss.current_session_id)

        # asyncio.to_thread copies the context in Python 3.12, so the variable
        # may or may not survive depending on the runtime.  Either way the
        # bucket must be addressed explicitly; this test pins the fact that the
        # value is not something to rely on for storage decisions.
        result = asyncio.run(_run())
        assert result in (None, "mcp:a")


class TestScopedMapping:
    """The dict-compatible view used in place of a module-level dict."""

    def test_mapping_protocol(self) -> None:
        """Set, get, contains, len, iterate, delete, and clear all work."""
        m = ss.ScopedMapping("evidence")
        m["a"] = 1
        m["b"] = 2

        assert m["a"] == 1
        assert m.get("b") == 2
        assert m.get("missing") is None
        assert "a" in m
        assert len(m) == 2
        assert sorted(m) == ["a", "b"]
        assert bool(m) is True

        del m["a"]
        assert "a" not in m

        m.clear()
        assert len(m) == 0
        assert bool(m) is False

    def test_follows_the_active_session(self) -> None:
        """The same instance reads a different bucket in a different scope."""
        m = ss.ScopedMapping("evidence")
        with ss.session_scope("mcp:a"):
            m["job"] = 111
        with ss.session_scope("mcp:b"):
            assert dict(m) == {}
            m["job"] = 222
        with ss.session_scope("mcp:a"):
            assert m["job"] == 111

    def test_unscoped_instance_matches_a_plain_dict(self) -> None:
        """Unscoped use is indistinguishable from the old global dict."""
        m = ss.ScopedMapping("evidence")
        m["k"] = {"nested": True}
        assert dict(m) == {"k": {"nested": True}}
        assert m.setdefault("k2", 5) == 5
        assert m.pop("k2") == 5

    def test_iteration_tolerates_mutation(self) -> None:
        """Iterating a snapshot means a concurrent write cannot raise."""
        m = ss.ScopedMapping("evidence")
        m["a"] = 1
        for key in m:
            m[f"{key}_copy"] = 2
        assert "a_copy" in m

    def test_repr_names_the_session(self) -> None:
        """The representation is useful when debugging a live server."""
        m = ss.ScopedMapping("evidence")
        with ss.session_scope("mcp:zz"):
            m["k"] = 1
            text = repr(m)
        assert "mcp:zz" in text
        assert "evidence" in text
