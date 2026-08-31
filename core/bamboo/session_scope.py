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

"""Session-scoped state for shared Bamboo deployments.

Why this module exists
----------------------
Several pieces of Bamboo state are *conversational*: they are written during
one turn and read during a later one.  The two that matter most are the
evidence produced by the last tool call (``bamboo_executor``) and the
coordinates of the last indexed prompt-log document (``bamboo.llm.prompt_log``).

Under the stdio transport that state can safely be process-global, because one
process serves exactly one user.  Under the HTTP transport it cannot: a single
uvicorn worker serves every connected client, so a process-global dict is
shared by all of them.  Two concrete consequences, both real before this
module existed:

* ``get_last_core_dump_offer()`` gates the bare-affirmative rule — a user
  typing "yes".  Reading a process-global store means user B's "yes" could
  start a core-dump analysis on user A's job.
* ``get_last_traceback_evidence()`` gates the rule 1b pilot-source route, so a
  question could be answered against another user's traceback.

This module provides named per-session buckets plus a context variable naming
the active session, so that state stays conversational without being global.

How the session id is set
-------------------------
The context variable is set once per session at the transport boundary rather
than per request:

* HTTP/MCP — ``bamboo.entrypoints.http._run_session`` sets it as its first
  statement.  Every tool call for that MCP session executes inside that task,
  and a task owns its own context, so one assignment covers the session's
  whole lifetime.
* stdio — nothing sets it.  :func:`current_session_id` then returns ``None``
  and readers fall back to the :data:`DEFAULT_SESSION_ID` bucket, which
  reproduces the previous process-global behaviour exactly.  This is why the
  TUI and the existing test-suite need no changes.

Threads
-------
Context variables do **not** propagate into ``threading.Thread`` workers, and
``asyncio.to_thread`` is a thread.  Code that hands work to a thread must
therefore capture the session id in the calling coroutine and pass it to
:func:`bucket` explicitly via the ``session_id`` argument.  The prompt-log
writer does exactly that.

Eviction
--------
A long-running HTTP server would otherwise accumulate one bucket per session
forever.  Buckets are therefore held in an LRU mapping capped at
``BAMBOO_SESSION_BUCKETS`` entries, and buckets untouched for
``BAMBOO_SESSION_TTL_S`` seconds are dropped.  The default bucket is exempt
from both, since evicting it would silently break cross-turn follow-ups in a
long stdio session.

Environment variables
---------------------
``BAMBOO_SESSION_BUCKETS``
    Maximum number of non-default session buckets retained (default: 128).

``BAMBOO_SESSION_TTL_S``
    Seconds of inactivity after which a non-default bucket is dropped
    (default: 7200).
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

#: Bucket name holding the last-tool evidence dict written by
#: ``bamboo.tools.bamboo_executor``.
EVIDENCE_BUCKET: str = "evidence"

#: Bucket name holding prompt-log coordinates written by
#: ``bamboo.llm.prompt_log``.
PROMPTLOG_BUCKET: str = "promptlog"

#: Session id used when no scope is active.  Buckets under this id behave
#: exactly like the process-global dicts they replaced.
DEFAULT_SESSION_ID: str = "default"

#: Fallbacks for the two tuning knobs, used when the environment variable is
#: unset or unparsable.  An unparsable value falls back rather than raising, so
#: a typo in a deployment environment cannot stop the server from starting.
DEFAULT_MAX_SESSIONS: int = 128
DEFAULT_SESSION_TTL_S: float = 7200.0

_session_id_var: ContextVar[str | None] = ContextVar(
    "bamboo_session_id", default=None
)


@dataclass
class _SessionState:
    """Buckets belonging to one session.

    Attributes:
        buckets: Mapping of bucket name to the bucket's own dict.
        touched: Monotonic timestamp of the most recent access, used for TTL
            and LRU decisions.
    """

    buckets: dict[str, dict[str, Any]] = field(default_factory=dict)
    touched: float = 0.0


#: Session id -> state, ordered oldest-access-first for LRU eviction.
_sessions: OrderedDict[str, _SessionState] = OrderedDict()

#: Guards ``_sessions``.  Reentrant because :func:`bucket` prunes while holding
#: it, and a prune may in turn look at the same mapping.  A plain lock rather
#: than an asyncio one because prompt-log writes reach this module from a
#: worker thread, where an asyncio primitive would be unusable.
_lock: threading.RLock = threading.RLock()


def _env_int(name: str, default: int) -> int:
    """Read a positive integer from the environment.

    Args:
        name: Environment variable name.
        default: Value returned when unset, unparsable, or not positive.

    Returns:
        The parsed value, or *default*.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_float(name: str, default: float) -> float:
    """Read a positive float from the environment.

    Args:
        name: Environment variable name.
        default: Value returned when unset, unparsable, or not positive.

    Returns:
        The parsed value, or *default*.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def max_sessions() -> int:
    """Return the configured cap on retained non-default session buckets.

    Returns:
        ``BAMBOO_SESSION_BUCKETS`` as an int, or
        :data:`DEFAULT_MAX_SESSIONS`.
    """
    return _env_int("BAMBOO_SESSION_BUCKETS", DEFAULT_MAX_SESSIONS)


def session_ttl_s() -> float:
    """Return the configured idle TTL for non-default session buckets.

    Returns:
        ``BAMBOO_SESSION_TTL_S`` as a float, or
        :data:`DEFAULT_SESSION_TTL_S`.
    """
    return _env_float("BAMBOO_SESSION_TTL_S", DEFAULT_SESSION_TTL_S)


def current_session_id() -> str | None:
    """Return the active session id, or ``None`` when no scope is set.

    The ``None`` return is meaningful and callers should preserve the
    distinction: it says "nothing scoped this call", which is the stdio case,
    and lets a caller keep its previous process-global behaviour rather than
    silently reading a bucket that will never be written.

    Returns:
        The session id set by :func:`set_session_id`, or ``None``.
    """
    return _session_id_var.get()


def set_session_id(session_id: str) -> Token[str | None]:
    """Bind *session_id* to the current context.

    Args:
        session_id: Identifier for the session, e.g. ``"mcp:<uuid>"``.

    Returns:
        A token that :func:`reset_session_id` accepts to restore the previous
        value.
    """
    return _session_id_var.set(session_id)


def reset_session_id(token: Token[str | None]) -> None:
    """Restore the session id that was active before *token* was issued.

    Resetting a token in a different context to the one that created it raises
    ``ValueError`` in CPython; that is swallowed here because this is only ever
    called from cleanup paths, where a stale token must not mask the error that
    caused the cleanup.

    Args:
        token: Token returned by :func:`set_session_id`.
    """
    try:
        _session_id_var.reset(token)
    except ValueError:
        pass


@contextmanager
def session_scope(session_id: str) -> Iterator[str]:
    """Bind *session_id* for the duration of the ``with`` block.

    Args:
        session_id: Identifier for the session.

    Yields:
        The bound session id.
    """
    token = set_session_id(session_id)
    try:
        yield session_id
    finally:
        reset_session_id(token)


def _prune_locked(now: float) -> None:
    """Drop expired and over-cap sessions.  Caller must hold :data:`_lock`.

    The default session is exempt from both rules.

    Args:
        now: Current monotonic timestamp.
    """
    ttl = session_ttl_s()
    expired = [
        sid
        for sid, state in _sessions.items()
        if sid != DEFAULT_SESSION_ID and (now - state.touched) > ttl
    ]
    for sid in expired:
        _sessions.pop(sid, None)

    cap = max_sessions()
    # Count only evictable sessions, so a configured cap of N always leaves
    # room for N real sessions regardless of whether the default exists.
    evictable = [sid for sid in _sessions if sid != DEFAULT_SESSION_ID]
    excess = len(evictable) - cap
    for sid in evictable[:excess] if excess > 0 else []:
        _sessions.pop(sid, None)


def bucket(name: str, session_id: str | None = None) -> dict[str, Any]:
    """Return the named bucket for a session, creating it if absent.

    Args:
        name: Bucket name, e.g. :data:`EVIDENCE_BUCKET`.
        session_id: Session to read.  When ``None`` the active session id is
            used, falling back to :data:`DEFAULT_SESSION_ID`.  Pass this
            explicitly from thread workers, where the context variable is not
            visible.

    Returns:
        The bucket dict.  The same object is returned for repeated calls with
        the same arguments, so callers may mutate it in place.
    """
    sid = session_id if session_id is not None else (current_session_id() or DEFAULT_SESSION_ID)
    now = time.monotonic()

    with _lock:
        state = _sessions.get(sid)
        if state is None:
            state = _SessionState()
            _sessions[sid] = state
        state.touched = now
        _sessions.move_to_end(sid)
        _prune_locked(now)
        # Re-insert if a pathological cap of 0 pruned the session we just
        # created; the caller must still get a usable dict back.
        if sid not in _sessions:
            _sessions[sid] = state
        return state.buckets.setdefault(name, {})


def clear_session(session_id: str) -> None:
    """Discard every bucket belonging to *session_id*.

    Args:
        session_id: Session to drop.  Unknown ids are ignored.
    """
    with _lock:
        _sessions.pop(session_id, None)


def reset_all() -> None:
    """Discard all session state.

    Intended for tests and for a clean process shutdown.
    """
    with _lock:
        _sessions.clear()


def active_sessions() -> list[str]:
    """Return the ids of all retained sessions, oldest access first.

    Returns:
        List of session ids.
    """
    with _lock:
        return list(_sessions)


class ScopedMapping(MutableMapping[str, Any]):
    """A dict-like view onto one named bucket of the *active* session.

    Every operation resolves the bucket afresh, so a single module-level
    instance can stand in for what used to be a module-level dict: readers and
    writers keep their existing syntax while the storage follows the session.

    This indirection is deliberate.  The alternative — rewriting each of the
    two dozen call sites in ``bamboo_executor`` to a helper function — leaves
    room for a missed site to silently reintroduce cross-user leakage, and a
    missed site is invisible in review because it still type-checks and still
    passes tests that run unscoped.

    Attributes:
        name: Bucket name this view is bound to.
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        """Bind the view to a bucket name.

        Args:
            name: Bucket name, e.g. :data:`EVIDENCE_BUCKET`.
        """
        self.name = name

    def _bucket(self) -> dict[str, Any]:
        """Return the underlying bucket for the active session.

        Returns:
            The bucket dict.
        """
        return bucket(self.name)

    def __getitem__(self, key: str) -> Any:
        """Return the value stored under *key*.

        Args:
            key: Bucket key.

        Returns:
            The stored value.

        Raises:
            KeyError: If *key* is absent.
        """
        return self._bucket()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Store *value* under *key*.

        Args:
            key: Bucket key.
            value: Value to store.
        """
        self._bucket()[key] = value

    def __delitem__(self, key: str) -> None:
        """Remove *key* from the bucket.

        Args:
            key: Bucket key.

        Raises:
            KeyError: If *key* is absent.
        """
        del self._bucket()[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over the bucket's keys.

        Returns:
            Iterator over a snapshot of the keys, so that mutation during
            iteration cannot raise.
        """
        return iter(list(self._bucket()))

    def __len__(self) -> int:
        """Return the number of keys in the bucket.

        Returns:
            Key count.
        """
        return len(self._bucket())

    def __repr__(self) -> str:
        """Return a debugging representation naming the session and bucket.

        Returns:
            Representation string.
        """
        sid = current_session_id() or DEFAULT_SESSION_ID
        return f"ScopedMapping(name={self.name!r}, session={sid!r}, keys={sorted(self._bucket())!r})"


__all__ = [
    "DEFAULT_MAX_SESSIONS",
    "DEFAULT_SESSION_ID",
    "DEFAULT_SESSION_TTL_S",
    "EVIDENCE_BUCKET",
    "PROMPTLOG_BUCKET",
    "ScopedMapping",
    "active_sessions",
    "bucket",
    "clear_session",
    "current_session_id",
    "max_sessions",
    "reset_all",
    "reset_session_id",
    "session_scope",
    "session_ttl_s",
    "set_session_id",
]
