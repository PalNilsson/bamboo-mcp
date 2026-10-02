"""SQLite compatibility shim for ChromaDB on systems with old system SQLite.

ChromaDB requires SQLite >= 3.35.0.  Several HPC/HEP Linux distributions
(notably RHEL 8/9 and AlmaLinux 9, as deployed on CERN lxplus) ship with
older SQLite versions (typically 3.26–3.34).  Importing ``chromadb`` on these
systems raises::

    RuntimeError: Your system has an unsupported version of sqlite3.
    Chroma requires sqlite3 >= 3.35.0.

The ``pysqlite3-binary`` wheel bundles its own modern SQLite and exposes it
as the ``pysqlite3`` module.  By monkey-patching ``sys.modules["sqlite3"]``
to point at ``pysqlite3`` *before* ``chromadb`` is imported, ChromaDB picks
up the bundled version instead of the system one.

This module exposes a single function, :func:`ensure_sqlite_compat`, which
must be called once before the first ``import chromadb`` statement in any
code path.  Subsequent calls are cheap no-ops (guarded by a module-level
flag).

Installation
------------
Add ``pysqlite3-binary`` to ``requirements-rag.txt`` and install it::

    pip install pysqlite3-binary

If ``pysqlite3`` is not installed (e.g. on a machine whose system SQLite is
already ≥ 3.35.0 and the extra wheel was therefore not needed), the function
logs a warning and returns ``False``.  The caller can decide whether to
proceed or surface the error.
"""
from __future__ import annotations

import logging
import sqlite3
import sys

_log = logging.getLogger(__name__)

# Module-level guard so the patch is applied at most once per interpreter
# session, regardless of how many tools call ensure_sqlite_compat().
_patched: bool = False

#: Minimum SQLite version required by ChromaDB (major, minor, patch).
_CHROMA_MIN_SQLITE = (3, 35, 0)


def _version_tuple(version_str: str) -> tuple[int, ...]:
    """Parse a dotted version string into a tuple of ints.

    Args:
        version_str: Version string such as ``"3.26.0"``.

    Returns:
        Tuple of integers, e.g. ``(3, 26, 0)``.
    """
    return tuple(int(part) for part in version_str.split("."))


def ensure_sqlite_compat() -> bool:
    """Apply the pysqlite3 monkey-patch when the system SQLite is too old.

    Checks the runtime SQLite version against :data:`_CHROMA_MIN_SQLITE`.
    When the system version is sufficient, this function is a no-op and
    returns ``True`` immediately.  When it is insufficient, the function
    attempts to import ``pysqlite3`` (from the ``pysqlite3-binary`` wheel)
    and patch ``sys.modules["sqlite3"]`` so that subsequent ``import sqlite3``
    or ``import chromadb`` statements use the bundled version.

    The patch is applied **at most once** per interpreter process.  Repeated
    calls after the first successful patch return ``True`` without repeating
    work.

    Returns:
        ``True`` if the SQLite version is acceptable (either already
        sufficient, or successfully patched via pysqlite3), ``False`` if
        pysqlite3 is not available and the system version is still too old.

    Example::

        from bamboo.tools._sqlite_compat import ensure_sqlite_compat

        def _ensure_collection(self) -> str | None:
            if not ensure_sqlite_compat():
                return (
                    "System SQLite is too old for ChromaDB and pysqlite3-binary "
                    "is not installed.  Run: pip install pysqlite3-binary"
                )
            import chromadb
            ...
    """
    global _patched  # noqa: PLW0603 -- intentional module-level flag

    if _patched:
        return True

    system_version = _version_tuple(sqlite3.sqlite_version)
    if system_version >= _CHROMA_MIN_SQLITE:
        # System SQLite is fine; nothing to do.
        _patched = True
        return True

    _log.debug(
        "System SQLite %s is below ChromaDB minimum %s; attempting pysqlite3 shim.",
        sqlite3.sqlite_version,
        ".".join(str(v) for v in _CHROMA_MIN_SQLITE),
    )

    try:
        import pysqlite3  # type: ignore[import-untyped]  # optional dep
    except ImportError:
        _log.warning(
            "System SQLite %s is too old for ChromaDB (need >= %s) and "
            "pysqlite3-binary is not installed.  "
            "Install it with: pip install pysqlite3-binary",
            sqlite3.sqlite_version,
            ".".join(str(v) for v in _CHROMA_MIN_SQLITE),
        )
        return False

    # Replace the system sqlite3 with the bundled one before chromadb loads.
    sys.modules["sqlite3"] = pysqlite3  # type: ignore[assignment]
    _patched = True
    # ``pysqlite3/__init__.py`` star-imports from ``dbapi2``, which in turn
    # pulls the name out of an unstubbed C extension, so ``sqlite_version`` is
    # not statically visible however the import is annotated.  Read it
    # dynamically: this is a log line, and a pysqlite3 build that did not
    # export it must not take down the shim it is reporting on.
    bundled_version: str = str(getattr(pysqlite3, "sqlite_version", "unknown"))
    _log.debug(
        "pysqlite3 shim applied (bundled SQLite %s replaces system %s).",
        bundled_version,
        sqlite3.sqlite_version,
    )
    return True


__all__ = ["ensure_sqlite_compat"]
