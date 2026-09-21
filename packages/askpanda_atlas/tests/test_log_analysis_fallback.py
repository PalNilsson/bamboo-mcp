"""Tests for ``_fetch_logs_payload``'s clean-``setup.stdout`` fallback.

``packages/askpanda_atlas/tests/test_log_analysis.py`` is the canonical suite
for ``panda_log_analysis`` and is frozen for the duration of Track B: it is
the regression gate, and it must pass byte-identical from B0 through B8.  New
monolith tests therefore land here until that gate is lifted, at which point
this module should be folded back into the canonical one.

What is covered
---------------
The branch ``_fetch_logs_payload`` takes when ``setup.stdout`` was read and
reported no error, and the payload logs then yielded nothing — both files
confirmed zero-length by the listing, or both undownloadable.  That branch
used to read ``result.setup_log_excerpt or ""``, a field only ever assigned
inside the has-error branch above it, which returns early.  The expression
could therefore only ever resolve to the empty string: the comment promised
the caller environment context and the code delivered none.

The tests below pin the fixed behaviour from both directions — the content is
returned, and the neighbouring cases that were already correct still are:
``setup_log_excerpt`` stays ``None`` for a clean setup log (it is a separate
evidence key, not a second copy of the excerpt), the has-error early return is
untouched, and the extraction is not run at all when the payload logs do carry
content.

All external HTTP calls are patched; no network access is required.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from askpanda_atlas import log_analysis_impl as mono
from askpanda_atlas.log_analysis_impl import (
    _MAX_EXCERPT_CHARS,
    FailureContext,
    _fetch_logs_payload,
    extract_failure_context,
)
from bamboo.tools.log_analysis import panda_log_analysis_tool

_BASE_URL = "https://bigpanda.example.org"
_JOB_ID = 1111
_TIMEOUT = 60

#: A setup log that ``_setup_log_has_error`` does not fire on.
_SETUP_CLEAN = (
    "Using AtlasSetup version 03-00-17\n"
    "Setting up Athena 21.0.15 for x86_64-centos7-gcc8-opt\n"
    "asetup completed\n"
)

#: A clean setup log that nevertheless carries a Python traceback, so the
#: fallback's exception handling is observable.
_SETUP_CLEAN_WITH_TRACEBACK = _SETUP_CLEAN + (
    "Traceback (most recent call last):\n"
    '  File "/cvmfs/x/pilot/control/job.py", line 10, in run_job\n'
    "    _stage_in()\n"
    "RuntimeError: environment probe failed\n"
)

_SETUP_ERROR = "!!!ERROR!!! No matched release is found for 21.0.15\n"


def _sizes(**overrides: int) -> dict[str, int]:
    """Build a top-level file-size index for the three payload-path files.

    Args:
        **overrides: Sizes to override, keyed by filename with ``.`` replaced
            by ``_`` (``setup_stdout``, ``payload_stdout``, ``payload_stderr``).

    Returns:
        Mapping of ``{filename: size_bytes}`` as ``_fetch_logs_payload``
        expects it.
    """
    index = {
        "setup.stdout": len(_SETUP_CLEAN),
        "payload.stdout": 0,
        "payload.stderr": 0,
    }
    for key, value in overrides.items():
        index[key.replace("_", ".", 1)] = value
    return index


def _texts(
    monkeypatch: pytest.MonkeyPatch, texts: dict[str, str | None]
) -> list[str]:
    """Patch the monolith's log download and record the fetch order.

    Args:
        monkeypatch: Pytest fixture.
        texts: Mapping of filename to the text the download should return.
            A missing key returns ``None``, i.e. "could not be downloaded".

    Returns:
        List that accumulates the filenames fetched, in order.
    """
    fetched: list[str] = []

    def _fetch(job_id: int, filename: str, base_url: str, timeout: int) -> str | None:
        fetched.append(filename)
        return texts.get(filename)

    monkeypatch.setattr("askpanda_atlas.log_analysis_impl._fetch_log_text", _fetch)
    return fetched


def _run(index: dict[str, int] | None) -> Any:
    """Call ``_fetch_logs_payload`` with the standard job and base URL.

    Args:
        index: Top-level file-size index, or ``None`` for fail-open.

    Returns:
        The populated ``_LogFetchResult``.
    """
    return _fetch_logs_payload(
        _JOB_ID, 1305, "Payload error", index, _BASE_URL, _TIMEOUT
    )


# ---------------------------------------------------------------------------
# The fallback returns content
# ---------------------------------------------------------------------------

def test_a_clean_setup_log_is_the_excerpt_when_the_payload_logs_are_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero-length payload files must not leave the caller with nothing."""
    _texts(monkeypatch, {"setup.stdout": _SETUP_CLEAN})

    result = _run(_sizes())

    assert result.log_available is True
    assert "asetup completed" in result.log_excerpt


def test_the_fallback_also_covers_undownloadable_payload_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fail-open listing that offers files which then 404 lands here too.

    ``_file_is_nonempty`` says yes for both payload files, both downloads
    return ``None``, and the ``log_text or stderr_text`` guard is false — the
    same branch, reached by a different route.
    """
    fetched = _texts(monkeypatch, {"setup.stdout": _SETUP_CLEAN})

    result = _run(_sizes(payload_stdout=500, payload_stderr=500))

    assert fetched == ["setup.stdout", "payload.stdout", "payload.stderr"]
    assert result.log_available is True
    assert "asetup completed" in result.log_excerpt


def test_the_fallback_carries_an_exception_found_in_the_setup_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A traceback in a clean setup log is evidence; it must not be dropped."""
    _texts(monkeypatch, {"setup.stdout": _SETUP_CLEAN_WITH_TRACEBACK})

    result = _run(_sizes(setup_stdout=len(_SETUP_CLEAN_WITH_TRACEBACK)))

    assert result.exception is not None
    assert result.exception.exc_type == "RuntimeError"
    assert result.traceback_count == 1


def test_the_fallback_excerpt_respects_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long setup log is capped at ``_MAX_EXCERPT_CHARS``, as on every path.

    Setup logs are routinely tens of thousands of characters of asetup
    chatter.  Uncapped, this branch would put all of it in the prompt.
    """
    long_setup = _SETUP_CLEAN + "setup chatter line\n" * 1500
    _texts(monkeypatch, {"setup.stdout": long_setup})

    result = _run(_sizes(setup_stdout=len(long_setup)))

    assert len(result.log_excerpt) == _MAX_EXCERPT_CHARS


def test_the_fallback_excerpt_is_the_extractor_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not a raw slice.

    The primitive path classifies the same file through
    ``extract_failure_context``; a slice here would put the two paths a
    traceback apart on any setup log longer than the budget.
    """
    _texts(monkeypatch, {"setup.stdout": _SETUP_CLEAN_WITH_TRACEBACK})

    result = _run(_sizes(setup_stdout=len(_SETUP_CLEAN_WITH_TRACEBACK)))

    expected: FailureContext = extract_failure_context(
        _SETUP_CLEAN_WITH_TRACEBACK, "setup.stdout", 1305,
        "Payload error", _MAX_EXCERPT_CHARS,
    )
    assert result.log_excerpt == expected.excerpt


# ---------------------------------------------------------------------------
# What the fallback must not change
# ---------------------------------------------------------------------------

def test_a_clean_setup_log_does_not_populate_setup_log_excerpt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``setup_log_excerpt`` means "setup.stdout reported an error".

    It is its own evidence key, carried into the prompt alongside
    ``log_excerpt``; populating it here would send the same text twice.
    """
    _texts(monkeypatch, {"setup.stdout": _SETUP_CLEAN})

    assert _run(_sizes()).setup_log_excerpt is None


def test_an_erroring_setup_log_still_returns_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The has-error branch is untouched: payload logs are never fetched."""
    fetched = _texts(
        monkeypatch,
        {"setup.stdout": _SETUP_ERROR, "payload.stdout": "payload chatter\n"},
    )

    result = _run(_sizes(setup_stdout=len(_SETUP_ERROR), payload_stdout=500))

    assert fetched == ["setup.stdout"]
    assert result.setup_log_excerpt is not None
    assert "No matched release" in result.log_excerpt


def test_a_usable_payload_log_wins_over_the_setup_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback is a fallback; payload content still takes precedence."""
    _texts(
        monkeypatch,
        {"setup.stdout": _SETUP_CLEAN, "payload.stdout": "AthenaMP ERROR: abort\n"},
    )

    result = _run(_sizes(payload_stdout=500))

    assert "AthenaMP ERROR: abort" in result.log_excerpt
    assert "asetup completed" not in result.log_excerpt


def test_the_setup_log_is_not_excerpted_when_the_payload_logs_are_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extraction is deferred into the branch that needs it.

    A clean setup log is excerpted only on the fallback path.  Running it
    eagerly would cost every 1305 job an extra pass over a file whose output
    is then discarded.
    """
    _texts(
        monkeypatch,
        {"setup.stdout": _SETUP_CLEAN, "payload.stdout": "AthenaMP ERROR: abort\n"},
    )
    seen: list[str] = []
    real = mono.extract_failure_context

    def _spy(
        log_text: str,
        log_filename: str,
        pilot_error_code: int,
        pilot_error_diag: str,
        max_chars: int = _MAX_EXCERPT_CHARS,
    ) -> FailureContext:
        seen.append(log_filename)
        return real(log_text, log_filename, pilot_error_code, pilot_error_diag, max_chars)

    monkeypatch.setattr(mono, "extract_failure_context", _spy)

    _run(_sizes(payload_stdout=500))

    assert "setup.stdout" not in seen


def test_no_usable_logs_at_all_is_still_metadata_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With nothing fetched there is nothing to fall back on."""
    _texts(monkeypatch, {})

    result = _run(_sizes(setup_stdout=0))

    assert result.log_available is False
    assert result.log_excerpt == ""


# ---------------------------------------------------------------------------
# Through the tool
# ---------------------------------------------------------------------------

def test_the_evidence_carries_the_setup_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: ``log_excerpt`` reaches the evidence dict, not ``None``.

    This is the change a TUI or REST caller sees.  ``log_excerpt or None`` in
    ``fetch_and_analyse`` turned the empty string into ``None``, so the
    synthesis step was told there was no log at all for a job whose setup log
    had been downloaded successfully.
    """
    job = {
        "pandaid": _JOB_ID,
        "jobstatus": "failed",
        "piloterrorcode": 1305,
        "piloterrordiag": "Payload error",
    }
    monkeypatch.setattr(
        "askpanda_atlas.log_analysis_impl._fetch_metadata",
        lambda job_id, base_url, timeout: {"job": job},
    )
    _texts(monkeypatch, {"setup.stdout": _SETUP_CLEAN})
    monkeypatch.setattr(
        "askpanda_atlas.log_analysis_impl._fetch_file_listing",
        lambda job_id, base_url, timeout: [
            {"relative_path": name, "name": name, "dirname": "", "size_bytes": size,
             "modification": ""}
            for name, size in _sizes().items()
        ],
    )

    result = asyncio.run(panda_log_analysis_tool.call({"job_id": _JOB_ID}))
    evidence: dict[str, Any] = json.loads(result[0]["text"])["evidence"]

    assert evidence["log_available"] is True
    assert evidence["log_excerpt"] is not None
    assert "asetup completed" in evidence["log_excerpt"]
    assert evidence["setup_log_excerpt"] is None
