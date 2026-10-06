"""Classification from the pilot error code when the text matches nothing.

``classify_failure`` was substring matching over ``piloterrordiag`` and the log
excerpt, and nothing else.  That fails in two ways which look identical from
the answer:

- The diag is free text the pilot writes, so a phrasing off by one word stops
  matching.  ``"Network is unreachable"`` — the POSIX ``ENETUNREACH`` string,
  and what XRootD actually reports — missed the table's ``"network
  unreachable"`` entry on every job that hit it.
- When no log can be read, the diag is the only text there is.  Three live
  jobs analysed on 2026-10-05 produced ``"unknown"`` for pilot errors 1361 and
  1324 while the diag stated the cause in plain language.

Both produced ``"unknown"``, which an operator reads as "nothing identifiable
was wrong" rather than "we could not look".

The fallback is consulted only after the exception and the pattern table, so
it can only replace ``"unknown"`` — no job that already classified changes its
answer.  These tests pin that ordering as much as the mapping itself.
"""
from __future__ import annotations

from typing import Any

import pytest

from askpanda_atlas.log_analysis_impl import (
    _PILOT_CODE_CATEGORIES,
    _pilot_error_code_of,
    classify_failure,
)
from askpanda_atlas._traceback_parse import ExceptionInfo


def _job(code: Any = 0, diag: str = "", **extra: Any) -> dict[str, Any]:
    """Build the minimal job dict the classifier reads.

    Args:
        code: ``piloterrorcode``.
        diag: ``piloterrordiag``.
        **extra: Further job fields.

    Returns:
        The job dict.
    """
    job: dict[str, Any] = {"piloterrorcode": code, "piloterrordiag": diag}
    job.update(extra)
    return job


# ---------------------------------------------------------------------------
# The three jobs that exposed this
# ---------------------------------------------------------------------------

def test_a_remote_file_that_would_not_open_is_a_stage_in_failure() -> None:
    """Job 7349880964: pilot error 1361, no readable log.

    The diag names the file and the protocol, and matches no pattern.
    """
    job = _job(
        1361,
        "Remote file could not be opened:Remote file(s) could not be opened: "
        "['root://eos.example.org:1094//atlasdatadisk/DAOD_PHYS.pool.root.1']",
    )
    assert classify_failure(job, "") == "stagein_failed"


def test_an_unreachable_network_is_matched_despite_the_intervening_word() -> None:
    """Job 7349876618: pilot error 1324, ``Network is unreachable``.

    One word between ``network`` and ``unreachable`` was the whole defect.
    """
    job = _job(
        1324,
        "Service not available at the moment: TXT.tar.gz.1 from EXAMPLE_DATADISK, "
        "Error on XrdCl:CopyProcess:Run(): [ERROR] Server responded with an "
        "error: [3014] Unable to open file; Network is unreachable (source)",
    )
    assert classify_failure(job, "") == "network"


def test_a_stage_in_timeout_is_unchanged() -> None:
    """Job 7349767980: pilot error 1151 already matched, and still does.

    The control for the other two: this one was never broken, and the
    fallback must not have moved it.
    """
    job = _job(
        1151,
        "File transfer timed out during stage-in: TXT.tar.gz.1 from "
        "EXAMPLE_DATADISK, copy command timed out",
    )
    assert classify_failure(job, "") == "stagein_timeout"


# ---------------------------------------------------------------------------
# Ordering: the fallback is last
# ---------------------------------------------------------------------------

def test_a_parsed_exception_still_outranks_the_code() -> None:
    """``_classify_from_exception`` runs first and keeps its answer.

    Pilot error 1305 maps to ``payload_error``, but a ``MemoryError`` in the
    traceback says what actually happened, and the exception path is
    consulted before both the pattern table and the code fallback.
    """
    job = _job(1305, "Payload failed")
    exception = ExceptionInfo(
        exc_type="MemoryError",
        exc_type_full="MemoryError",
        message="cannot allocate 4 GiB",
    )

    assert classify_failure(job, "MemoryError: cannot allocate", exception) == "memory"


def test_a_matching_pattern_still_outranks_the_code() -> None:
    """A text match wins, so no previously-classified job changes answer.

    Pilot error 1305 maps to ``payload_error``, but this job's metadata says
    it was reassigned by JEDI, which outranks everything.
    """
    job = _job(1305, "Payload failed", jobsubstatus="toreassign")
    assert classify_failure(job, "") == "reassigned_by_jedi"


def test_the_code_is_consulted_only_when_the_text_says_nothing() -> None:
    """A 1305 job whose diag matches a pattern keeps the pattern's verdict."""
    job = _job(1305, "job has exceeded the memory limit")
    assert classify_failure(job, "") == "memory"


# ---------------------------------------------------------------------------
# The fallback itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("code", "expected"), sorted(_PILOT_CODE_CATEGORIES.items()))
def test_every_mapped_code_classifies_from_metadata_alone(
    code: int, expected: str
) -> None:
    """Each entry in the table is reachable with no text at all.

    Args:
        code: The pilot error code.
        expected: The category it must produce.
    """
    assert classify_failure(_job(code, ""), "") == expected


def test_an_unmapped_code_is_a_pilot_error_not_an_unknown() -> None:
    """A non-zero code the table does not interpret still means *something*.

    ``pilot_error`` is weak, but ``unknown`` is wrong: the pilot set a code,
    so it did report a failure.
    """
    assert classify_failure(_job(1201, ""), "") == "pilot_error"


def test_no_code_and_no_text_is_still_unknown() -> None:
    """The fallback must not manufacture a verdict out of nothing."""
    assert classify_failure(_job(0, ""), "") == "unknown"


def test_a_deliberately_unmapped_code_is_left_to_the_text() -> None:
    """1201 and 1324 are omitted because one code covers several causes.

    1201 is "caught signal" and 1324 spans stage-in and stage-out, so mapping
    either would assert more than the code supports.
    """
    assert 1201 not in _PILOT_CODE_CATEGORIES
    assert 1324 not in _PILOT_CODE_CATEGORIES


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [(1305, 1305), ("1305", 1305), (None, 0), ("", 0), ("n/a", 0), ([], 0)],
)
def test_the_code_is_coerced_without_raising(raw: Any, expected: int) -> None:
    """BigPanDA sometimes sends a string, and sometimes not a number at all.

    Args:
        raw: The value BigPanDA supplied.
        expected: The coerced code.
    """
    assert _pilot_error_code_of({"piloterrorcode": raw}) == expected


def test_an_unparseable_code_classifies_as_unknown_rather_than_raising() -> None:
    """Classification must survive whatever is in the field."""
    assert classify_failure(_job("n/a", ""), "") == "unknown"
