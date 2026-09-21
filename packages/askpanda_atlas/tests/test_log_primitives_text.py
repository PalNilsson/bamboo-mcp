"""Tests for the ATLAS log-excerpt and classification primitives.

The parts with teeth are the ones that would otherwise become the agent's
problem: the character budget derived from the role, the ``setup_has_error``
signal being emitted for ``setup.stdout`` and for nothing else, extraction
running over the whole downloaded file rather than a pre-capped slice, and the
excerpt-joining and traceback-precedence rules :func:`classify` transcribes
from ``_fetch_logs_payload``.  Each of those is a domain rule that, if left
out, an agent would have to reinvent and would reinvent differently.

One end-to-end equivalence check runs the primitive loop and the monolith over
the same fixture and compares the excerpt, the exception and the verdict.  It
is a small preview of the B7 walkthrough, kept here so a change that breaks
equivalence fails in the commit that makes it.

Patching note
-------------
``log_primitives_impl`` imports the fetch helpers into its own namespace with
``from ... import``, so patching ``askpanda_atlas.log_analysis_impl._fetch_log_text``
would leave the primitive calling the original.  Every patch below targets
``askpanda_atlas.log_primitives_impl``, except where the monolith is driven
deliberately.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from askpanda_atlas import log_analysis_impl as mono
from askpanda_atlas import log_primitives_impl as impl
from askpanda_atlas.log_analysis_impl import (
    _MAX_EXCERPT_CHARS,
    _STDERR_RESERVED_CHARS,
    FailureContext,
    _fetch_logs_payload,
    classify_failure,
)
from askpanda_atlas.log_primitives_impl import (
    DEFAULT_MAX_CHARS,
    ENV_MAX_CHARS,
    PAYLOAD_STDERR,
    PAYLOAD_STDOUT,
    ROLE_PRIMARY,
    ROLE_SECONDARY,
    ROLE_SETUP,
    SETUP_LOG,
    SETUP_SIGNAL,
    STDERR_SEPARATOR,
    classify,
    classify_tool,
    fetch_text,
    fetch_text_tool,
    get_classify_definition,
    get_fetch_text_definition,
)

_BASE_URL = "https://bigpanda.example.org"
_JOB_ID = 6799893074
_PILOTLOG = "pilotlog.txt"

_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/cvmfs/x/pilot/control/job.py", line 10, in run_job\n'
    "    _stage_in()\n"
    "ValueError: kaboom\n"
)
_SETUP_ERROR = "!!!ERROR!!! no matched release is found\n"


def _job(code: int = 1305, **overrides: Any) -> dict[str, Any]:
    """Build a minimal BigPanDA ``job`` dict."""
    job: dict[str, Any] = {
        "jobstatus": "failed",
        "piloterrorcode": code,
        "piloterrordiag": "payload failed",
    }
    job.update(overrides)
    return job


def _wire(
    monkeypatch: pytest.MonkeyPatch,
    texts: dict[str, str | None],
    job: dict[str, Any] | None = None,
    payload_none: bool = False,
) -> None:
    """Patch the primitive's metadata fetch and log downloads."""
    payload = None if payload_none else {"job": job if job is not None else _job()}
    monkeypatch.setattr(
        impl, "_fetch_metadata", lambda job_id, base_url, timeout: payload
    )
    monkeypatch.setattr(
        impl,
        "_fetch_log_text",
        lambda job_id, filename, base_url, timeout: texts.get(filename),
    )


def _fetch(filename: str, role: str = ROLE_PRIMARY) -> dict[str, Any]:
    """Call :func:`fetch_text` with the standard job and base URL."""
    return fetch_text(_JOB_ID, filename, role, _BASE_URL, 60)


def _budgets(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the ``max_chars`` each extraction is given."""
    seen: list[int] = []
    real = impl.extract_failure_context

    def _spy(
        log_text: str,
        log_filename: str,
        pilot_error_code: int,
        pilot_error_diag: str,
        max_chars: int = _MAX_EXCERPT_CHARS,
    ) -> FailureContext:
        seen.append(max_chars)
        return real(log_text, log_filename, pilot_error_code, pilot_error_diag, max_chars)

    monkeypatch.setattr(impl, "extract_failure_context", _spy)
    return seen


# ---------------------------------------------------------------------------
# Budget, derived from the role
# ---------------------------------------------------------------------------

def test_secondary_gets_the_stderr_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """payload.stderr is excerpted against the reservation, as in the monolith."""
    _wire(monkeypatch, {PAYLOAD_STDERR: _TRACEBACK})
    seen = _budgets(monkeypatch)
    _fetch(PAYLOAD_STDERR, ROLE_SECONDARY)
    assert seen == [_STDERR_RESERVED_CHARS]


def test_primary_on_the_payload_path_reserves_room_for_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reduction is unconditional, exactly as ``_fetch_logs_payload`` takes it.

    The monolith computes ``_MAX_EXCERPT_CHARS - _STDERR_RESERVED_CHARS``
    before it knows whether payload.stderr has any content, so an excerpt that
    ended up unjoined is still budgeted as if it had been joined.  Making the
    reduction conditional here would produce a different excerpt for exactly
    those jobs.
    """
    _wire(monkeypatch, {PAYLOAD_STDOUT: "output\n" * 50})
    seen = _budgets(monkeypatch)
    _fetch(PAYLOAD_STDOUT, ROLE_PRIMARY)
    assert seen == [DEFAULT_MAX_CHARS - _STDERR_RESERVED_CHARS]


def test_primary_on_the_pilotlog_path_gets_the_whole_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pilot log is never joined to a second file, so it keeps the budget."""
    _wire(monkeypatch, {_PILOTLOG: _TRACEBACK}, job=_job(code=1151))
    seen = _budgets(monkeypatch)
    _fetch(_PILOTLOG, ROLE_PRIMARY)
    assert seen == [DEFAULT_MAX_CHARS]


def test_setup_gets_the_whole_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """setup.stdout is not joined either."""
    _wire(monkeypatch, {SETUP_LOG: "clean setup\n"})
    seen = _budgets(monkeypatch)
    _fetch(SETUP_LOG, ROLE_SETUP)
    assert seen == [DEFAULT_MAX_CHARS]


def test_the_budget_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The primitive budget is configurable independently of the monolith's."""
    monkeypatch.setenv(ENV_MAX_CHARS, "20000")
    _wire(monkeypatch, {_PILOTLOG: _TRACEBACK}, job=_job(code=1151))
    seen = _budgets(monkeypatch)
    _fetch(_PILOTLOG, ROLE_PRIMARY)
    assert seen == [20000]
    assert _MAX_EXCERPT_CHARS == 8000, "the monolith's budget must not move"


@pytest.mark.parametrize("value", ["", "   ", "lots", "0", "-1", "8.5"])
def test_a_misconfigured_budget_falls_back(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """A typo must not fail every call, as with an unrecognised tool profile."""
    monkeypatch.setenv(ENV_MAX_CHARS, value)
    assert impl._max_chars() == DEFAULT_MAX_CHARS


def test_a_tiny_budget_is_not_reduced_further(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Halving an already-tiny budget would leave nothing of the traceback."""
    monkeypatch.setenv(ENV_MAX_CHARS, str(_STDERR_RESERVED_CHARS))
    _wire(monkeypatch, {PAYLOAD_STDOUT: "output\n" * 50})
    seen = _budgets(monkeypatch)
    _fetch(PAYLOAD_STDOUT, ROLE_PRIMARY)
    assert seen == [_STDERR_RESERVED_CHARS]


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def test_setup_error_is_signalled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The predicate is evaluated server-side and handed back opaquely."""
    _wire(monkeypatch, {SETUP_LOG: _SETUP_ERROR})
    assert _fetch(SETUP_LOG, ROLE_SETUP)["signals"] == {SETUP_SIGNAL: True}


def test_a_clean_setup_log_signals_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean setup log still reports, so plan_fetch knows it was read."""
    _wire(monkeypatch, {SETUP_LOG: "asetup ok\n"})
    assert _fetch(SETUP_LOG, ROLE_SETUP)["signals"] == {SETUP_SIGNAL: False}


@pytest.mark.parametrize("filename", [PAYLOAD_STDOUT, PAYLOAD_STDERR, _PILOTLOG])
def test_no_setup_signal_for_any_other_file(
    monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    """Emitting the key for another file would derail the next plan.

    ``_observed_setup_seen`` counts the *presence* of the key as "setup has
    been read".  A ``setup_has_error: false`` picked up from payload.stdout
    and merged into ``observed`` would make the following plan_fetch skip
    setup.stdout entirely, which is the one file it exists to read first.
    """
    _wire(monkeypatch, {filename: _TRACEBACK})
    result = _fetch(filename, ROLE_PRIMARY)
    assert result["signals"] == {}
    assert not impl._observed_setup_seen(result["signals"], frozenset())


def test_the_signal_keys_on_the_filename_not_the_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mislabelled payload log must not start signalling about setup."""
    _wire(monkeypatch, {PAYLOAD_STDOUT: _SETUP_ERROR})
    assert _fetch(PAYLOAD_STDOUT, ROLE_SETUP)["signals"] == {}


def test_the_whole_file_rule_also_keys_on_the_filename(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A payload log that happens to contain !!!ERROR!!! is excerpted normally.

    ``_setup_log_has_error`` is evaluated for every file, so without the
    filename guard any payload log echoing a setup-style error string would be
    returned head-first as a whole capped file instead of tail-excerpted —
    dropping the failure at the end of the log, which is the part that matters.
    """
    text = "HEAD " + _SETUP_ERROR + "chatter\n" * 3000 + "TAIL\n"
    _wire(monkeypatch, {PAYLOAD_STDOUT: text})
    excerpt = _fetch(PAYLOAD_STDOUT, ROLE_PRIMARY)["context"]["excerpt"]
    assert "HEAD" not in excerpt
    assert "TAIL" in excerpt


def test_an_unreadable_setup_log_still_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the signal a fail-open plan would ask for setup.stdout forever."""
    _wire(monkeypatch, {SETUP_LOG: None})
    assert _fetch(SETUP_LOG, ROLE_SETUP)["signals"] == {SETUP_SIGNAL: False}


# ---------------------------------------------------------------------------
# Excerpting
# ---------------------------------------------------------------------------

def test_extraction_searches_the_whole_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A traceback far above the budget must still anchor the excerpt.

    This is the difference between excerpting server-side and returning a
    capped slice: a tail cap would discard this traceback and the primitive
    path would diagnose a different failure from the monolith's.
    """
    text = "noise\n" * 5000 + _TRACEBACK + "trailing\n"
    _wire(monkeypatch, {_PILOTLOG: text}, job=_job(code=1151))
    result = _fetch(_PILOTLOG, ROLE_PRIMARY)
    assert result["context"]["exception"]["exc_type"] == "ValueError"
    assert "kaboom" in result["context"]["excerpt"]


def test_an_erroring_setup_log_without_a_traceback_keeps_the_whole_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setup failures are shell output; an anchor would crop the diagnostics."""
    text = _SETUP_ERROR * 20
    _wire(monkeypatch, {SETUP_LOG: text})
    result = _fetch(SETUP_LOG, ROLE_SETUP)
    assert result["context"]["excerpt"] == text[:DEFAULT_MAX_CHARS]


def test_an_erroring_setup_log_with_a_traceback_is_anchored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When there *is* a traceback the monolith prefers the anchored window."""
    text = "chatter\n" * 200 + _SETUP_ERROR + _TRACEBACK
    _wire(monkeypatch, {SETUP_LOG: text})
    result = _fetch(SETUP_LOG, ROLE_SETUP)
    assert result["context"]["exception"]["exc_type"] == "ValueError"
    assert result["context"]["excerpt"] != text[:DEFAULT_MAX_CHARS]


def test_a_clean_setup_log_is_excerpted_normally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole-file rule applies only when the error predicate fired."""
    text = "HEAD\n" + "asetup ok\n" * 2000 + "TAIL\n"
    _wire(monkeypatch, {SETUP_LOG: text})
    excerpt = _fetch(SETUP_LOG, ROLE_SETUP)["context"]["excerpt"]
    # Normal extraction takes the tail; the whole-file rule would keep the head.
    assert "HEAD" not in excerpt
    assert "TAIL" in excerpt


def test_the_verbatim_traceback_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """``to_state`` carries ``raw`` uncapped; the cap belongs at this boundary."""
    monkeypatch.setenv(ENV_MAX_CHARS, "600")
    frames = (
        '  File "/cvmfs/x/pilot/control/job.py", line %d, in run_job\n    _stage_in()\n'
    )
    text = (
        "Traceback (most recent call last):\n"
        + "".join(frames % n for n in range(200))
        + "ValueError: kaboom\n"
    )
    _wire(monkeypatch, {_PILOTLOG: text}, job=_job(code=1151))
    raw = _fetch(_PILOTLOG, ROLE_PRIMARY)["context"]["exception"]["raw"]
    assert len(raw) <= 600
    # The elision keeps the terminal line, which a slice would discard.
    assert "ValueError: kaboom" in raw


def test_bytes_counts_the_file_not_the_excerpt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``bytes`` is comparable with the listing's ``size_bytes``."""
    text = "π payload output\n" * 100
    _wire(monkeypatch, {PAYLOAD_STDOUT: text})
    result = _fetch(PAYLOAD_STDOUT, ROLE_PRIMARY)
    assert result["bytes"] == len(text.encode("utf-8"))
    assert result["bytes"] > len(text), "non-ASCII must not be counted as characters"


def test_truncated_reports_a_shortened_excerpt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file larger than its budget is flagged; a small one is not."""
    _wire(monkeypatch, {PAYLOAD_STDOUT: "x" * 50000, PAYLOAD_STDERR: "short\n"})
    assert _fetch(PAYLOAD_STDOUT, ROLE_PRIMARY)["truncated"] is True
    assert _fetch(PAYLOAD_STDERR, ROLE_SECONDARY)["truncated"] is False


def test_the_pilot_version_comes_from_the_full_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pilot prints its version at start-up, far from the failure point."""
    text = "pilot version 3.14.0.22\n" + "noise\n" * 5000 + _TRACEBACK
    _wire(monkeypatch, {_PILOTLOG: text}, job=_job(code=1151))
    assert _fetch(_PILOTLOG, ROLE_PRIMARY)["pilot_version"] == "3.14.0.22"


@pytest.mark.parametrize("filename", [PAYLOAD_STDOUT, PAYLOAD_STDERR, SETUP_LOG])
def test_no_pilot_version_from_anything_but_the_pilot_log(
    monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    """A version line in a payload log is not the pilot's own report.

    ``parse_pilot_version`` matches its pattern anywhere in the text, and
    ``_fetch_logs_payload`` never parses a version at all — it falls back to
    the ``pilotid`` metadata field.  Parsing one here would make a composed
    loop report a version ``fetch_and_analyse`` does not, from a file the
    monolith never reads for that purpose.
    """
    _wire(monkeypatch, {filename: "pilot version 9.9.9.9\npayload chatter\n"})
    assert _fetch(filename, ROLE_PRIMARY)["pilot_version"] == ""


def test_the_pilot_log_reports_its_version_whatever_role_it_was_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keyed on the filename, like the setup signal — not on the role.

    An agent composing off-plan may fetch ``pilotlog.txt`` on a 1305 job and
    label it however it likes; the version is a property of the file.
    """
    _wire(monkeypatch, {_PILOTLOG: "pilot version 3.14.0.22\n"}, job=_job(code=1305))
    assert _fetch(_PILOTLOG, ROLE_SECONDARY)["pilot_version"] == "3.14.0.22"


# ---------------------------------------------------------------------------
# Unreadable files and bad arguments
# ---------------------------------------------------------------------------

def test_a_missing_file_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 404 yields an empty context, not an error payload."""
    _wire(monkeypatch, {PAYLOAD_STDOUT: None})
    result = _fetch(PAYLOAD_STDOUT, ROLE_PRIMARY)
    assert result["available"] is False
    assert result["bytes"] == 0
    assert result["context"]["excerpt"] == ""
    assert "could not be downloaded" in result["notes"][0]


def test_an_empty_file_is_distinguishable_from_a_missing_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """"Zero bytes" and "not there" are different operational facts."""
    _wire(monkeypatch, {PAYLOAD_STDOUT: ""})
    result = _fetch(PAYLOAD_STDOUT, ROLE_PRIMARY)
    assert result["available"] is False
    assert "is empty" in result["notes"][0]


def test_an_unknown_role_degrades_to_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo should cost accuracy, not the answer."""
    _wire(monkeypatch, {PAYLOAD_STDOUT: "output\n"})
    result = _fetch(PAYLOAD_STDOUT, "tertiary")
    assert result["role"] == ROLE_PRIMARY
    assert result["notes"]


def test_a_failed_metadata_fetch_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The excerpt rule depends on the pilot error code, so this cannot proceed."""
    _wire(monkeypatch, {PAYLOAD_STDOUT: "output\n"}, payload_none=True)
    result = _fetch(PAYLOAD_STDOUT, ROLE_PRIMARY)
    assert result["error"] == "Failed to fetch job metadata from BigPanDA"


def test_a_missing_job_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Distinguishable from a failed fetch, as in fetch_metadata."""
    _wire(monkeypatch, {PAYLOAD_STDOUT: "output\n"}, job={})
    assert "was not found" in _fetch(PAYLOAD_STDOUT, ROLE_PRIMARY)["error"]


def test_the_url_uses_the_shared_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    """The URL that was read matches the one plan_fetch advertised."""
    _wire(monkeypatch, {PAYLOAD_STDOUT: "output\n"})
    assert _fetch(PAYLOAD_STDOUT, ROLE_PRIMARY)["url"] == (
        f"{_BASE_URL}/filebrowser/?pandaid={_JOB_ID}&json&filename={PAYLOAD_STDOUT}"
    )


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------

def _context(excerpt: str, exception: Any = None) -> dict[str, Any]:
    """Build a context state dict as fetch_text would emit it."""
    return FailureContext(
        excerpt=excerpt, exception=exception, traceback_count=1 if exception else 0
    ).to_state()


def _entry(role: str, excerpt: str, exception: Any = None) -> dict[str, Any]:
    """Build a fetch_text-shaped entry."""
    return {"role": role, "context": _context(excerpt, exception)}


def _parsed(text: str) -> Any:
    """Parse a traceback into an ExceptionInfo."""
    from askpanda_atlas._traceback_parse import find_primary_exception

    exception, _block, _count = find_primary_exception(text)
    return exception


def test_classify_joins_the_payload_excerpts() -> None:
    """The separator is the monolith's, not one the agent invents."""
    result = classify(
        _job(),
        [_entry(ROLE_PRIMARY, "stdout tail"), _entry(ROLE_SECONDARY, "stderr tail")],
    )
    assert result["context"]["excerpt"] == (
        "stdout tail" + STDERR_SEPARATOR + "stderr tail"
    )


def test_classify_prefers_the_stderr_traceback() -> None:
    """Tracebacks and segfaults go to stderr, so that is the terminal one."""
    stdout_exc = _parsed(_TRACEBACK.replace("ValueError", "KeyError"))
    stderr_exc = _parsed(_TRACEBACK)
    result = classify(
        _job(),
        [
            _entry(ROLE_PRIMARY, "stdout", stdout_exc),
            _entry(ROLE_SECONDARY, "stderr", stderr_exc),
        ],
    )
    assert result["context"]["exception"]["exc_type"] == "ValueError"


def test_classify_falls_back_to_the_stdout_traceback() -> None:
    """When stderr has none, the stdout exception is the one to report."""
    stdout_exc = _parsed(_TRACEBACK)
    result = classify(
        _job(),
        [
            _entry(ROLE_PRIMARY, "stdout", stdout_exc),
            _entry(ROLE_SECONDARY, "stderr"),
        ],
    )
    assert result["context"]["exception"]["exc_type"] == "ValueError"


def test_an_exception_without_an_excerpt_still_counts_as_content() -> None:
    """A context is usable if it has *either* text or a parsed exception.

    ``fetch_text`` does not produce that pair, but ``classify`` reads
    unvalidated caller input and the exception is the more valuable half: a
    hand-assembled entry carrying only the traceback must still be classified
    from it rather than discarded as an empty file.
    """
    result = classify(
        _job(),
        [
            _entry(ROLE_PRIMARY, "stdout tail"),
            _entry(ROLE_SECONDARY, "", _parsed(_TRACEBACK)),
        ],
    )
    assert result["context"]["exception"]["exc_type"] == "ValueError"
    assert result["context"]["excerpt"] == "stdout tail" + STDERR_SEPARATOR


def test_an_empty_stderr_is_not_joined() -> None:
    """An unavailable stderr must not append a bare separator."""
    result = classify(
        _job(), [_entry(ROLE_PRIMARY, "stdout tail"), _entry(ROLE_SECONDARY, "")]
    )
    assert result["context"]["excerpt"] == "stdout tail"


def test_stderr_alone_is_still_joined() -> None:
    """The monolith joins whenever stderr has content, even with no stdout."""
    result = classify(
        _job(), [_entry(ROLE_PRIMARY, ""), _entry(ROLE_SECONDARY, "stderr tail")]
    )
    assert result["context"]["excerpt"] == STDERR_SEPARATOR + "stderr tail"


def test_setup_is_used_only_when_no_payload_content_exists() -> None:
    """The setup context is the fallback, never an addition."""
    both = classify(
        _job(),
        [_entry(ROLE_SETUP, "setup text"), _entry(ROLE_PRIMARY, "stdout tail")],
    )
    assert both["context"]["excerpt"] == "stdout tail"

    alone = classify(_job(), [_entry(ROLE_SETUP, "setup text")])
    assert alone["context"]["excerpt"] == "setup text"
    assert alone["notes"]


def test_classification_uses_the_metadata_when_there_are_no_logs() -> None:
    """A job whose status carries no logs is classified from metadata alone."""
    job = _job(code=0, commandtopilot="toreassign")
    result = classify(job, [])
    assert result["failure_type"] == "reassigned_by_jedi"
    assert result["notes"]


def test_classification_matches_classify_failure() -> None:
    """The verdict is the monolith's function, not a reimplementation."""
    job = _job(piloterrordiag="File transfer timed out during stage-in")
    result = classify(job, [_entry(ROLE_PRIMARY, "")])
    assert result["failure_type"] == classify_failure(job, "", None)


@pytest.mark.parametrize("fetched", [None, "stdout", 7, {"role": ROLE_PRIMARY}])
def test_a_non_list_fetched_degrades(fetched: Any) -> None:
    """An unvalidated argument must degrade rather than raise."""
    assert classify(_job(), fetched)["context"]["excerpt"] == ""


def test_junk_members_do_not_discard_the_usable_ones() -> None:
    """One malformed entry costs its own content, not the classification."""
    result = classify(
        _job(), [None, 7, _entry(ROLE_PRIMARY, "stdout tail"), {"role": ROLE_SECONDARY}]
    )
    assert result["context"]["excerpt"] == "stdout tail"


def test_an_entry_without_a_context_is_empty_not_fatal() -> None:
    """``from_state`` tolerates a missing state; so must this."""
    assert classify(_job(), [{"role": ROLE_PRIMARY}])["context"]["excerpt"] == ""


def test_an_unknown_role_is_classified_as_primary() -> None:
    """Content the caller fetched should be used even when mislabelled."""
    result = classify(_job(), [{"role": "tertiary", "context": _context("text")}])
    assert result["context"]["excerpt"] == "text"


def test_duplicate_roles_use_the_first_and_say_so() -> None:
    """Two primaries is not a shape the loop produces; report it rather than guess."""
    result = classify(
        _job(), [_entry(ROLE_PRIMARY, "first"), _entry(ROLE_PRIMARY, "second")]
    )
    assert result["context"]["excerpt"] == "first"
    assert any("primary" in note for note in result["notes"])


@pytest.mark.parametrize("job", [None, "nope", 7, []])
def test_a_non_mapping_job_degrades(job: Any) -> None:
    """Metadata-free classification beats an exception."""
    assert classify(job, [_entry(ROLE_PRIMARY, "")])["failure_type"] == "unknown"


# ---------------------------------------------------------------------------
# Equivalence with the monolith (a B7 preview)
# ---------------------------------------------------------------------------

def test_the_primitive_loop_reproduces_the_monolith_on_the_payload_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same fixture, both paths: excerpt, exception and verdict must agree.

    The primitive loop runs exactly as an agent would: plan, fetch each file
    the plan names with the role it assigned, classify what came back.
    """
    job = _job(code=1305)
    texts: dict[str, str | None] = {
        SETUP_LOG: "asetup ok, release found\n",
        PAYLOAD_STDOUT: "payload chatter\n" * 100,
        PAYLOAD_STDERR: _TRACEBACK,
    }
    sizes = {SETUP_LOG: 10, PAYLOAD_STDOUT: 20, PAYLOAD_STDERR: 30}

    _wire(monkeypatch, texts, job=job)
    monkeypatch.setattr(
        impl,
        "_fetch_file_listing",
        lambda job_id, base_url, timeout: [
            {"relative_path": path, "size_bytes": size, "name": path}
            for path, size in sizes.items()
        ],
    )
    monkeypatch.setattr(
        mono,
        "_fetch_log_text",
        lambda job_id, filename, base_url, timeout: texts.get(filename),
    )

    # --- the primitive loop ---
    observed: dict[str, Any] = {"fetched": []}
    fetched: list[dict[str, Any]] = []
    plan = impl.plan_fetch(_JOB_ID, _BASE_URL, 60, None)
    for _ in range(3):
        for entry in plan["next"]:
            got = _fetch(entry["filename"], entry["role"])
            fetched.append(got)
            observed["fetched"].append(entry["filename"])
            observed.update(got["signals"])
        if plan["done"]:
            break
        plan = impl.plan_fetch(_JOB_ID, _BASE_URL, 60, observed)
    verdict = classify(impl.fetch_metadata(_JOB_ID, _BASE_URL, 60), fetched)

    # --- the monolith ---
    monolith = _fetch_logs_payload(_JOB_ID, 1305, "payload failed", sizes, _BASE_URL, 60)
    expected = classify_failure(job, monolith.log_excerpt, monolith.exception)

    assert monolith.exception is not None, "the fixture must produce a traceback"
    assert verdict["context"]["excerpt"] == monolith.log_excerpt
    assert verdict["context"]["exception"]["exc_type"] == monolith.exception.exc_type
    assert verdict["failure_type"] == expected


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("definition", "name"),
    [
        (get_fetch_text_definition(), "atlas.log.fetch_text"),
        (get_classify_definition(), "atlas.log.classify"),
    ],
)
def test_definitions_are_primitive_and_schema_bearing(
    definition: dict[str, Any], name: str
) -> None:
    """Name, profile and the D-15 envelope, for both tools."""
    assert definition["name"] == name
    assert definition["profiles"] == ["primitive"]
    assert definition["outputSchema"]["type"] == "object"
    assert "required" not in definition["outputSchema"]
    assert "error" in definition["outputSchema"]["properties"]


def test_fetch_text_requires_a_filename() -> None:
    """Without one there is nothing to fetch; role and timeout are optional."""
    schema = get_fetch_text_definition()["inputSchema"]
    assert schema["required"] == ["job_id", "filename"]
    assert set(schema["properties"]) == {"job_id", "filename", "role", "timeout"}


def test_fetch_text_pins_the_role_vocabulary() -> None:
    """The enum is the one plan_fetch emits, so the two cannot drift."""
    role = get_fetch_text_definition()["inputSchema"]["properties"]["role"]
    assert set(role["enum"]) == {ROLE_SETUP, ROLE_PRIMARY, ROLE_SECONDARY}


def test_classify_takes_no_job_id() -> None:
    """It is pure; a job ID would imply it could fetch something."""
    schema = get_classify_definition()["inputSchema"]
    assert schema["required"] == ["job"]
    assert set(schema["properties"]) == {"job", "fetched"}


def test_both_tools_share_one_context_shape() -> None:
    """fetch_text output feeds classify input, so the shapes must be identical."""
    produced = get_fetch_text_definition()["outputSchema"]["properties"]["context"]
    returned = get_classify_definition()["outputSchema"]["properties"]["context"]
    assert produced == returned


# ---------------------------------------------------------------------------
# The tool boundary: structured returns on every path
# ---------------------------------------------------------------------------

def _call(tool: Any, definition: dict[str, Any], arguments: Any) -> dict[str, Any]:
    """Invoke a tool and assert the CombinationContent shape."""
    jsonschema = pytest.importorskip("jsonschema")
    result = asyncio.run(tool.call(arguments))
    assert isinstance(result, tuple), "a tool with outputSchema must return a tuple"
    assert len(result) == 2
    content, structured = result
    assert isinstance(content, list)
    assert isinstance(structured, dict)
    assert json.loads(content[0]["text"]) == structured
    jsonschema.validate(instance=structured, schema=definition["outputSchema"])
    return structured


@pytest.mark.parametrize(
    "arguments",
    [
        "not a dict",
        None,
        [],
        {},
        {"job_id": "abc"},
        {"job_id": _JOB_ID},
        {"job_id": _JOB_ID, "filename": ""},
        {"job_id": _JOB_ID, "filename": 7},
    ],
)
def test_fetch_text_argument_errors_return_structured_content(
    monkeypatch: pytest.MonkeyPatch, arguments: Any
) -> None:
    """A bad argument must not become an opaque output-validation error."""
    monkeypatch.setattr(impl, "get_base_url", lambda: _BASE_URL)
    assert "error" in _call(fetch_text_tool, get_fetch_text_definition(), arguments)


@pytest.mark.parametrize("arguments", ["not a dict", None, [], {}, {"fetched": []}])
def test_classify_argument_errors_return_structured_content(
    arguments: Any,
) -> None:
    """The same guarantee for the pure tool."""
    assert "error" in _call(classify_tool, get_classify_definition(), arguments)


def test_fetch_text_call_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """The happy path validates against the declared schema."""
    _wire(monkeypatch, {PAYLOAD_STDOUT: _TRACEBACK})
    monkeypatch.setattr(impl, "get_base_url", lambda: _BASE_URL)
    structured = _call(
        fetch_text_tool,
        get_fetch_text_definition(),
        {"job_id": _JOB_ID, "filename": PAYLOAD_STDOUT, "role": ROLE_PRIMARY},
    )
    assert structured["available"] is True


def test_classify_call_succeeds() -> None:
    """The happy path validates against the declared schema."""
    structured = _call(
        classify_tool,
        get_classify_definition(),
        {"job": _job(), "fetched": [_entry(ROLE_PRIMARY, "stdout tail")]},
    )
    assert structured["context"]["excerpt"] == "stdout tail"


def test_classify_accepts_a_null_job() -> None:
    """An explicit null is supplied, not missing; it classifies from nothing."""
    structured = _call(
        classify_tool, get_classify_definition(), {"job": None, "fetched": []}
    )
    assert structured["failure_type"] == "unknown"


def test_fetch_text_unexpected_exception_returns_structured_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception inside the worker is reported, not raised through the SDK."""
    monkeypatch.setattr(impl, "get_base_url", lambda: _BASE_URL)

    def _boom(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("filebrowser exploded")

    monkeypatch.setattr(impl, "fetch_text", _boom)
    structured = _call(
        fetch_text_tool,
        get_fetch_text_definition(),
        {"job_id": _JOB_ID, "filename": PAYLOAD_STDOUT},
    )
    assert "filebrowser exploded" in structured["error"]
    assert structured["filename"] == PAYLOAD_STDOUT


def test_classify_unexpected_exception_returns_structured_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pure tool has the same guarantee, with no job_id to report under."""
    def _boom(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("classifier exploded")

    monkeypatch.setattr(impl, "classify_failure", _boom)
    structured = _call(
        classify_tool, get_classify_definition(), {"job": _job(), "fetched": []}
    )
    assert "classifier exploded" in structured["error"]


def test_fetch_text_defaults_the_role_to_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An omitted role is the common case for a single ad-hoc fetch."""
    _wire(monkeypatch, {_PILOTLOG: "text\n"}, job=_job(code=1151))
    monkeypatch.setattr(impl, "get_base_url", lambda: _BASE_URL)
    structured = _call(
        fetch_text_tool,
        get_fetch_text_definition(),
        {"job_id": _JOB_ID, "filename": _PILOTLOG},
    )
    assert structured["role"] == ROLE_PRIMARY
    assert structured["notes"] == []
