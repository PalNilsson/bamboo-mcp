"""Equivalence walkthrough: the ``atlas.log.*`` loop vs ``fetch_and_analyse``.

The code-mode primitives exist to let an agent compose the steps
``panda_log_analysis`` performs in one call.  That is only worth anything if
the composition reaches the same answer, so this module runs both paths over
the same job — the same metadata, the same file listing, the same downloaded
text — and compares what they produce, scenario by scenario, from
``log_scenarios.py``.

What is compared
----------------
Both paths are projected onto one dict (:func:`_project`):

===================  =============================  ==========================
Key                  ``fetch_and_analyse``          the primitive loop
===================  =============================  ==========================
``failure_type``     ``evidence["failure_type"]``   ``verdict["failure_type"]``
``excerpt``          ``evidence["log_excerpt"]``    the combined context
``exception_type``   ``evidence["exception_type"]`` the context's exception
``traceback_count``  ``evidence["traceback_count"]``  likewise
``pilot_version``    ``evidence["pilot_version"]``  first fetched, else pilotid
``log_available``    ``evidence["log_available"]``  any file came back
``urls``             the three URL evidence keys    each fetched file's ``url``
``fetched``          download order, from a spy     its own spy
===================  =============================  ==========================

``fetched`` is the assertion with the most teeth.  The excerpt agreeing tells
you the two paths arrived somewhere together; the download order agreeing
tells you ``plan_fetch`` transcribed ``_fetch_logs_payload``'s control flow —
the setup-first rule, the early return on a setup error, the zero-length skips
— rather than approximating it.

Differences that are intended
-----------------------------
*Evidence bundling.*  The primitives stop at ``classify`` by design, so the
link block, the follow-up offer, the core-dump probe and the metadata
pass-through have no counterpart here.  The metadata subset is compared
against the evidence separately, in
:func:`test_the_metadata_subset_agrees_with_the_evidence`.

*The verbatim traceback.*  ``context.exception.raw`` is capped at the tool
boundary (D-27) where the monolith carries it whole, so ``exception_type`` and
``traceback_count`` are compared and ``raw`` is not.

*URLs for files that were never read.*  ``_fetch_logs_payload`` assigns
``log_url`` before it knows whether ``payload.stdout`` is worth downloading,
so the evidence can carry a link to a file neither path read — deliberately,
since the link is for a human to click.  The primitives only produce a URL for
a file they fetched, so ``urls`` is compared over the files actually
downloaded.

Lockstep drift
--------------
Two paths can agree and both be wrong: a rule changed in the monolith and
transcribed faithfully into ``plan_fetch`` would keep every comparison above
green.  ``expect_fetched`` and ``expect_failure_type`` in ``log_scenarios.py``
pin the behaviour itself, so that change has to be made in the fixture table
where it is reviewable.

What this module structurally cannot cover
------------------------------------------
Three rules in the primitives guard a caller that does *not* follow the loop
above, so mutating them away leaves every assertion here green — the loop
never exercises them:

- ``plan_fetch`` not re-offering a file the caller already fetched.  The loop
  fetches each plan's files and re-plans once; nothing is ever offered twice.
- the whole-file rule for an erroring ``setup.stdout`` keying on the filename
  rather than the role.  In the loop those always agree.
- ``signals`` carrying ``setup_has_error`` only for ``setup.stdout``.  The
  payload files are fetched in the final round, so their signals are never
  read back.

Each protects an agent that composes the primitives its own way, which is the
point of shipping primitives at all, and each is covered in
``test_log_primitives.py`` and ``test_log_primitives_text.py``.  Recorded here
so the gap is a known one rather than a surprise the next time these are
mutation-tested.

Patching note
-------------
``log_primitives_impl`` imports the fetch helpers into its own namespace with
``from ... import``, so the two paths are patched independently: the monolith
through ``askpanda_atlas.log_analysis_impl``, the primitives through
``askpanda_atlas.log_primitives_impl``.  Each gets its own download spy, which
is what makes the ``fetched`` comparison meaningful — one shared spy would
record both paths into one list.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys
from typing import Any

import pytest

from askpanda_atlas import log_analysis_impl as mono
from askpanda_atlas import log_primitives_impl as impl
from askpanda_atlas.log_primitives_impl import (
    ROLE_PRIMARY,
    ROLE_SECONDARY,
    ROLE_SETUP,
    classify_tool,
    fetch_metadata_tool,
    fetch_text_tool,
    get_classify_definition,
    get_fetch_text_definition,
    get_metadata_definition,
    plan_fetch_tool,
)
from askpanda_atlas.log_primitives_impl import get_definition as get_plan_definition

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from log_scenarios import SCENARIOS, LogScenario, by_name  # noqa: E402

_BASE_URL = "https://bigpanda.example.org"
_JOB_ID = 6799893074
_TIMEOUT = 60

#: The evidence key each role's URL lands in on the monolith side.
_URL_KEY_FOR_ROLE: dict[str, str] = {
    ROLE_SETUP: "setup_log_url",
    ROLE_PRIMARY: "log_url",
    ROLE_SECONDARY: "stderr_url",
}

#: Most plan_fetch calls a terminating loop can need.  ``plan_fetch`` is
#: re-entrant but bounded: setup, then the payload logs, then a terminal empty
#: plan for a caller that re-plans anyway.
_MAX_PLAN_CALLS = 4


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def _wire(
    monkeypatch: pytest.MonkeyPatch, module: Any, scenario: LogScenario
) -> list[str]:
    """Patch one module's metadata, listing and download entry points.

    Args:
        monkeypatch: Pytest fixture.
        module: Either ``log_analysis_impl`` or ``log_primitives_impl``.  Both
            call these three helpers under the same names, but the primitives
            hold their own references, so each must be patched separately.
        scenario: The scenario supplying the responses.

    Returns:
        List accumulating the filenames this module downloads, in order.
    """
    fetched: list[str] = []

    def _metadata(job_id: int, base_url: str, timeout: int) -> dict[str, Any]:
        return {"job": scenario.job}

    def _listing(
        job_id: int, base_url: str, timeout: int
    ) -> list[dict[str, Any]] | None:
        return scenario.listing()

    def _text(job_id: int, filename: str, base_url: str, timeout: int) -> str | None:
        fetched.append(filename)
        return scenario.text_for(filename)

    monkeypatch.setattr(module, "_fetch_metadata", _metadata)
    monkeypatch.setattr(module, "_fetch_file_listing", _listing)
    monkeypatch.setattr(module, "_fetch_log_text", _text)
    return fetched


def _project(
    failure_type: str,
    excerpt: str,
    exception_type: str | None,
    traceback_count: int,
    pilot_version: str,
    log_available: bool,
    urls: dict[str, str],
    fetched: list[str],
) -> dict[str, Any]:
    """Build the comparison dict both paths are reduced to.

    Args:
        failure_type: The classification verdict.
        excerpt: The diagnostic text the verdict was taken from.
        exception_type: Parsed exception type, or ``None``.
        traceback_count: Distinct tracebacks in the log the exception came
            from.
        pilot_version: Pilot release the job ran, or an empty string.
        log_available: Whether any log content was obtained.
        urls: Filebrowser URL per role, for the files actually downloaded.
        fetched: Filenames downloaded, in order.

    Returns:
        The comparison dict.
    """
    return {
        "failure_type": failure_type,
        "excerpt": excerpt,
        "exception_type": exception_type,
        "traceback_count": traceback_count,
        "pilot_version": pilot_version,
        "log_available": log_available,
        "urls": urls,
        "fetched": list(fetched),
    }


# ---------------------------------------------------------------------------
# The two paths
# ---------------------------------------------------------------------------

def _run_monolith(
    fetched: list[str], roles: set[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run ``fetch_and_analyse`` and project its evidence.

    Args:
        fetched: The spy list installed for the monolith.
        roles: Roles the primitive loop downloaded a file for.  The monolith
            emits ``log_url`` even for a file it skipped, so the URL
            comparison is restricted to what was actually read.

    Returns:
        Tuple of the comparison dict and the full evidence dict.
    """
    evidence: dict[str, Any] = mono.fetch_and_analyse(_JOB_ID, _BASE_URL, _TIMEOUT)[
        "evidence"
    ]
    urls = {
        role: evidence[key]
        for role, key in _URL_KEY_FOR_ROLE.items()
        if role in roles and evidence[key]
    }
    return _project(
        failure_type=evidence["failure_type"],
        excerpt=evidence["log_excerpt"] or "",
        exception_type=evidence["exception_type"],
        traceback_count=evidence["traceback_count"],
        pilot_version=evidence["pilot_version"] or "",
        log_available=evidence["log_available"],
        urls=urls,
        fetched=fetched,
    ), evidence


def _run_primitive_loop(
    fetched_spy: list[str],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Run the composed loop exactly as the module docstring writes it.

    The loop below is the contract: the agent fetches what ``next`` names
    with the role it was given, merges the returned ``signals`` into
    ``observed``, and stops when the plan says ``done``.  It evaluates no
    domain predicate of its own.

    Args:
        fetched_spy: The spy list installed for the primitives.

    Returns:
        Tuple of the comparison dict, the metadata subset, and the
        ``fetch_text`` results in fetch order.
    """
    meta: dict[str, Any] = impl.fetch_metadata(_JOB_ID, _BASE_URL, _TIMEOUT)
    observed: dict[str, Any] = {"fetched": []}
    results: list[dict[str, Any]] = []

    plan = impl.plan_fetch(_JOB_ID, _BASE_URL, _TIMEOUT, None)
    for _ in range(_MAX_PLAN_CALLS):
        for entry in plan["next"]:
            got = impl.fetch_text(
                _JOB_ID, entry["filename"], entry["role"], _BASE_URL, _TIMEOUT
            )
            results.append(got)
            observed["fetched"].append(entry["filename"])
            observed.update(got["signals"])
        if plan["done"]:
            break
        plan = impl.plan_fetch(_JOB_ID, _BASE_URL, _TIMEOUT, observed)
    else:  # pragma: no cover - a non-terminating plan is a bug, not a case
        raise AssertionError("plan_fetch did not terminate")

    verdict = impl.classify(meta, results)
    context: dict[str, Any] = verdict["context"]
    exception: dict[str, Any] | None = context.get("exception")

    version = next((r["pilot_version"] for r in results if r["pilot_version"]), "")
    if not version:
        version = meta.get("pilot_version_from_pilotid") or ""

    return _project(
        failure_type=verdict["failure_type"],
        excerpt=context["excerpt"],
        exception_type=exception.get("exc_type") if exception else None,
        traceback_count=context["traceback_count"],
        pilot_version=version,
        log_available=any(r["available"] for r in results),
        urls={r["role"]: r["url"] for r in results},
        fetched=fetched_spy,
    ), meta, results


# ---------------------------------------------------------------------------
# The walkthrough
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_the_primitive_loop_reproduces_fetch_and_analyse(
    monkeypatch: pytest.MonkeyPatch, scenario: LogScenario
) -> None:
    """Both paths, one job: same files, same excerpt, same verdict.

    Args:
        monkeypatch: Pytest fixture.
        scenario: One row of the table in ``log_scenarios.py``.  Its ``pins``
            field says what it is there to hold in place.
    """
    mono_spy = _wire(monkeypatch, mono, scenario)
    prim_spy = _wire(monkeypatch, impl, scenario)

    primitive, _meta, _results = _run_primitive_loop(prim_spy)
    monolith, _evidence = _run_monolith(mono_spy, set(primitive["urls"]))

    assert primitive == monolith, scenario.pins
    assert tuple(primitive["fetched"]) == scenario.expect_fetched, scenario.pins
    assert primitive["failure_type"] == scenario.expect_failure_type, scenario.pins


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_the_metadata_subset_agrees_with_the_evidence(
    monkeypatch: pytest.MonkeyPatch, scenario: LogScenario
) -> None:
    """Every key both sides carry must hold the same value.

    ``fetch_metadata`` projects a subset of the BigPanDA ``job`` dict and
    ``fetch_and_analyse`` promotes an overlapping set into evidence.  Two
    projections of one source drift; this pins them together on the
    intersection, which is every metadata key the evidence has.

    Args:
        monkeypatch: Pytest fixture.
        scenario: One row of the table in ``log_scenarios.py``.
    """
    _wire(monkeypatch, mono, scenario)
    _wire(monkeypatch, impl, scenario)

    meta = impl.fetch_metadata(_JOB_ID, _BASE_URL, _TIMEOUT)
    evidence = mono.fetch_and_analyse(_JOB_ID, _BASE_URL, _TIMEOUT)["evidence"]

    shared = set(meta) & set(evidence)
    assert "piloterrorcode" in shared, "the coerced error code must be comparable"
    for key in sorted(shared):
        assert meta[key] == evidence[key], key


def test_the_subset_carries_every_field_the_classifier_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A field ``_build_search_text`` reads but the subset drops is silent.

    The two paths would still agree whenever that field is empty, which is
    most jobs, and disagree on exactly the jobs it was added for.  Comparing
    the search text itself catches the drop directly rather than waiting for
    a scenario to exercise the field.
    """
    scenario = by_name("metadata_commandtopilot_outranks_the_log")
    _wire(monkeypatch, impl, scenario)

    meta = impl.fetch_metadata(_JOB_ID, _BASE_URL, _TIMEOUT)

    assert mono._build_search_text(meta, "") == mono._build_search_text(
        scenario.job, ""
    )


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_an_unreadable_metadata_response_fails_on_both_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither path invents an answer when BigPanDA cannot be reached."""
    for module in (mono, impl):
        monkeypatch.setattr(
            module, "_fetch_metadata", lambda job_id, base_url, timeout: None
        )

    evidence = mono.fetch_and_analyse(_JOB_ID, _BASE_URL, _TIMEOUT)["evidence"]

    assert "error" in evidence
    assert "error" in impl.fetch_metadata(_JOB_ID, _BASE_URL, _TIMEOUT)
    assert "error" in impl.plan_fetch(_JOB_ID, _BASE_URL, _TIMEOUT, None)


def test_a_missing_job_fails_on_both_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A job ID that does not exist is reported, not classified.

    The monolith says so with ``not_found`` and the primitives with ``error``
    — different shapes, because a primitive has no evidence dict to put a
    flag in, and a partial success whose every field is null would be worse.
    What matters is that neither returns a verdict.
    """
    for module in (mono, impl):
        monkeypatch.setattr(
            module, "_fetch_metadata", lambda job_id, base_url, timeout: {"job": {}}
        )

    evidence = mono.fetch_and_analyse(_JOB_ID, _BASE_URL, _TIMEOUT)["evidence"]

    assert evidence.get("not_found") is True
    assert "failure_type" not in evidence
    assert "error" in impl.fetch_metadata(_JOB_ID, _BASE_URL, _TIMEOUT)
    assert "error" in impl.plan_fetch(_JOB_ID, _BASE_URL, _TIMEOUT, None)


# ---------------------------------------------------------------------------
# The same loop through the tool boundary
# ---------------------------------------------------------------------------

def _call(tool: Any, definition: dict[str, Any], arguments: dict[str, Any]) -> Any:
    """Invoke a tool and validate its structured content against its schema.

    Args:
        tool: The tool object.
        definition: Its definition, carrying ``outputSchema``.
        arguments: Arguments as an MCP client would send them.

    Returns:
        The structured half of the result.
    """
    jsonschema = pytest.importorskip("jsonschema")
    content, structured = asyncio.run(tool.call(arguments))
    assert json.loads(content[0]["text"]) == structured
    jsonschema.validate(instance=structured, schema=definition["outputSchema"])
    return structured


def test_the_loop_runs_through_the_tool_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The walkthrough above calls the implementations; an agent calls tools.

    Everything between the two is argument coercion, the ``_tool_result``
    envelope and ``outputSchema`` validation, and a payload that an agent's
    SDK would reject before the loop ever closed is not equivalent to
    anything.  Run once, over the fullest scenario, rather than once per row:
    the per-tool boundary behaviour is covered in the primitive test modules.
    """
    scenario = by_name("payload_1305_stdout_and_stderr")
    _wire(monkeypatch, mono, scenario)
    prim_spy = _wire(monkeypatch, impl, scenario)
    monkeypatch.setattr(impl, "get_base_url", lambda: _BASE_URL)

    meta = _call(fetch_metadata_tool, get_metadata_definition(), {"job_id": _JOB_ID})
    observed: dict[str, Any] = {"fetched": []}
    results: list[dict[str, Any]] = []

    plan = _call(plan_fetch_tool, get_plan_definition(), {"job_id": _JOB_ID})
    for _ in range(_MAX_PLAN_CALLS):
        for entry in plan["next"]:
            got = _call(
                fetch_text_tool,
                get_fetch_text_definition(),
                {
                    "job_id": _JOB_ID,
                    "filename": entry["filename"],
                    "role": entry["role"],
                },
            )
            results.append(got)
            observed["fetched"].append(entry["filename"])
            observed.update(got["signals"])
        if plan["done"]:
            break
        plan = _call(
            plan_fetch_tool,
            get_plan_definition(),
            {"job_id": _JOB_ID, "observed": observed},
        )

    verdict = _call(
        classify_tool,
        get_classify_definition(),
        {"job": meta, "fetched": results},
    )

    assert tuple(prim_spy) == scenario.expect_fetched
    assert verdict["failure_type"] == scenario.expect_failure_type

    evidence = mono.fetch_and_analyse(_JOB_ID, _BASE_URL, _TIMEOUT)["evidence"]
    assert verdict["context"]["excerpt"] == evidence["log_excerpt"]


# ---------------------------------------------------------------------------
# The table itself
# ---------------------------------------------------------------------------

def test_every_scenario_is_distinct_and_documented() -> None:
    """A duplicated name silently overrides a row in the pytest report."""
    names = [scenario.name for scenario in SCENARIOS]

    assert len(names) == len(set(names))
    assert all(scenario.pins for scenario in SCENARIOS)
