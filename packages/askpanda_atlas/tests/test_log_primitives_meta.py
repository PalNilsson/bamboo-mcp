"""Tests for the ATLAS metadata and file-listing primitives.

Covers the projected metadata subset — including the two fields the monolith
consumes without promoting to evidence, whose absence would silently change a
classification — the fail-open reporting of an unavailable listing, and the
root-first ordering that keeps the diagnostic files inside the entry cap.

As in :mod:`test_log_primitives`, every ``call()`` return path is asserted to
yield the ``(content, structured)`` tuple the MCP SDK requires from a tool
advertising an ``outputSchema``, with the payload validated against that
schema by real ``jsonschema``.

Patching note
-------------
``log_primitives_impl`` imports the fetch helpers into its own namespace with
``from ... import``, so patching ``askpanda_atlas.log_analysis_impl._fetch_metadata``
would leave the primitive calling the original.  Every patch below therefore
targets ``askpanda_atlas.log_primitives_impl``.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from askpanda_atlas import log_primitives_impl as impl
from askpanda_atlas.log_analysis_impl import classify_failure
from askpanda_atlas.log_primitives_impl import (
    MAX_LISTING_ENTRIES,
    fetch_metadata,
    fetch_metadata_tool,
    get_list_files_definition,
    get_metadata_definition,
    list_files,
    list_files_tool,
)

_BASE_URL = "https://bigpanda.example.org"
_JOB_ID = 6799893074


def _job(**overrides: Any) -> dict[str, Any]:
    """Build a BigPanDA ``job`` dict with plausible values."""
    job: dict[str, Any] = {
        "jobstatus": "failed",
        "jobsubstatus": "pilot_failed",
        "computingsite": "BNL-ATLAS",
        "cloud": "US",
        "atlasrelease": "Athena-22.0.1",
        "jeditaskid": 12345,
        "attemptnr": 2,
        "maxattempt": 3,
        "transformation": "Sim_tf.py",
        "piloterrorcode": 1305,
        "piloterrordiag": "payload failed",
        "exeerrorcode": 65,
        "exeerrordiag": "athena crashed",
        "taskbuffererrorcode": None,
        "taskbuffererrordiag": None,
        "ddmerrorcode": None,
        "ddmerrordiag": None,
        "starttime": "2026-05-01T09:00:00",
        "endtime": "2026-05-01T09:40:00",
        "duration": 2400,
        "commandtopilot": "",
        "pilotid": "aipanda042.cern.ch|3.14.0.22|SLOT",
    }
    job.update(overrides)
    return job


def _wire_metadata(
    monkeypatch: pytest.MonkeyPatch,
    job: dict[str, Any] | None = None,
    payload_none: bool = False,
) -> None:
    """Patch the primitive's metadata fetch."""
    payload = None if payload_none else {"job": job if job is not None else _job()}
    monkeypatch.setattr(
        impl, "_fetch_metadata", lambda job_id, base_url, timeout: payload
    )


def _wire_listing(
    monkeypatch: pytest.MonkeyPatch,
    listing: list[dict[str, Any]] | None,
) -> None:
    """Patch the primitive's file-listing fetch."""
    monkeypatch.setattr(
        impl, "_fetch_file_listing", lambda job_id, base_url, timeout: listing
    )


def _entry(relative_path: str, size: int = 100) -> dict[str, Any]:
    """Build one normalised listing entry."""
    name = relative_path.rsplit("/", 1)[-1]
    dirname = relative_path[: -len(name)].strip("/")
    return {
        "relative_path": relative_path,
        "name": name,
        "dirname": dirname,
        "size_bytes": size,
        "modification": "2026-05-01 09:40:00",
    }


# ---------------------------------------------------------------------------
# fetch_metadata — the subset
# ---------------------------------------------------------------------------

def test_metadata_projects_the_evidence_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every field the monolith promotes into evidence is projected."""
    _wire_metadata(monkeypatch)
    result = fetch_metadata(_JOB_ID, _BASE_URL, 60)
    assert result["jobstatus"] == "failed"
    assert result["computingsite"] == "BNL-ATLAS"
    assert result["transformation"] == "Sim_tf.py"
    assert result["duration"] == 2400
    assert result["monitor_url"] == f"{_BASE_URL}/job?pandaid={_JOB_ID}"


def test_metadata_carries_commandtopilot(monkeypatch: pytest.MonkeyPatch) -> None:
    """``commandtopilot`` is searched by classify_failure but is not evidence.

    ``_build_search_text`` reads it, and the JEDI-reassignment signal arrives
    there rather than in any error-diagnosis field.  A subset modelled on the
    evidence keys alone would drop it, and a classification taken over the
    subset would then disagree with one taken over the full job dict — for
    exactly the jobs that never really failed.
    """
    job = _job(commandtopilot="tobekilled toreassign")
    _wire_metadata(monkeypatch, job=job)
    subset = fetch_metadata(_JOB_ID, _BASE_URL, 60)

    assert classify_failure(job, "", None) == "reassigned_by_jedi"
    assert classify_failure(subset, "", None) == "reassigned_by_jedi"


@pytest.mark.parametrize(
    "field_name",
    ["taskbuffererrordiag", "piloterrordiag", "exeerrordiag", "jobsubstatus"],
)
def test_every_classification_input_survives_the_projection(
    monkeypatch: pytest.MonkeyPatch, field_name: str
) -> None:
    """Each field ``_build_search_text`` reads must reach the subset."""
    job = _job(**{field_name: "lost heartbeat"})
    _wire_metadata(monkeypatch, job=job)
    subset = fetch_metadata(_JOB_ID, _BASE_URL, 60)
    assert classify_failure(subset, "", None) == classify_failure(job, "", None)


def test_metadata_carries_the_pilotid_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pilot version is derived server-side, not left to the agent."""
    _wire_metadata(monkeypatch)
    result = fetch_metadata(_JOB_ID, _BASE_URL, 60)
    assert result["pilot_version_from_pilotid"] == "3.14.0.22"


def test_absent_fields_are_null_rather_than_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shape is fixed so a consumer need not probe with ``in``."""
    _wire_metadata(monkeypatch, job={"jobstatus": "failed"})
    result = fetch_metadata(_JOB_ID, _BASE_URL, 60)
    assert "ddmerrordiag" in result
    assert result["ddmerrordiag"] is None


def test_pilot_error_code_is_coerced(monkeypatch: pytest.MonkeyPatch) -> None:
    """A string code is coerced exactly as ``fetch_and_analyse`` coerces it."""
    _wire_metadata(monkeypatch, job=_job(piloterrorcode="1305"))
    assert fetch_metadata(_JOB_ID, _BASE_URL, 60)["piloterrorcode"] == 1305


def test_unparseable_pilot_error_code_reads_as_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unparseable code degrades to 0, as the monolith's coercion does."""
    _wire_metadata(monkeypatch, job=_job(piloterrorcode="n/a"))
    assert fetch_metadata(_JOB_ID, _BASE_URL, 60)["piloterrorcode"] == 0


def test_metadata_reports_no_decisions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Which file to read, and whether logs exist, belong to plan_fetch."""
    _wire_metadata(monkeypatch)
    result = fetch_metadata(_JOB_ID, _BASE_URL, 60)
    assert "primary_log" not in result
    assert "log_bearing" not in result


def test_metadata_failure_reports_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed fetch is an error, not a payload of nulls."""
    _wire_metadata(monkeypatch, payload_none=True)
    result = fetch_metadata(_JOB_ID, _BASE_URL, 60)
    assert result["error"] == "Failed to fetch job metadata from BigPanDA"
    assert result["job_id"] == _JOB_ID


def test_missing_job_is_distinguishable_from_a_failed_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two failure modes need different messages to be actionable."""
    _wire_metadata(monkeypatch, job={})
    result = fetch_metadata(_JOB_ID, _BASE_URL, 60)
    assert "was not found" in result["error"]
    assert result["error"] != "Failed to fetch job metadata from BigPanDA"


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------

def test_listing_projects_the_entry_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each entry keeps the five normalised keys and nothing else."""
    _wire_listing(monkeypatch, [_entry("pilotlog.txt", 4096)])
    result = list_files(_JOB_ID, _BASE_URL, 60)
    assert result["listing_available"] is True
    assert result["files"][0] == {
        "relative_path": "pilotlog.txt",
        "name": "pilotlog.txt",
        "dirname": "",
        "size_bytes": 4096,
        "modification": "2026-05-01 09:40:00",
    }


def test_a_sparse_entry_still_matches_the_declared_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An entry missing keys must not emit nulls against string types.

    ``additionalProperties`` is false and ``dirname``/``modification`` are
    declared strings, so a passed-through ``None`` makes the SDK reject the
    whole listing with an output-validation error — the opaque failure the
    D-15 envelope exists to avoid.  Caught by the end-to-end run, not by a
    unit test, which is why it is pinned here.
    """
    jsonschema = pytest.importorskip("jsonschema")
    _wire_listing(monkeypatch, [{"relative_path": "pilotlog.txt"}])
    result = list_files(_JOB_ID, _BASE_URL, 60)

    assert result["files"][0]["dirname"] == ""
    assert result["files"][0]["size_bytes"] == 0
    jsonschema.validate(
        instance=result, schema=get_list_files_definition()["outputSchema"]
    )


def test_root_files_come_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """Job-root entries lead, so the cap cannot drop the diagnostic files."""
    _wire_listing(
        monkeypatch,
        [
            _entry("workDir/in.txt"),
            _entry("setup.stdout"),
            _entry("workDir/usr/lib/thing.so"),
            _entry("pilotlog.txt"),
        ],
    )
    paths = [f["relative_path"] for f in list_files(_JOB_ID, _BASE_URL, 60)["files"]]
    assert paths[:2] == ["setup.stdout", "pilotlog.txt"]


def test_listing_is_capped_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tarball of thousands of files must not swamp the agent's context."""
    listing = [_entry("setup.stdout")] + [
        _entry(f"workDir/f{i}.txt") for i in range(MAX_LISTING_ENTRIES + 50)
    ]
    _wire_listing(monkeypatch, listing)
    result = list_files(_JOB_ID, _BASE_URL, 60)

    assert len(result["files"]) == MAX_LISTING_ENTRIES
    assert result["total"] == MAX_LISTING_ENTRIES + 51
    assert result["truncated"] is True
    assert result["notes"]
    # The root-level log survived the cap; that is the point of the ordering.
    assert result["files"][0]["relative_path"] == "setup.stdout"


def test_a_listing_at_the_cap_is_not_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The boundary is inclusive: exactly the cap is a complete listing."""
    _wire_listing(
        monkeypatch, [_entry(f"f{i}.txt") for i in range(MAX_LISTING_ENTRIES)]
    )
    result = list_files(_JOB_ID, _BASE_URL, 60)
    assert result["truncated"] is False
    assert result["notes"] == []
    assert len(result["files"]) == MAX_LISTING_ENTRIES


def test_unavailable_listing_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """``None`` means unknown, and every monolith consumer treats it fail-open.

    Reporting it as an error would make this primitive disagree with
    ``plan_fetch``, which offers every file when the listing is unavailable.
    """
    _wire_listing(monkeypatch, None)
    result = list_files(_JOB_ID, _BASE_URL, 60)
    assert "error" not in result
    assert result["listing_available"] is False
    assert result["files"] == []
    assert result["notes"]


def test_an_empty_listing_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """``[]`` is a real answer and must not read as an unavailable listing."""
    _wire_listing(monkeypatch, [])
    result = list_files(_JOB_ID, _BASE_URL, 60)
    assert result["listing_available"] is True
    assert result["total"] == 0
    assert result["truncated"] is False


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("definition", "name"),
    [
        (get_metadata_definition(), "atlas.log.fetch_metadata"),
        (get_list_files_definition(), "atlas.log.list_files"),
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
    assert definition["inputSchema"]["required"] == ["job_id"]
    assert definition["inputSchema"]["additionalProperties"] is False


def test_metadata_schema_declares_every_projected_field() -> None:
    """A field present in the payload but absent from the schema is rejected.

    ``additionalProperties`` is false, so the two lists have to agree or the
    SDK refuses the result.
    """
    properties = get_metadata_definition()["outputSchema"]["properties"]
    for field_name in impl._METADATA_FIELDS:
        assert field_name in properties


def test_pass_through_fields_are_untyped() -> None:
    """BigPanDA decides these types; declaring one here would reject jobs."""
    properties = get_metadata_definition()["outputSchema"]["properties"]
    assert "type" not in properties["jeditaskid"]
    assert properties["piloterrorcode"]["type"] == "integer"


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
    ["not a dict", None, [], {}, {"job_id": None}, {"job_id": "abc"}, {"job_id": []}],
)
def test_metadata_argument_errors_return_structured_content(
    monkeypatch: pytest.MonkeyPatch, arguments: Any
) -> None:
    """A bad argument must not become an opaque output-validation error."""
    monkeypatch.setattr(impl, "get_base_url", lambda: _BASE_URL)
    assert "error" in _call(fetch_metadata_tool, get_metadata_definition(), arguments)


@pytest.mark.parametrize(
    "arguments",
    ["not a dict", None, [], {}, {"job_id": None}, {"job_id": "abc"}],
)
def test_list_files_argument_errors_return_structured_content(
    monkeypatch: pytest.MonkeyPatch, arguments: Any
) -> None:
    """The same guarantee for the listing tool."""
    monkeypatch.setattr(impl, "get_base_url", lambda: _BASE_URL)
    assert "error" in _call(list_files_tool, get_list_files_definition(), arguments)


def test_metadata_call_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """The happy path validates against the declared schema."""
    _wire_metadata(monkeypatch)
    monkeypatch.setattr(impl, "get_base_url", lambda: _BASE_URL)
    structured = _call(
        fetch_metadata_tool, get_metadata_definition(), {"job_id": _JOB_ID}
    )
    assert structured["jobstatus"] == "failed"


def test_list_files_call_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """The happy path validates against the declared schema."""
    _wire_listing(monkeypatch, [_entry("pilotlog.txt")])
    monkeypatch.setattr(impl, "get_base_url", lambda: _BASE_URL)
    structured = _call(
        list_files_tool, get_list_files_definition(), {"job_id": _JOB_ID}
    )
    assert structured["total"] == 1


@pytest.mark.parametrize(
    ("tool_name", "target"),
    [("fetch_metadata", "fetch_metadata"), ("list_files", "list_files")],
)
def test_unexpected_exceptions_return_structured_content(
    monkeypatch: pytest.MonkeyPatch, tool_name: str, target: str
) -> None:
    """An exception inside the worker is reported, not raised through the SDK."""
    monkeypatch.setattr(impl, "get_base_url", lambda: _BASE_URL)

    def _boom(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("bigpanda exploded")

    monkeypatch.setattr(impl, target, _boom)
    tool = fetch_metadata_tool if tool_name == "fetch_metadata" else list_files_tool
    definition = (
        get_metadata_definition() if tool_name == "fetch_metadata"
        else get_list_files_definition()
    )
    structured = _call(tool, definition, {"job_id": _JOB_ID})
    assert "bigpanda exploded" in structured["error"]
    assert structured["job_id"] == _JOB_ID


def test_bad_timeout_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unparseable timeout degrades rather than failing the call."""
    monkeypatch.setattr(impl, "get_base_url", lambda: _BASE_URL)
    seen: list[int] = []

    def _spy(job_id: int, base_url: str, timeout: int) -> dict[str, Any]:
        seen.append(timeout)
        return {"job_id": job_id, "listing_available": True, "files": [],
                "total": 0, "truncated": False, "notes": []}

    monkeypatch.setattr(impl, "list_files", _spy)
    _call(
        list_files_tool,
        get_list_files_definition(),
        {"job_id": _JOB_ID, "timeout": "soon"},
    )
    assert seen == [60]
