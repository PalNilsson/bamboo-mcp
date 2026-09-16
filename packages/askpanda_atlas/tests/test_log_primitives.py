"""Tests for the ATLAS log-fetch planning primitive.

Covers strategy selection, the re-entrant loop and its bound, every
zero-length and fail-open skip, tolerant reading of ``observed``, and — the
part with teeth — that *every* ``call()`` return path yields the
``(content, structured)`` tuple the MCP SDK requires from a tool advertising
an ``outputSchema``, with both success and failure payloads validated against
that schema by real ``jsonschema``.

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
from askpanda_atlas.log_analysis_impl import _fetch_logs_payload, _log_file_url
from askpanda_atlas.log_primitives_impl import (
    PAYLOAD_STDERR,
    PAYLOAD_STDOUT,
    ROLE_PRIMARY,
    ROLE_SECONDARY,
    ROLE_SETUP,
    SETUP_LOG,
    STRATEGY_METADATA_ONLY,
    STRATEGY_PAYLOAD_1305,
    STRATEGY_PILOTLOG,
    get_definition,
    plan_fetch,
    plan_fetch_tool,
)

_BASE_URL = "https://bigpanda.example.org"
_JOB_ID = 6799893074

_PILOTLOG = "pilotlog.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _job(status: str = "failed", code: int = 1305) -> dict[str, Any]:
    """Build a minimal BigPanDA ``job`` dict."""
    return {"jobstatus": status, "piloterrorcode": code, "piloterrordiag": "boom"}


def _listing(sizes: dict[str, int]) -> list[dict[str, Any]]:
    """Build a filebrowser listing from ``{relative_path: size_bytes}``."""
    return [
        {"relative_path": path, "size_bytes": size, "name": path.rsplit("/", 1)[-1]}
        for path, size in sizes.items()
    ]


_ALL_PRESENT = {SETUP_LOG: 100, PAYLOAD_STDOUT: 200, PAYLOAD_STDERR: 300, _PILOTLOG: 400}


def _wire(
    monkeypatch: pytest.MonkeyPatch,
    job: dict[str, Any] | None = None,
    sizes: dict[str, int] | None = None,
    listing_none: bool = False,
    metadata_none: bool = False,
) -> None:
    """Patch the primitive's metadata and listing fetches."""
    payload = None if metadata_none else {"job": job if job is not None else _job()}
    monkeypatch.setattr(
        impl, "_fetch_metadata", lambda job_id, base_url, timeout: payload
    )
    result = None if listing_none else _listing(sizes if sizes is not None else _ALL_PRESENT)
    monkeypatch.setattr(
        impl, "_fetch_file_listing", lambda job_id, base_url, timeout: result
    )


def _run(observed: Any = None) -> dict[str, Any]:
    """Call :func:`plan_fetch` with the standard job and base URL."""
    return plan_fetch(_JOB_ID, _BASE_URL, 60, observed)


def _filenames(plan: dict[str, Any]) -> list[str]:
    """Return the filenames a plan asks for, in order."""
    return [entry["filename"] for entry in plan["next"]]


def _roles(plan: dict[str, Any]) -> list[str]:
    """Return the roles a plan assigns, in order."""
    return [entry["role"] for entry in plan["next"]]


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

def test_pilot_1305_selects_the_payload_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pilot error 1305 plans the payload-failure sequence."""
    _wire(monkeypatch)
    plan = _run()
    assert plan["strategy"] == STRATEGY_PAYLOAD_1305


def test_other_pilot_codes_select_the_pilotlog_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any code other than 1305 is diagnosed from the pilot log."""
    _wire(monkeypatch, job=_job(code=1151))
    plan = _run()
    assert plan["strategy"] == STRATEGY_PILOTLOG
    assert _filenames(plan) == [_PILOTLOG]
    assert _roles(plan) == [ROLE_PRIMARY]
    assert plan["done"] is True


def test_unparseable_pilot_code_is_not_1305(monkeypatch: pytest.MonkeyPatch) -> None:
    """A junk pilot error code coerces to 0, matching fetch_and_analyse."""
    _wire(monkeypatch, job={"jobstatus": "failed", "piloterrorcode": "not-a-number"})
    assert _run()["strategy"] == STRATEGY_PILOTLOG


def test_missing_pilot_code_is_not_1305(monkeypatch: pytest.MonkeyPatch) -> None:
    """An absent pilot error code coerces to 0."""
    _wire(monkeypatch, job={"jobstatus": "failed"})
    assert _run()["strategy"] == STRATEGY_PILOTLOG


@pytest.mark.parametrize("status", ["failed", "holding", "cancelled"])
def test_log_bearing_statuses_produce_a_fetch_plan(
    monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    """The three statuses fetch_and_analyse downloads logs for all plan a fetch."""
    _wire(monkeypatch, job=_job(status=status))
    plan = _run()
    assert plan["strategy"] == STRATEGY_PAYLOAD_1305
    assert _filenames(plan) == [SETUP_LOG]


@pytest.mark.parametrize("status", ["finished", "running", "closed", "merging", ""])
def test_other_statuses_plan_no_fetch_at_all(
    monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    """Statuses outside the log-bearing set are metadata-only, as in the monolith."""
    _wire(monkeypatch, job=_job(status=status))
    plan = _run()
    assert plan["strategy"] == STRATEGY_METADATA_ONLY
    assert plan["next"] == []
    assert plan["done"] is True
    assert plan["notes"]


def test_metadata_only_does_not_fetch_the_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The status gate short-circuits before the listing request."""
    calls: list[int] = []

    monkeypatch.setattr(
        impl,
        "_fetch_metadata",
        lambda job_id, base_url, timeout: {"job": _job(status="finished")},
    )

    def _listing_spy(job_id: int, base_url: str, timeout: int) -> None:
        calls.append(job_id)
        return None

    monkeypatch.setattr(impl, "_fetch_file_listing", _listing_spy)
    _run()
    assert calls == []


# ---------------------------------------------------------------------------
# Failure reporting
# ---------------------------------------------------------------------------

def test_metadata_fetch_failure_reports_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreachable metadata endpoint yields an error payload, not a plan."""
    _wire(monkeypatch, metadata_none=True)
    plan = _run()
    assert "error" in plan
    assert "strategy" not in plan
    assert plan["job_id"] == _JOB_ID
    # The two failure causes must stay distinguishable: an unreachable endpoint
    # is a problem with the monitor, a missing job is a problem with the job ID,
    # and they send the reader to different places.
    assert "metadata" in plan["error"].lower()
    assert "not found" not in plan["error"].lower()


def test_missing_job_reports_a_different_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Metadata with no ``job`` key means the job was not found."""
    monkeypatch.setattr(
        impl, "_fetch_metadata", lambda job_id, base_url, timeout: {"job": {}}
    )
    plan = _run()
    assert "error" in plan
    assert str(_JOB_ID) in plan["error"]
    assert "not found" in plan["error"].lower()


# ---------------------------------------------------------------------------
# The 1305 re-entrant loop
# ---------------------------------------------------------------------------

def test_first_call_asks_for_setup_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """setup.stdout is planned first and the plan is explicitly unfinished."""
    _wire(monkeypatch)
    plan = _run()
    assert _filenames(plan) == [SETUP_LOG]
    assert _roles(plan) == [ROLE_SETUP]
    assert plan["done"] is False


def test_setup_error_skips_the_payload_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reported setup error ends the plan: the payload never ran."""
    _wire(monkeypatch)
    plan = _run({"setup_has_error": True})
    assert plan["next"] == []
    assert plan["done"] is True
    assert any("skipped" in note for note in plan["notes"])


def test_clean_setup_falls_through_to_the_payload_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No setup error means payload.stdout then payload.stderr."""
    _wire(monkeypatch)
    plan = _run({"setup_has_error": False, "fetched": [SETUP_LOG]})
    assert _filenames(plan) == [PAYLOAD_STDOUT, PAYLOAD_STDERR]
    assert _roles(plan) == [ROLE_PRIMARY, ROLE_SECONDARY]
    assert plan["done"] is True


def test_setup_signal_as_the_string_false_falls_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``"false"`` must read as false; ``bool()`` would invert the decision."""
    _wire(monkeypatch)
    plan = _run({"setup_has_error": "false", "fetched": [SETUP_LOG]})
    assert _filenames(plan) == [PAYLOAD_STDOUT, PAYLOAD_STDERR]


def test_setup_signal_as_the_string_true_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    """``"true"`` is honoured as a setup error."""
    _wire(monkeypatch)
    assert _run({"setup_has_error": "true"})["next"] == []


def test_fetched_list_alone_marks_setup_as_seen(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller reporting only the filename still advances past the setup step.

    A fetched but empty setup.stdout yields no signal, and the monolith falls
    through to the payload logs in exactly that case.
    """
    _wire(monkeypatch)
    plan = _run({"fetched": [SETUP_LOG]})
    assert _filenames(plan) == [PAYLOAD_STDOUT, PAYLOAD_STDERR]


def test_terminal_replan_returns_an_empty_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller that re-plans after fetching everything is told there is nothing left."""
    _wire(monkeypatch)
    plan = _run(
        {
            "setup_has_error": False,
            "fetched": [SETUP_LOG, PAYLOAD_STDOUT, PAYLOAD_STDERR],
        }
    )
    assert plan["next"] == []
    assert plan["done"] is True


def test_the_loop_is_bounded_at_three_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """The naive ``while not done`` loop terminates within the documented bound."""
    _wire(monkeypatch)
    observed: dict[str, Any] = {"fetched": []}
    calls = 0
    order: list[str] = []

    while True:
        calls += 1
        assert calls <= 3, "plan_fetch loop exceeded its bound"
        plan = _run(observed)
        for entry in plan["next"]:
            order.append(entry["filename"])
            observed["fetched"].append(entry["filename"])
            if entry["filename"] == SETUP_LOG:
                observed["setup_has_error"] = False
        if plan["done"]:
            break

    assert order == [SETUP_LOG, PAYLOAD_STDOUT, PAYLOAD_STDERR]
    assert calls <= 3


def test_the_loop_is_bounded_when_setup_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """The early-return branch terminates too, one step sooner."""
    _wire(monkeypatch)
    observed: dict[str, Any] = {"fetched": []}
    calls = 0
    order: list[str] = []

    while True:
        calls += 1
        assert calls <= 3
        plan = _run(observed)
        for entry in plan["next"]:
            order.append(entry["filename"])
            observed["fetched"].append(entry["filename"])
            if entry["filename"] == SETUP_LOG:
                observed["setup_has_error"] = True
        if plan["done"]:
            break

    assert order == [SETUP_LOG]


# ---------------------------------------------------------------------------
# Zero-length skips and fail-open
# ---------------------------------------------------------------------------

def test_zero_length_setup_falls_through_in_one_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-byte setup.stdout is skipped without costing a round trip."""
    _wire(monkeypatch, sizes={**_ALL_PRESENT, SETUP_LOG: 0})
    plan = _run()
    assert _filenames(plan) == [PAYLOAD_STDOUT, PAYLOAD_STDERR]
    assert plan["done"] is True
    assert any(SETUP_LOG in note for note in plan["notes"])


def test_zero_length_stderr_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """payload.stderr is dropped when the index confirms it is empty."""
    _wire(monkeypatch, sizes={**_ALL_PRESENT, PAYLOAD_STDERR: 0})
    plan = _run({"setup_has_error": False})
    assert _filenames(plan) == [PAYLOAD_STDOUT]
    assert any(PAYLOAD_STDERR in note for note in plan["notes"])


def test_zero_length_payload_stdout_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """payload.stdout is dropped independently of payload.stderr."""
    _wire(monkeypatch, sizes={**_ALL_PRESENT, PAYLOAD_STDOUT: 0})
    plan = _run({"setup_has_error": False})
    assert _filenames(plan) == [PAYLOAD_STDERR]


def test_both_payload_logs_empty_yields_an_empty_terminal_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing usable means a finished plan with nothing to fetch."""
    _wire(monkeypatch, sizes={SETUP_LOG: 0, PAYLOAD_STDOUT: 0, PAYLOAD_STDERR: 0})
    plan = _run()
    assert plan["next"] == []
    assert plan["done"] is True


def test_zero_length_pilotlog_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pilotlog path skips its one file when the index says it is empty."""
    _wire(monkeypatch, job=_job(code=1151), sizes={**_ALL_PRESENT, _PILOTLOG: 0})
    plan = _run()
    assert plan["next"] == []
    assert plan["done"] is True
    assert any(_PILOTLOG in note for note in plan["notes"])


def test_unavailable_listing_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``None`` listing must offer files, not suppress them."""
    _wire(monkeypatch, listing_none=True)
    assert _filenames(_run()) == [SETUP_LOG]
    assert _filenames(_run({"setup_has_error": False})) == [
        PAYLOAD_STDOUT,
        PAYLOAD_STDERR,
    ]


def test_unlisted_file_is_still_offered(monkeypatch: pytest.MonkeyPatch) -> None:
    """A file absent from the listing is attempted, as in _file_is_nonempty."""
    _wire(monkeypatch, sizes={PAYLOAD_STDOUT: 200})
    assert _filenames(_run()) == [SETUP_LOG]


def test_nested_namesake_cannot_answer_for_the_root_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-byte ``workDir/setup.stdout`` must not mark the root file empty."""
    _wire(
        monkeypatch,
        sizes={**_ALL_PRESENT, SETUP_LOG: 0, f"workDir/{SETUP_LOG}": 4096},
    )
    plan = _run()
    # The root file is empty; the nested one is not.  The root wins.
    assert _filenames(plan) == [PAYLOAD_STDOUT, PAYLOAD_STDERR]


def test_nested_namesake_does_not_mask_a_good_root_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The inverse: a zero-byte nested copy must not suppress a usable root file."""
    _wire(monkeypatch, sizes={**_ALL_PRESENT, f"workDir/{SETUP_LOG}": 0})
    assert _filenames(_run()) == [SETUP_LOG]


# ---------------------------------------------------------------------------
# Tolerant reading of ``observed``
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("observed", [None, [], "setup.stdout", 7, object()])
def test_non_mapping_observed_reads_as_nothing_fetched(
    monkeypatch: pytest.MonkeyPatch, observed: Any
) -> None:
    """An unusable ``observed`` degrades to the first step rather than raising."""
    _wire(monkeypatch)
    assert _filenames(_run(observed)) == [SETUP_LOG]


@pytest.mark.parametrize("fetched", ["setup.stdout", 7, None, {"a": 1}])
def test_wrong_typed_fetched_is_ignored(
    monkeypatch: pytest.MonkeyPatch, fetched: Any
) -> None:
    """A bare string must not be iterated, nor substring-matched, into a filename."""
    _wire(monkeypatch)
    assert _filenames(_run({"fetched": fetched})) == [SETUP_LOG]


def test_non_string_members_of_fetched_are_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One junk member must not discard the usable ones."""
    _wire(monkeypatch)
    plan = _run({"fetched": [SETUP_LOG, 7, None], "setup_has_error": False})
    assert _filenames(plan) == [PAYLOAD_STDOUT, PAYLOAD_STDERR]


def test_unhashable_member_of_fetched_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nested list or dict must be dropped, not fed to ``frozenset``.

    ``observed`` is an unvalidated tool argument and JSON permits nesting, so
    without the member filter ``frozenset`` raises ``TypeError: unhashable``
    and a malformed argument becomes a server-side crash rather than a plan.
    """
    _wire(monkeypatch)
    plan = _run({"fetched": [SETUP_LOG, {"a": 1}, [2]], "setup_has_error": False})
    assert _filenames(plan) == [PAYLOAD_STDOUT, PAYLOAD_STDERR]


def test_unknown_observed_keys_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Extra signal keys from a future fetch primitive pass through harmlessly."""
    _wire(monkeypatch)
    plan = _run({"setup_has_error": False, "bytes": 512, "truncated": True})
    assert _filenames(plan) == [PAYLOAD_STDOUT, PAYLOAD_STDERR]


def test_fetched_accepts_any_sequence_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tuples and sets are as acceptable as lists."""
    _wire(monkeypatch)
    for container in ([SETUP_LOG], (SETUP_LOG,), {SETUP_LOG}, frozenset({SETUP_LOG})):
        plan = _run({"fetched": container})
        assert _filenames(plan) == [PAYLOAD_STDOUT, PAYLOAD_STDERR]


# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------

def test_urls_use_the_shared_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Planned URLs are byte-identical to the ones the monolith records."""
    _wire(monkeypatch)
    entry = _run()["next"][0]
    assert entry["url"] == _log_file_url(_JOB_ID, SETUP_LOG, _BASE_URL)
    assert entry["url"] == (
        f"{_BASE_URL}/filebrowser/?pandaid={_JOB_ID}&json&filename={SETUP_LOG}"
    )


def test_planned_urls_match_the_monolith_for_the_same_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Role-to-URL mapping agrees with _LogFetchResult, which B7 relies on."""
    _wire(monkeypatch)

    monkeypatch.setattr(
        "askpanda_atlas.log_analysis_impl._fetch_log_text",
        lambda job_id, filename, base_url, timeout: "clean setup, no errors",
    )
    monolith = _fetch_logs_payload(
        _JOB_ID, 1305, "boom", dict(_ALL_PRESENT), _BASE_URL, 60
    )

    planned: dict[str, str] = {}
    for observed in (None, {"setup_has_error": False, "fetched": [SETUP_LOG]}):
        for entry in _run(observed)["next"]:
            planned[entry["role"]] = entry["url"]

    assert planned[ROLE_SETUP] == monolith.setup_log_url
    assert planned[ROLE_PRIMARY] == monolith.log_url
    assert planned[ROLE_SECONDARY] == monolith.stderr_url


# ---------------------------------------------------------------------------
# Definition
# ---------------------------------------------------------------------------

def test_definition_is_restricted_to_the_primitive_profile() -> None:
    """The primitive must not reach Bamboo's own planner catalog."""
    assert get_definition()["profiles"] == ["primitive"]


def test_definition_advertises_the_wire_name() -> None:
    """The advertised name matches the entry-point key, so both spellings agree."""
    assert get_definition()["name"] == "atlas.log.plan_fetch"


def test_definition_declares_an_output_schema() -> None:
    """Structured returns require the schema to be declared (D-6)."""
    assert get_definition()["outputSchema"]["type"] == "object"


def test_output_schema_has_no_top_level_required() -> None:
    """A required success key would make every error payload a protocol error."""
    assert "required" not in get_definition()["outputSchema"]


def test_output_schema_declares_error() -> None:
    """The error key must be expressible under the schema."""
    assert "error" in get_definition()["outputSchema"]["properties"]


def test_input_schema_requires_only_job_id() -> None:
    """observed and timeout are optional; job_id is not."""
    schema = get_definition()["inputSchema"]
    assert schema["required"] == ["job_id"]
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"job_id", "observed", "timeout"}


# ---------------------------------------------------------------------------
# The tool boundary: structured returns on every path
# ---------------------------------------------------------------------------

def _validate(payload: dict[str, Any]) -> None:
    """Validate a payload against the declared outputSchema."""
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(instance=payload, schema=get_definition()["outputSchema"])


def _call(arguments: Any) -> tuple[list[Any], dict[str, Any]]:
    """Invoke the tool and assert the CombinationContent shape."""
    result = asyncio.run(plan_fetch_tool.call(arguments))
    assert isinstance(result, tuple), "a tool with outputSchema must return a tuple"
    assert len(result) == 2
    content, structured = result
    assert isinstance(content, list)
    assert isinstance(structured, dict)
    assert json.loads(content[0]["text"]) == structured
    _validate(structured)
    return content, structured


def test_call_returns_structured_content_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The happy path carries the plan as structured content."""
    _wire(monkeypatch)
    monkeypatch.setattr(impl, "get_base_url", lambda: _BASE_URL)
    _, structured = _call({"job_id": _JOB_ID})
    assert structured["strategy"] == STRATEGY_PAYLOAD_1305


@pytest.mark.parametrize(
    "arguments",
    [
        "not a dict",
        None,
        [],
        {},
        {"job_id": None},
        {"job_id": "abc"},
        {"job_id": []},
    ],
)
def test_every_argument_error_path_returns_structured_content(
    monkeypatch: pytest.MonkeyPatch, arguments: Any
) -> None:
    """A bad argument must not become an opaque output-validation error."""
    monkeypatch.setattr(impl, "get_base_url", lambda: _BASE_URL)
    _, structured = _call(arguments)
    assert "error" in structured


def test_unexpected_exception_returns_structured_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception inside planning is reported, not raised through the SDK."""
    monkeypatch.setattr(impl, "get_base_url", lambda: _BASE_URL)

    def _boom(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("listing exploded")

    monkeypatch.setattr(impl, "plan_fetch", _boom)
    _, structured = _call({"job_id": _JOB_ID})
    assert "listing exploded" in structured["error"]
    assert structured["job_id"] == _JOB_ID


def test_call_honours_a_string_job_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Numeric strings are coerced, matching the monolith's tolerance."""
    _wire(monkeypatch)
    monkeypatch.setattr(impl, "get_base_url", lambda: _BASE_URL)
    _, structured = _call({"job_id": str(_JOB_ID)})
    assert structured["job_id"] == _JOB_ID


def test_call_passes_observed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """The observed argument reaches plan_fetch unmodified."""
    _wire(monkeypatch)
    monkeypatch.setattr(impl, "get_base_url", lambda: _BASE_URL)
    _, structured = _call(
        {"job_id": _JOB_ID, "observed": {"setup_has_error": True}}
    )
    assert structured["next"] == []
    assert structured["done"] is True


def test_bad_timeout_falls_back_to_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unparseable timeout must not fail the call."""
    _wire(monkeypatch)
    monkeypatch.setattr(impl, "get_base_url", lambda: _BASE_URL)
    seen: list[int] = []

    real = impl.plan_fetch

    def _spy(job_id: int, base_url: str, timeout: int, observed: Any = None) -> Any:
        seen.append(timeout)
        return real(job_id, base_url, timeout, observed)

    monkeypatch.setattr(impl, "plan_fetch", _spy)
    _call({"job_id": _JOB_ID, "timeout": "soon"})
    assert seen == [60]
