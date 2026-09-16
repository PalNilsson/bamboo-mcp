"""Tests for the argument-error result shape in ``bamboo.core``.

The MCP SDK rejects a result carrying no structured content from a tool that
advertises an ``outputSchema``, answering *"Output validation error:
outputSchema defined but no structured output returned"*.  Bamboo's own
argument validation runs after the SDK's and returns a plain text result, so
for such a tool a precise argument error would have been replaced by an opaque
protocol one.

These tests pin both halves of the fix: tools declaring an ``outputSchema``
get structured content, and every tool that does not — which is the entire
orchestrated surface — keeps the previous return shape byte for byte.
"""
from __future__ import annotations

from typing import Any

import pytest

from bamboo.core import TOOLS, _argument_error_result, _validate_arguments

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"error": {"type": "string"}, "ok": {"type": "boolean"}},
    "additionalProperties": False,
}


def test_tool_without_output_schema_returns_a_plain_list() -> None:
    """The orchestrated surface is unchanged: a one-element content list."""
    result = _argument_error_result("panda_job_status", "Required argument missing: 'job_id'.", {})
    assert isinstance(result, list)
    assert result[0]["type"] == "text"
    assert "panda_job_status" in result[0]["text"]
    assert "Required argument missing" in result[0]["text"]


def test_explicit_none_output_schema_returns_a_plain_list() -> None:
    """A definition carrying ``outputSchema: None`` is treated as not declaring one."""
    result = _argument_error_result("t", "bad", {"outputSchema": None})
    assert isinstance(result, list)


def test_tool_with_output_schema_returns_a_tuple() -> None:
    """A declared outputSchema means the error must also be structured."""
    result = _argument_error_result("atlas.log.plan_fetch", "bad", {"outputSchema": _SCHEMA})
    assert isinstance(result, tuple)
    assert len(result) == 2
    content, structured = result
    assert isinstance(content, list)
    assert isinstance(structured, dict)
    assert structured == {"error": content[0]["text"]}


def test_the_message_names_the_tool_and_the_fault() -> None:
    """Both halves carry the same actionable message."""
    result = _argument_error_result("atlas.log.plan_fetch", "Unexpected argument(s): ['x'].", {"outputSchema": _SCHEMA})
    _content, structured = result
    assert "atlas.log.plan_fetch" in structured["error"]
    assert "Unexpected argument(s)" in structured["error"]


def test_structured_error_validates_against_an_empty_object_schema() -> None:
    """The ``{"error": ...}`` shape must satisfy a schema with no required keys."""
    jsonschema = pytest.importorskip("jsonschema")
    _content, structured = _argument_error_result("t", "bad", {"outputSchema": _SCHEMA})
    jsonschema.validate(instance=structured, schema=_SCHEMA)


def test_structured_error_validates_against_the_primitive_schema() -> None:
    """The live consumer: the schema atlas.log.plan_fetch actually declares."""
    jsonschema = pytest.importorskip("jsonschema")
    primitives = pytest.importorskip("askpanda_atlas.log_primitives_impl")
    schema = primitives.get_definition()["outputSchema"]

    result = _argument_error_result("atlas.log.plan_fetch", "Required argument missing: 'job_id'.", {"outputSchema": schema})
    assert isinstance(result, tuple)
    jsonschema.validate(instance=result[1], schema=schema)


def test_no_builtin_tool_declares_an_output_schema_yet() -> None:
    """Guard: the orchestrated surface returns unstructured content throughout.

    ``bamboo_executor`` unpacks in-process tool results as ``result[0]["text"]``
    and calls only built-in and planner-visible tools.  The day a built-in
    declares an ``outputSchema`` it will start returning a tuple, which that
    unpacker reads as an empty dict rather than as evidence.  This test is the
    tripwire for that change, not a prohibition on it.
    """
    declaring = [
        name
        for name, tool in TOOLS.items()
        if isinstance(getattr(tool, "get_definition", lambda: None)(), dict)
        and tool.get_definition().get("outputSchema") is not None
    ]
    assert declaring == []


def test_validate_arguments_still_gates_the_path() -> None:
    """The helper is only reached when validation actually fails."""
    defn = {
        "inputSchema": {
            "type": "object",
            "properties": {"job_id": {"type": "integer"}},
            "required": ["job_id"],
            "additionalProperties": False,
        }
    }
    assert _validate_arguments(defn, {"job_id": 1}) is None
    assert _validate_arguments(defn, {}) is not None
    assert _validate_arguments(defn, {"job_id": 1, "x": 2}) is not None
