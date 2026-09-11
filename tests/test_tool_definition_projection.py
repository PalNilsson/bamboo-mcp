"""Tests for the ``tools/list`` definition projection in ``bamboo.core``.

``mcp.types.Tool`` sets ``model_config = ConfigDict(extra="allow")``, so any key
a tool definition carries is accepted by ``Tool(**d)`` and then serialised to
the client.  Bamboo definitions carry ``tags`` and ``examples``, neither of which
has a consumer anywhere in the tree, and a forthcoming ``profiles`` key will be
internal too.  :func:`bamboo.core._to_wire_definition` is the projection that
keeps them off the wire.

These tests exercise the projection directly rather than through ``list_tools``.
``tests/conftest.py`` replaces ``mcp.types.Tool`` with a ``MagicMock``, and
``inspect.isclass(MagicMock)`` is true, so ``list_tools`` returns mock instances
whose constructor arguments are not observable as a dict — the projection's
effect cannot be asserted from its output.
"""
from __future__ import annotations

from typing import Any

import bamboo.core as core


def test_internal_keys_are_dropped() -> None:
    """``tags`` and ``examples`` must not survive into the wire definition."""
    defn: dict[str, Any] = {
        "name": "panda_log_analysis",
        "description": "Diagnose why a PanDA job failed.",
        "inputSchema": {"type": "object", "properties": {}},
        "tags": ["atlas", "panda"],
        "examples": [{"job_id": 1}],
    }

    wire = core._to_wire_definition(defn)

    assert set(wire) == {"name", "description", "inputSchema"}


def test_unknown_future_keys_are_dropped() -> None:
    """A key the allowlist does not name is dropped, not passed through.

    This is the property that makes the projection a rule rather than a list of
    two removals: ``profiles`` is added by the tool-profile work and must not
    reach clients, and no edit to this function is needed for that to hold.
    """
    defn: dict[str, Any] = {
        "name": "atlas.log.plan_fetch",
        "description": "Advise which log files to fetch.",
        "inputSchema": {"type": "object"},
        "profiles": ["primitive"],
    }

    assert "profiles" not in core._to_wire_definition(defn)


def test_output_schema_survives_projection() -> None:
    """``outputSchema`` must reach the wire or output validation silently stops.

    The SDK reads ``outputSchema`` from the definition it cached at
    ``tools/list`` time to decide whether a tool's result is required to carry
    ``structuredContent``.  Strip it and the server accepts unstructured results
    from a tool that promised structured ones, with no error on either side.
    """
    defn: dict[str, Any] = {
        "name": "atlas.log.classify",
        "description": "Classify a failure from a log excerpt.",
        "inputSchema": {"type": "object"},
        "outputSchema": {"type": "object", "properties": {"failure_type": {}}},
        "tags": ["atlas"],
    }

    wire = core._to_wire_definition(defn)

    assert wire["outputSchema"] == {
        "type": "object",
        "properties": {"failure_type": {}},
    }


def test_allowlist_names_output_schema() -> None:
    """Guard the allowlist itself against a well-meaning trim.

    Asserted separately from the projection because the two fail differently:
    dropping ``outputSchema`` from :data:`~bamboo.core._MCP_TOOL_FIELDS` breaks
    output validation for every tool at once, whereas a projection bug is
    per-definition.
    """
    assert "outputSchema" in core._MCP_TOOL_FIELDS
    assert "inputSchema" in core._MCP_TOOL_FIELDS
    assert "name" in core._MCP_TOOL_FIELDS


def test_projection_does_not_mutate_its_input() -> None:
    """The projection must copy, because definitions are sometimes shared.

    ``PandaLogAnalysisTool`` caches ``get_definition()`` in ``self._def`` and
    returns the same object on every call, so an in-place filter would strip a
    tool's metadata permanently on the first ``tools/list``.
    """
    defn: dict[str, Any] = {
        "name": "t",
        "description": "d",
        "inputSchema": {},
        "tags": ["keep-me"],
    }

    wire = core._to_wire_definition(defn)

    assert defn["tags"] == ["keep-me"]
    assert wire is not defn


def test_malformed_definition_yields_empty_dict() -> None:
    """A non-dict definition is tolerated, matching the rest of discovery.

    ``_tool_names._advertised_name`` and ``planner._tool_def_from_obj`` both
    return a falsy value rather than raising when a plugin's ``get_definition``
    misbehaves, so that one bad plugin cannot break discovery for the others.
    """
    assert core._to_wire_definition(None) == {}  # type: ignore[arg-type]
    assert core._to_wire_definition("not a definition") == {}  # type: ignore[arg-type]


def test_every_builtin_definition_keeps_its_required_fields() -> None:
    """Real definitions must not lose the three fields MCP requires in practice.

    A guard on the live ``TOOLS`` registry rather than on a fixture: it is the
    projection applied to the definitions that actually ship, so a tool that
    spells a key differently (``input_schema``, say) fails here.
    """
    assert core.TOOLS, "TOOLS registry is empty; the guard would be vacuous."

    for name, tool in core.TOOLS.items():
        wire = core._to_wire_definition(tool.get_definition())
        assert wire.get("name"), f"{name} lost its name in projection"
        assert wire.get("description"), f"{name} lost its description in projection"
        assert "inputSchema" in wire, f"{name} lost its inputSchema in projection"
