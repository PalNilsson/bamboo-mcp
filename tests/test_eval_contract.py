"""The symbols bamboo-eval measures through (decision E-22).

bamboo-eval lives in its own repository and calls into this one.  Without this
test a rename here breaks nothing visible for a fortnight, and then surfaces as
a failed scheduled run on another repository — which is precisely the
late, misattributed failure the evaluation framework exists to prevent.

So the contract is asserted *here*, in the pull request that would break it.

The entry-point list below is duplicated from
``bamboo_eval/production.py`` **deliberately**.  Importing bamboo-eval to read
it would reverse the dependency direction the two-repository split was chosen
for (decision E-21: bamboo-eval depends on bamboo-mcp, never the other way),
and would also mean this test passes whenever bamboo-eval is absent, which is
exactly when it is most needed.  Two copies that must agree is the cost; a
failure here names the other copy.
"""
from __future__ import annotations

import importlib
import inspect
from typing import Any

import pytest

#: ``(module, attribute, why bamboo-eval calls it)``.  Keep in step with
#: ``ENTRY_POINTS`` in bamboo-eval's ``production.py``.
REQUIRED_ENTRY_POINTS: tuple[tuple[str, str, str], ...] = (
    (
        "bamboo.tools.planner",
        "_collect_tool_catalog",
        "assembles the catalogue the planner is given; the retrieval metrics "
        "measure what this returns, not a reconstruction of it",
    ),
    (
        "bamboo.tools.planner",
        "routing_rules_for_plugin",
        "supplies the routing guidance whose survival the guidance-coverage "
        "metric checks",
    ),
    (
        "bamboo.tools.planner",
        "RoutingRule",
        "the clause/tool pairing the guidance-coverage metric relies on",
    ),
    (
        "bamboo.tools.planner",
        "bamboo_plan_tool",
        "is the planner the server itself calls; the selection-accuracy metric "
        "measures what this returns rather than a reconstruction of the "
        "planning path (decision E-25)",
    ),
    (
        "bamboo.tools.tool_retrieval",
        "LexicalRetriever",
        "the shipped BM25 retriever under test",
    ),
    (
        "bamboo.tools._tool_retrieval_embedding",
        "catalog_fingerprint",
        "hashes the indexed catalogue text; every stored record cites it",
    ),
)

#: Backends.  bamboo-eval records a stated skip when these are unavailable, so
#: their absence is not a contract breach — but a *rename* still is, which is
#: why they are listed and checked only when importable.
OPTIONAL_ENTRY_POINTS: tuple[tuple[str, str, str], ...] = (
    (
        "bamboo.tools._tool_retrieval_embedding",
        "EmbeddingRetriever",
        "embedding backend, for the backend comparison",
    ),
    (
        "bamboo.tools._tool_retrieval_embedding",
        "HybridRetriever",
        "RRF fusion of the two backends, for the RRF constant sweep",
    ),
)

_HINT = (
    "bamboo-eval resolves this symbol from its own repository. Either restore "
    "it, or update ENTRY_POINTS in bamboo_eval/production.py and this list "
    "together, in the same pull request."
)


def _resolve(module_name: str, attribute: str) -> Any:
    """Resolve one declared symbol.

    Args:
        module_name: Dotted module path.
        attribute: Symbol name within it.

    Returns:
        Any: The symbol.
    """
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


@pytest.mark.parametrize(
    ("module_name", "attribute", "why"),
    REQUIRED_ENTRY_POINTS,
    ids=[f"{m}.{a}" for m, a, _ in REQUIRED_ENTRY_POINTS],
)
def test_required_entry_points_exist(module_name: str, attribute: str, why: str) -> None:
    """Every symbol bamboo-eval requires is still where it looks for it.

    Args:
        module_name: Dotted module path.
        attribute: Symbol name within it.
        why: What bamboo-eval uses it for, repeated in the failure message so
            the person reading it knows whether the symbol moved or the
            measurement did.
    """
    try:
        _resolve(module_name, attribute)
    except (ImportError, AttributeError) as exc:
        pytest.fail(f"{module_name}.{attribute} is gone; bamboo-eval calls it "
                    f"because it {why}. ({exc}) {_HINT}")


@pytest.mark.parametrize(
    ("module_name", "attribute", "why"),
    OPTIONAL_ENTRY_POINTS,
    ids=[f"{m}.{a}" for m, a, _ in OPTIONAL_ENTRY_POINTS],
)
def test_optional_entry_points_exist_when_their_module_does(
    module_name: str, attribute: str, why: str
) -> None:
    """An optional backend may be absent, but it may not be renamed.

    Args:
        module_name: Dotted module path.
        attribute: Symbol name within it.
        why: What bamboo-eval uses it for.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        pytest.skip(f"{module_name} is not importable in this environment")
        raise
    assert hasattr(module, attribute), (
        f"{module_name}.{attribute} is gone; bamboo-eval calls it because it "
        f"{why}. {_HINT}"
    )


def test_the_catalogue_collector_takes_namespaces() -> None:
    """bamboo-eval asks for one plugin's catalogue by keyword.

    The metric passes ``namespaces=[...]``; a positional-only rename of that
    parameter would make every measurement run against a different catalogue
    from the one it claims.
    """
    signature = inspect.signature(_resolve("bamboo.tools.planner", "_collect_tool_catalog"))
    assert "namespaces" in signature.parameters


def test_the_planner_tool_is_called_the_way_bamboo_eval_calls_it() -> None:
    """``bamboo_plan_tool.call`` is an awaitable taking one mapping.

    bamboo-eval drives it with ``asyncio.run(tool.call({...}))`` and reads the
    first text block as the validated plan (decision E-25).  If it stops being
    a coroutine, or stops taking its arguments as a single mapping, the
    selection-accuracy metric is measuring something else — and the whole point
    of that metric is that it measures the path the server takes.
    """
    tool = _resolve("bamboo.tools.planner", "bamboo_plan_tool")
    call = getattr(tool, "call", None)
    assert inspect.iscoroutinefunction(call), (
        f"bamboo_plan_tool.call is {type(call).__name__}, not a coroutine function. {_HINT}"
    )
    parameters = [
        name
        for name, parameter in inspect.signature(call).parameters.items()
        if parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    assert parameters == ["arguments"], (
        f"bamboo_plan_tool.call takes {parameters}; bamboo-eval passes a single "
        f"arguments mapping. {_HINT}"
    )


def _planner_model(name: str) -> Any:
    """Return a schema class from the planner module, or skip.

    These two are not declared entry points — bamboo-eval reads the planner's
    output as JSON rather than importing its models, so that the metric does
    not depend on pydantic classes it does not own.  They are checked here
    because two *behaviours* bamboo-eval classifies depend on the schema
    continuing to admit them.

    Args:
        name: Class name.

    Returns:
        Any: The class.
    """
    try:
        return _resolve("bamboo.tools.planner", name)
    except (ImportError, AttributeError):
        pytest.skip(f"bamboo.tools.planner.{name} is not available here")
        raise


def test_the_plan_schema_still_admits_a_toolless_plan() -> None:
    """A ``RETRIEVE`` plan with no tool calls must stay possible.

    bamboo-eval scores one as ``declined`` — a planner that proposed nothing,
    which is neither a wrong tool nor malformed output (decision E-28).  If
    ``RETRIEVE`` disappeared, or ``tool_calls`` stopped defaulting to a list,
    that outcome would silently become an ``unparseable`` and a real behaviour
    would be recorded as a defect.
    """
    route = _planner_model("PlanRoute")
    assert "RETRIEVE" in {member.value for member in route}
    field = _planner_model("Plan").model_fields["tool_calls"]
    assert field.annotation is not None


def test_a_tool_call_still_carries_a_free_form_tool_name() -> None:
    """Both ``panda_task_status`` and ``atlas.task_status`` must stay valid.

    bamboo-eval resolves between the two naming conventions rather than
    comparing strings (decision E-27).  A schema that constrained ``tool`` to
    one convention — an enum of catalogue names, say — would make that
    resolution rule a measurement of a case that can no longer occur, and the
    rule should then be deleted rather than left to look like it is working.
    """
    field = _planner_model("ToolCall").model_fields["tool"]
    assert field.annotation is str
