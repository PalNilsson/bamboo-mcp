"""The symbols bamboo-eval measures through must stay where it looks for them.

Decision E-22.  The evaluation framework lives in a separate repository
(``PalNilsson/bamboo-eval``) and depends on this one; the dependency is
one-way, so this repository cannot import it.  Without this file a rename here
would break the framework silently, and the breakage would surface in a
scheduled run on the other repository a fortnight later, in a stack trace that
looks like a bug in the framework.

So the coupling is pinned here instead, in this repository's own suite, where
it fails in the pull request that causes it.

The authoritative list is ``ENTRY_POINTS`` in ``bamboo_eval/production.py``.
This file deliberately duplicates it rather than importing it: importing would
make bamboo-eval a test dependency of bamboo-mcp and reverse the dependency
direction the split was chosen for.  The duplication is small, and
``bamboo-eval check-contract`` is the richer check for CI jobs that do install
the framework.

What this file does *not* check is behaviour.  It asserts that the symbols
exist and are callable with the signature the framework uses.  Whether they
still return what the framework believes they return is what the framework's
own parity test is for.
"""
from __future__ import annotations

import inspect

import pytest

#: Mirrors ENTRY_POINTS in bamboo_eval/production.py.  Each entry is
#: (module, attribute, why bamboo-eval calls it).  Keep the two in step: a
#: symbol removed from one and not the other is how this test stops meaning
#: anything.
EVAL_ENTRY_POINTS: tuple[tuple[str, str, str], ...] = (
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
        "bamboo.tools.tool_retrieval",
        "LexicalRetriever",
        "the shipped BM25 retriever under test",
    ),
    (
        "bamboo.tools._tool_retrieval_embedding",
        "catalog_fingerprint",
        "hashes the indexed catalogue text; every stored eval record cites it",
    ),
)


@pytest.mark.parametrize(
    ("module_name", "attribute", "why"),
    EVAL_ENTRY_POINTS,
    ids=[f"{m}.{a}" for m, a, _ in EVAL_ENTRY_POINTS],
)
def test_entry_point_exists(module_name: str, attribute: str, why: str) -> None:
    """The symbol bamboo-eval resolves is still importable under this name.

    Args:
        module_name: Dotted module path.
        attribute: Symbol name.
        why: What the framework uses it for, quoted back in the failure.
    """
    module = pytest.importorskip(module_name)
    assert hasattr(module, attribute), (
        f"{module_name}.{attribute} is gone. bamboo-eval calls it because it "
        f"{why}. Renaming it breaks the evaluation framework silently: update "
        f"ENTRY_POINTS in bamboo_eval/production.py and this file together, or "
        f"keep an alias."
    )


def test_collect_tool_catalog_still_takes_namespaces() -> None:
    """The catalogue entry point keeps the keyword the framework passes.

    bamboo-eval calls ``_collect_tool_catalog(namespaces=[...])``.  A change to
    positional-only, or a rename of the parameter, would be caught here rather
    than as an unexplained TypeError in another repository.
    """
    planner = pytest.importorskip("bamboo.tools.planner")
    signature = inspect.signature(planner._collect_tool_catalog)
    assert "namespaces" in signature.parameters
    assert signature.parameters["namespaces"].kind is not inspect.Parameter.POSITIONAL_ONLY


def test_routing_rule_exposes_tools_and_text() -> None:
    """The guidance metric reads ``rule.tools`` and ``rule.text``.

    It pairs each clause with the tools it names, and counts a case as
    uncovered when a surviving catalogue no longer carries the clause that
    explains its tool.  Both attribute names are part of the contract.
    """
    planner = pytest.importorskip("bamboo.tools.planner")
    rule = planner.RoutingRule(frozenset({"a"}), "- use a.")
    assert rule.tools == frozenset({"a"})
    assert rule.text == "- use a."


def test_routing_rules_for_plugin_returns_rules() -> None:
    """The ATLAS guidance is reachable by plugin id, as the framework asks."""
    planner = pytest.importorskip("bamboo.tools.planner")
    rules = planner.routing_rules_for_plugin("atlas")
    assert rules, "ATLAS routing guidance is empty; guidance coverage would be vacuous"
    assert all(hasattr(r, "tools") and hasattr(r, "text") for r in rules)
