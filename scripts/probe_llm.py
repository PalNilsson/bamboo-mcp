#!/usr/bin/env python3
"""Check that the configured LLM provider is reachable and the key is valid.

Prints the provider and model each profile resolves to, then sends one
minimal request and reports the outcome.

    source bamboo_env.sh
    python scripts/probe_llm.py

The `bamboo_llm_probe` MCP tool does the same send, but reads a selector that
`bamboo.core.create_server()` installs, so calling it from a bare interpreter
answers ``not_configured`` — a message that reads like a configuration fault
and is really a bootstrap one.  This script builds the same selector directly
from `bamboo.config.Config`, which is also why it does not import `mcp`: a
connectivity check should not fail because the SDK version is wrong.

Profiles are printed whether or not the send succeeds, because the usual
mistake is a provider or model that silently resolved to something other than
the intended one — a typo in `LLM_FAST_PROVIDER` leaves that profile on the
default provider rather than raising.

Exit status is 0 when the provider answers, 1 otherwise, so this works in a
deployment smoke test.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "core", REPO_ROOT / "packages" / "askpanda_atlas"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from bamboo.config import Config  # noqa: E402
from bamboo.llm.config_loader import build_model_registry_from_config  # noqa: E402
from bamboo.llm.manager import LLMClientManager  # noqa: E402
from bamboo.llm.runtime import set_llm_manager, set_llm_selector  # noqa: E402
from bamboo.llm.selector import LLMSelector  # noqa: E402
from bamboo.tools.llm_probe import bamboo_llm_probe_tool  # noqa: E402

PROFILES = ("default", "fast", "reasoning")


def _bootstrap() -> Any:
    """Install an LLM selector and client manager into the runtime context.

    Mirrors phase 0 of :func:`bamboo.core.create_server` without building an
    MCP server.

    Returns:
        Any: The model registry, so the caller can report what each profile
        resolved to.
    """
    try:
        config_obj: Any = Config()
    except TypeError:
        config_obj = Config

    registry = build_model_registry_from_config(config_obj)
    set_llm_selector(
        LLMSelector(
            registry=registry,
            default_profile=getattr(config_obj, "LLM_DEFAULT_PROFILE", "default"),
            fast_profile=getattr(config_obj, "LLM_FAST_PROFILE", "fast"),
            reasoning_profile=getattr(config_obj, "LLM_REASONING_PROFILE", "reasoning"),
        )
    )
    set_llm_manager(LLMClientManager())
    return registry


def _describe_profiles(registry: Any) -> None:
    """Print the provider, model and base URL each profile resolved to.

    Args:
        registry: The model registry returned by :func:`_bootstrap`.
    """
    print("Resolved profiles:")
    for name in PROFILES:
        try:
            spec = registry.get(name)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"  {name:<10} <unresolved: {exc}>")
            continue
        base_url = getattr(spec, "base_url", None) or ""
        suffix = f"  base_url={base_url}" if base_url else ""
        print(f"  {name:<10} provider={spec.provider:<16} model={spec.model}{suffix}")
    print()


def _probe() -> dict[str, Any]:
    """Send one minimal request to the default profile.

    Returns:
        Dict[str, Any]: The probe tool's parsed result, or a synthesised
        ``probe_error`` record if its output could not be parsed — an
        unparseable probe is still a failed probe, and must not look like a
        pass.
    """
    content = asyncio.run(bamboo_llm_probe_tool.call({}))
    try:
        return json.loads(content[0]["text"])
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return {"status": "probe_error", "detail": f"unparseable probe output: {exc!r}"}


def main(argv: Sequence[str] | None = None) -> int:
    """Describe the configured profiles and probe the provider.

    Args:
        argv: Argument vector, or ``None`` to read ``sys.argv``.

    Returns:
        int: 0 if the provider answered, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the probe result line.",
    )
    args = parser.parse_args(argv)

    registry = _bootstrap()
    if not args.quiet:
        _describe_profiles(registry)

    result = _probe()
    status = str(result.get("status", "unknown"))
    detail = str(result.get("detail", ""))
    print(f"probe: {status}" + (f" — {detail}" if detail else ""))

    if status == "ok":
        return 0
    if status == "not_configured":
        print(
            "  No LLM is configured. Set LLM_DEFAULT_PROVIDER and LLM_DEFAULT_MODEL, "
            "and source bamboo_env.sh.",
            file=sys.stderr,
        )
    elif status == "config_error":
        print(
            "  Check the API key variable for this provider, and — for "
            "openai_compat — that ASKPANDA_OPENAI_COMPAT_BASE_URL is set and ends "
            "in /v1.",
            file=sys.stderr,
        )
    elif status == "auth_error":
        print("  The endpoint was reached but rejected the key.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
