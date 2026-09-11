"""Single source of truth for the tool-profile switch.

Bamboo's ``panda_log_analysis`` is a compound tool: one call runs metadata
fetch, file listing, log download, excerpt, classify and evidence bundling.
Agentic frameworks built around *code mode* — where the model writes code
composing small tool primitives rather than selecting one compound tool per
turn — need a primitive surface instead.

Rather than splitting the monolith, Bamboo exposes a **second advertised
surface** over the same implementation functions, selected by this switch.  The
monolith is untouched; the primitives are additional definitions that are simply
not advertised unless the primitive profile is active.

Two vocabularies would be one too many, so there is only one.  The server-level
``BAMBOO_TOOL_PROFILE`` environment variable and a tool definition's
``profiles`` key draw from the same names:

* ``orchestrated`` — the compound surface Bamboo's own planner drives.
* ``primitive`` — the fine-grained surface a code-mode agent composes.
* ``both`` — env var only.  A definition never names it; it is the union.

A definition that does not usefully name any profile is **profile-agnostic** and
advertised under every profile.  That is the state of every definition in the
tree today, which is what makes introducing this module a no-op, and it is also
the right long-term default: ``panda_job_status``, ``cric_query`` and
``bamboo_health`` are as useful to a code-mode agent as to the planner.  Only
the log-analysis compound/primitive pair is ever split.

Two properties are deliberate and load-bearing:

* **Advertising only.**  ``bamboo.core.call_tool`` is not gated by profile, and
  neither are ``bamboo_executor``'s in-process resolution or
  ``_tool_names``' alias map.  The profile controls catalog size and selection
  accuracy, not access; gating dispatch would risk the orchestrated path for no
  gain, and a primitive still needs a canonical name to key evidence under.
* **Fail-open.**  A definition whose ``profiles`` key cannot be understood is
  treated as profile-agnostic rather than as advertised-nowhere.  Under the
  opposite rule a typo such as ``["primitve"]`` would make a tool silently
  vanish from ``tools/list``, which is markedly harder to diagnose than one
  extra tool being advertised.
"""
from __future__ import annotations

import logging
import os
from typing import Any, cast

logger = logging.getLogger(__name__)

#: Environment variable naming the active server profile.
ENV_VAR: str = "BAMBOO_TOOL_PROFILE"

#: Key a tool definition uses to restrict the profiles it is advertised under.
#: Dropped from the wire by ``bamboo.core._to_wire_definition``; it is internal
#: metadata, not an MCP field.
DEFINITION_KEY: str = "profiles"

PROFILE_ORCHESTRATED: str = "orchestrated"
PROFILE_PRIMITIVE: str = "primitive"
PROFILE_BOTH: str = "both"

#: Profiles a tool *definition* may name.  ``both`` is excluded on purpose: it
#: describes a server configuration, not a surface a tool belongs to.
TOOL_PROFILES: frozenset[str] = frozenset({PROFILE_ORCHESTRATED, PROFILE_PRIMITIVE})

#: Profiles the environment variable accepts.
SERVER_PROFILES: frozenset[str] = TOOL_PROFILES | {PROFILE_BOTH}

DEFAULT_PROFILE: str = PROFILE_ORCHESTRATED

#: Server profile to the set of definition profiles it activates.
_EXPANSIONS: dict[str, frozenset[str]] = {
    PROFILE_ORCHESTRATED: frozenset({PROFILE_ORCHESTRATED}),
    PROFILE_PRIMITIVE: frozenset({PROFILE_PRIMITIVE}),
    PROFILE_BOTH: TOOL_PROFILES,
}

#: Raw environment values already warned about.  ``active_profile`` is called
#: once per ``tools/list`` and once per plan, so warning unconditionally would
#: let a single typo fill the log of a long-running server.
_WARNED: set[str] = set()


def expand_profile(profile: str) -> frozenset[str]:
    """Return the definition profiles a server profile activates.

    Args:
        profile: A server profile name.  Leading/trailing whitespace and case
            are ignored.

    Returns:
        The set of definition profiles that are active.  ``both`` expands to
        every profile; an unrecognised name expands as
        :data:`DEFAULT_PROFILE` does.  Validation and the operator-facing
        warning live in :func:`active_profile`, so this stays a pure mapping
        and the fallback here is only defence in depth.
    """
    return _EXPANSIONS.get(profile.strip().lower(), _EXPANSIONS[DEFAULT_PROFILE])


def active_profile() -> str:
    """Return the server profile named by the environment.

    Read at call time rather than at import, matching how ``list_tools``
    already reads ``ASKPANDA_PLUGIN``.  ``bamboo.config.Config`` is a frozen
    dataclass whose defaults bind at import, which would make this variable
    untestable without reimporting the module.

    Returns:
        One of :data:`SERVER_PROFILES`.  An unset or empty variable yields
        :data:`DEFAULT_PROFILE` silently; an unrecognised value yields it after
        a warning, logged once per distinct value.  Raising instead would turn
        a configuration typo into a server that fails on every listing.
    """
    raw: str = os.getenv(ENV_VAR, "")
    normalised: str = raw.strip().lower()
    if not normalised:
        return DEFAULT_PROFILE
    if normalised not in SERVER_PROFILES:
        if raw not in _WARNED:
            _WARNED.add(raw)
            logger.warning(
                "%s=%r is not a recognised tool profile (expected one of %s); "
                "falling back to %r.",
                ENV_VAR,
                raw,
                ", ".join(sorted(SERVER_PROFILES)),
                DEFAULT_PROFILE,
            )
        return DEFAULT_PROFILE
    return normalised


def active_profiles() -> frozenset[str]:
    """Return the definition profiles the environment activates.

    Returns:
        The expansion of :func:`active_profile`.  Callers that advertise a set
        of tools should compute this once and pass it to
        :func:`is_advertised` per definition, rather than re-reading the
        environment for every tool.
    """
    return expand_profile(active_profile())


def definition_profiles(defn: Any) -> frozenset[str]:
    """Return the profiles a tool definition restricts itself to.

    Args:
        defn: A tool definition as returned by ``get_definition()``.

    Returns:
        The recognised profile names the definition's ``profiles`` key holds,
        or an **empty set** when it names none.  Empty is the profile-agnostic
        signal and is reached by four routes that are deliberately not
        distinguished: the key is absent, its value is not one of the accepted
        sequence types, the sequence is empty, or no member is a recognised
        profile name.

        The accepted types are listed rather than tested for iterability, which
        is what rejects a bare ``"profiles": "primitive"`` before it can be
        iterated into characters.  (Those characters would match no profile
        either, so the two guards agree today; the type list is the one that
        would still hold if a one-character profile name were ever added.)
    """
    if not isinstance(defn, dict):
        return frozenset()

    raw: Any = cast("dict[str, Any]", defn).get(DEFINITION_KEY)
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return frozenset()

    named: set[str] = set()
    for item in cast("list[Any]", list(raw)):
        if not isinstance(item, str):
            continue
        candidate: str = item.strip().lower()
        if candidate in TOOL_PROFILES:
            named.add(candidate)
    return frozenset(named)


def is_advertised(defn: Any, profiles: frozenset[str]) -> bool:
    """Report whether a definition should be advertised under *profiles*.

    Args:
        defn: A tool definition as returned by ``get_definition()``.
        profiles: The active definition profiles, typically from
            :func:`active_profiles`.  A caller bound to one surface regardless
            of configuration — Bamboo's planner is — passes that surface
            directly instead.

    Returns:
        ``True`` when the definition is profile-agnostic, or when it names at
        least one of *profiles*.  ``False`` only when it names profiles and
        none of them is active.
    """
    declared: frozenset[str] = definition_profiles(defn)
    if not declared:
        return True
    return bool(declared & profiles)


def reset_profile_warnings() -> None:
    """Discard the record of already-warned environment values.

    Only needed by tests, which assert that a bad value warns once and would
    otherwise see the memo carried over from an earlier test.  Mirrors
    ``bamboo.tools._tool_names.reset_alias_cache``.

    Returns:
        None.
    """
    _WARNED.clear()


__all__ = [
    "DEFAULT_PROFILE",
    "DEFINITION_KEY",
    "ENV_VAR",
    "PROFILE_BOTH",
    "PROFILE_ORCHESTRATED",
    "PROFILE_PRIMITIVE",
    "SERVER_PROFILES",
    "TOOL_PROFILES",
    "active_profile",
    "active_profiles",
    "definition_profiles",
    "expand_profile",
    "is_advertised",
    "reset_profile_warnings",
]
