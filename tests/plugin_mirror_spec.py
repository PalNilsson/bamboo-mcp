"""Canonical specification of the askpanda_atlas -> askpanda_epic file mirrors.

Some plugin modules are deliberately duplicated between ``askpanda_atlas`` and
``askpanda_epic`` rather than shared, because plugin packages must stay
independently installable and must not import each other or bamboo core at
module scope.  The duplicates differ only in experiment naming.

Duplication invites drift: a change applied to the ATLAS copy and forgotten in
the ePIC copy leaves two files that look the same and behave differently.  That
has already happened once — ``_strip_directory_listing`` was added to the ATLAS
``log_analysis_impl`` after the ePIC copy had been mirrored, and the divergence
surfaced only as a downstream type-check error against a stale symbol.

This module holds the substitutions as data so that:

- the mirror can be regenerated mechanically, and
- ``tests/test_plugin_mirror_parity.py`` can assert the copies are still in
  sync, turning silent drift into a test failure.

To add a mirrored file, add an entry to :data:`MIRRORS`.
"""
from __future__ import annotations

# Substitutions applied to the ATLAS source to produce the ePIC copy, in order.
# Longest/most specific first: plain "askpanda_atlas" -> "askpanda_epic" must run
# last so it cannot clobber a more specific phrase containing that token.
_LOG_ANALYSIS_SUBS: tuple[tuple[str, str], ...] = (
    ('"""ATLAS PanDA job log analysis tool', '"""ePIC PanDA job log analysis tool'),
    (
        "Fetches job metadata and pilot log from BigPanDA, extracts a relevant",
        "Fetches job metadata and pilot log from the PanDA monitor, extracts a relevant",
    ),
    (
        '"pilot log and error metadata from BigPanDA, extracts the "',
        '"pilot log and error metadata from the PanDA monitor, extracts the "',
    ),
    (
        'f"Job {job_id} was not found in BigPanDA."',
        'f"Job {job_id} was not found in the PanDA monitor."',
    ),
    (
        '"Failed to fetch job metadata from BigPanDA"',
        '"Failed to fetch job metadata from the PanDA monitor"',
    ),
    (
        "base_url: BigPanDA base URL (from environment or default).",
        "base_url: PanDA monitor base URL (from environment or default).",
    ),
    ("base_url: BigPanDA base URL.", "base_url: PanDA monitor base URL."),
    (
        '"""Fetch job metadata JSON from BigPanDA, using the in-process TTL cache.',
        '"""Fetch job metadata JSON from the PanDA monitor, using the in-process TTL cache.',
    ),
    (
        "BigPanDA's filebrowser JSON listing uses the following structure::",
        "The PanDA monitor's filebrowser JSON listing uses the following structure::",
    ),
    (
        "Fetches job metadata and pilot/payload logs directly from BigPanDA,",
        "Fetches job metadata and pilot/payload logs directly from the PanDA monitor,",
    ),
    (
        '[f"- [BigPanDA Monitor]({monitor_url})"]',
        '[f"- [PanDA Monitor]({monitor_url})"]',
    ),
    (
        '"tags": ["atlas", "panda", "bigpanda", "job", "log", "failure", "diagnosis"],',
        '"tags": ["epic", "eic", "panda", "job", "log", "failure", "diagnosis"],',
    ),
    (
        "# Some BigPanDA versions wrap the list under a",
        "# Some PanDA monitor versions wrap the list under a",
    ),
    (
        "The ``job`` dict from the BigPanDA metadata response.",
        "The ``job`` dict from the PanDA monitor metadata response.",
    ),
    # Core-dump probe.  The probe is experiment-neutral and runs in both copies,
    # but the analysis tool it offers as a follow-up (atlas.core_dump_analysis)
    # reconstructs the job's ATLAS release container and is registered for ATLAS
    # only.  Flipping this one flag disables the offer in the ePIC copy while
    # keeping the probe's evidence keys, so an ePIC user can still be told
    # whether a job produced a core dump but is never offered an analysis that
    # cannot be run.  This is the recorded intentional divergence; do not
    # hand-edit the constant in the ePIC copy.
    (
        "# an offer that cannot be accepted, so the ePIC mirror sets this to False (see\n"
        "# tests/plugin_mirror_spec.py).  The evidence keys are still populated for\n"
        '# both, since "does this job have a core dump" is a useful answer either way.\n'
        "_CORE_DUMP_ANALYSIS_AVAILABLE: bool = True",
        "# an offer that cannot be accepted, so this mirrored copy has it set to False\n"
        "# (see tests/plugin_mirror_spec.py) and the offer builder below is therefore\n"
        '# unreachable here.  The evidence keys are still populated, since "does this\n'
        '# job have a core dump" is a useful answer either way.\n'
        "_CORE_DUMP_ANALYSIS_AVAILABLE: bool = False",
    ),
    (
        "forward: BigPanDA omits the separator between ``dirname`` and ``name``",
        "forward: the PanDA monitor omits the separator between ``dirname`` and ``name``",
    ),
    (
        "the CERN SSO login page, so the query string must not be altered.",
        "the monitor's login page, so the query string must not be altered.",
    ),
    (
        "populated.  A large job log makes BigPanDA report a warning such as",
        "populated.  A large job log makes the PanDA monitor report a warning such as",
    ),
    # Code-mode note and the profile it is gated on.  The log primitives are
    # ATLAS-only (D-18: ePIC registers no primitive entry points, so a mirrored
    # copy would be dead code plus parity-test surface).  There is therefore no
    # ePIC alternative for the note to point at, and the whole block collapses
    # to a constant here.  This is a recorded intentional divergence; do not
    # hand-edit the mirrored copy.
    (
        "#: Appended only when the primitive surface is advertised alongside this tool.\n"
        "_PRIMITIVE_NOTE: str = (\n"
        '    " This runs the whole sequence in one call and returns bundled evidence; "\n'
        '    "the atlas.log.* primitives expose the same steps individually for an "\n'
        '    "agent that composes them itself. Prefer this one unless you need "\n'
        '    "per-step control."\n'
        ")\n"
        "\n"
        "\n"
        "def _primitive_surface_is_advertised() -> bool:",
        "#: Unreachable in this mirrored copy; see _primitive_surface_is_advertised.\n"
        '_PRIMITIVE_NOTE: str = ""\n'
        "\n"
        "\n"
        "def _primitive_surface_is_advertised() -> bool:",
    ),
    (
        '    """Report whether the log primitives are advertised alongside this tool.\n'
        "\n"
        "    Read at call time rather than baked in, and the reason is the planner\n"
        "    rather than tidiness: under the default ``orchestrated`` profile the\n"
        "    primitives are withheld, and ``_collect_tool_catalog`` — which is pinned\n"
        "    to that profile — would otherwise put four tool names into the planner\n"
        "    prompt that the planner cannot select.  Naming an unselectable tool there\n"
        "    is the known failure mode where a plan reaches for it and falls through to\n"
        '    RAG, answering "the documentation doesn\'t cover this" to a question that\n'
        "    had a perfectly good answer.\n"
        "\n"
        "    ``bamboo.tools._tool_profiles`` is imported here (deferred) so this module\n"
        "    stays importable when bamboo core is absent, which is the isolated-exercise\n"
        "    case ``_fallback_log_analysis`` serves.  The profile vocabulary is read\n"
        "    from that module rather than from the environment directly: two spellings\n"
        "    of ``primitive`` would be one too many.\n"
        "\n"
        "    Returns:\n"
        "        ``True`` when the primitive profile is active.  ``False`` when bamboo\n"
        "        core is unavailable — the conservative answer, since without it no\n"
        "        primitive is registered to compose either.\n"
        '    """\n'
        "    try:\n"
        "        from bamboo.tools._tool_profiles import (  # deferred — see docstring\n"
        "            PROFILE_PRIMITIVE,\n"
        "            active_profiles,\n"
        "        )\n"
        "    except Exception:  # pylint: disable=broad-exception-caught\n"
        "        return False\n"
        "    return PROFILE_PRIMITIVE in active_profiles()",
        '    """Report whether log primitives are advertised alongside this tool.\n'
        "\n"
        "    Returns:\n"
        "        Always ``False`` in this mirrored copy: the log primitives are\n"
        "        ATLAS-only, so there is no ePIC code-mode alternative to point at.\n"
        "        See tests/plugin_mirror_spec.py; do not hand-edit.\n"
        '    """\n'
        "    return False",
    ),
    # Profile restriction.  The ATLAS monolith withdraws from the primitive
    # surface because atlas.log.* replaces it there.  Nothing replaces the ePIC
    # copy, so withholding it under the primitive profile would leave an ePIC
    # code-mode agent with no log analysis at all; the mirrored copy therefore
    # stays profile-agnostic and is advertised under every profile.
    (
        "        # The compound surface only.  Under the primitive profile this tool is\n"
        "        # withheld and atlas.log.* replaces it; under ``both`` the description\n"
        "        # above says which to reach for.  Advertising only — call_tool serves\n"
        "        # it under every profile, so the TUI, the REST facade and\n"
        "        # bamboo_executor are unaffected.\n"
        '        "profiles": ["orchestrated"],\n',
        "        # Deliberately profile-agnostic: no ePIC primitive replaces this tool,\n"
        "        # so restricting it to the orchestrated surface would leave an ePIC\n"
        "        # code-mode agent with no log analysis at all.  See\n"
        "        # tests/plugin_mirror_spec.py; do not hand-edit.\n",
    ),
    ("askpanda_atlas", "askpanda_epic"),
)

_TRACEBACK_PARSE_SUBS: tuple[tuple[str, str], ...] = (
    ("traceback parsing — ATLAS plugin copy", "traceback parsing — ePIC plugin copy"),
    (
        ":mod:`askpanda_atlas.log_analysis_impl` to locate",
        ":mod:`askpanda_epic.log_analysis_impl` to locate",
    ),
    (
        "This file is intentionally duplicated in ``askpanda_epic`` (kept byte-identical",
        "This file is intentionally duplicated in ``askpanda_atlas`` (kept byte-identical",
    ),
    (
        "be mirrored in ``askpanda_epic/_traceback_parse.py``.",
        "be mirrored in ``askpanda_atlas/_traceback_parse.py``.",
    ),
)

#: Mirrored files as (atlas_relative_path, epic_relative_path, substitutions).
#: Paths are relative to the repository root.
MIRRORS: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = (
    (
        "packages/askpanda_atlas/askpanda_atlas/log_analysis_impl.py",
        "packages/askpanda_epic/askpanda_epic/log_analysis_impl.py",
        _LOG_ANALYSIS_SUBS,
    ),
    (
        "packages/askpanda_atlas/askpanda_atlas/_traceback_parse.py",
        "packages/askpanda_epic/askpanda_epic/_traceback_parse.py",
        _TRACEBACK_PARSE_SUBS,
    ),
)


def render_epic_copy(atlas_source: str, subs: tuple[tuple[str, str], ...]) -> str:
    """Apply mirror substitutions to ATLAS source to produce the ePIC copy.

    Args:
        atlas_source: Full text of the ATLAS module.
        subs: Ordered ``(find, replace)`` pairs for this mirror.

    Returns:
        The expected text of the corresponding ePIC module.
    """
    result = atlas_source
    for find, replace in subs:
        result = result.replace(find, replace)
    return result


__all__ = ["MIRRORS", "render_epic_copy"]
