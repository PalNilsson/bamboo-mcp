"""Guard that the installed mcp is one whose Server API ``core.py`` can use.

``core.build_server()`` registers handlers with the low-level decorator API
(``@app.list_tools()``, ``@app.call_tool()``, ``@app.list_prompts()``,
``@app.get_prompt()``).  mcp 2.0.0 removed those decorators, so
``requirements.txt`` pins ``mcp>=0.9.0,<2.0.0``.

The failure mode being guarded against is unusually quiet, for two compounding
reasons:

1. Those decorators run inside ``build_server()``, which the test suite never
   calls, so an incompatible mcp does not fail any test.
2. ``tests/conftest.py`` installs stub modules for ``mcp`` and friends via
   ``sys.modules.setdefault`` and assigns ``MagicMock`` to
   ``mcp.server.Server``.  ``setdefault`` on ``sys.modules`` only no-ops if the
   module has already been *imported* — not merely installed — so in a fresh
   session the stub shadows a genuine mcp installation as well.  Every test in
   the suite therefore runs against a mock ``Server`` regardless of which mcp
   version is present.

Together those mean the incompatibility first surfaced only as four pyright
``reportAttributeAccessIssue`` errors in CI, and only in CI, because the
requirement had no upper bound so a fresh install resolved 2.0.0 while developer
machines kept a 1.x.

Consequence for this module: it must not ``import mcp``.  An in-process import
inside the test session returns conftest's ``MagicMock``, whose class does not
carry the decorator attributes, so the assertions would fail for the wrong
reason.  The checks therefore go around the import system entirely —
distribution metadata for the version, and a subprocess with a clean interpreter
for the actual API surface.
"""
from __future__ import annotations

import json
import subprocess
import sys
from importlib import metadata

import pytest

#: Major version at which the low-level Server decorator API was removed.
_INCOMPATIBLE_MAJOR: int = 2

#: The decorator factories ``build_server()`` calls on the Server instance.
_REQUIRED_SERVER_DECORATORS: tuple[str, ...] = (
    "list_tools",
    "call_tool",
    "list_prompts",
    "get_prompt",
)

_PIN_ADVICE: str = (
    "requirements.txt pins mcp>=0.9.0,<2.0.0 because core.build_server() uses "
    "the low-level Server decorator API that mcp 2.0.0 removed. No other test "
    "can catch this: the decorators run inside build_server(), which the suite "
    "never calls, and tests/conftest.py stubs mcp.server.Server with a MagicMock "
    "anyway. If the pin has been lifted deliberately, port core.py to the new "
    "API before removing this test."
)


def _installed_mcp_version() -> str:
    """Return the installed mcp distribution version, skipping if absent.

    Read from distribution metadata rather than ``mcp.__version__`` so the result
    is unaffected by the stub modules ``tests/conftest.py`` places in
    ``sys.modules``.

    Returns:
        The version string of the installed ``mcp`` distribution.

    Raises:
        AssertionError: Never in practice.  See the comment below the
            ``pytest.skip()`` call for why the statement has to be written.
    """
    try:
        return metadata.version("mcp")
    except metadata.PackageNotFoundError:
        pytest.skip("mcp is not installed; nothing to verify.")
    # Unreachable.  pytest.skip() is annotated NoReturn, but pyright only sees
    # that annotation when it can resolve pytest, and the pre-commit hook runs
    # pyright from a Node environment against an interpreter that need not have
    # the test dependencies installed.  Unresolved, pytest.skip() is Unknown,
    # the except branch appears to fall through, and the declared -> str fails
    # with reportReturnType.  An explicit raise terminates the path whether or
    # not the annotation is visible.
    raise AssertionError("unreachable: pytest.skip() does not return")


def test_installed_mcp_major_version_is_supported() -> None:
    """The installed mcp must predate the removal of the Server decorator API."""
    version = _installed_mcp_version()

    major_text = version.split(".", 1)[0]
    if not major_text.isdigit():
        pytest.skip(f"Cannot parse mcp version {version!r} as major.minor.patch.")

    assert int(major_text) < _INCOMPATIBLE_MAJOR, (
        f"Installed mcp is {version}, but Bamboo requires "
        f"mcp<{_INCOMPATIBLE_MAJOR}.0.0. {_PIN_ADVICE}"
    )


def test_real_mcp_server_exposes_build_server_decorators() -> None:
    """The genuine mcp Server must expose every decorator ``build_server()`` uses.

    Runs in a subprocess with a clean interpreter so the real ``mcp.server`` is
    imported rather than the ``MagicMock`` that ``tests/conftest.py`` installs
    into ``sys.modules`` for the duration of the test session.  This is the check
    that matches production, where ``build_server()`` runs against the installed
    SDK with no stubs in place.
    """
    _installed_mcp_version()

    probe = (
        "import json\n"
        "from mcp.server import Server\n"
        f"names = {list(_REQUIRED_SERVER_DECORATORS)!r}\n"
        "print(json.dumps([n for n in names if not hasattr(Server, n)]))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    if completed.returncode != 0:
        pytest.fail(
            "Could not import mcp.server in a clean interpreter, so "
            "core.build_server() would fail at start-up too.\n"
            f"stderr:\n{completed.stderr.strip()}\n\n{_PIN_ADVICE}"
        )

    missing = json.loads(completed.stdout.strip() or "[]")
    assert not missing, (
        f"The installed mcp Server is missing {missing}. core.build_server() "
        f"registers handlers with these decorators, so the MCP server will fail "
        f"at start-up even though the rest of the suite passes. {_PIN_ADVICE}"
    )
