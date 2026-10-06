"""The BigPanDA credential must reach every copy of the HTTP and cache layers.

Three modules fetch from BigPanDA and three more cache the result, and they
are copies of one another kept in step by hand rather than generated:

- ``bamboo.tools._panda_http`` — canonical
- ``askpanda_atlas._fallback_http`` — the plugin's copy, used when core is absent
- ``askpanda_epic._fallback_http`` — the ePIC plugin's copy
- ``askpanda_atlas._cache`` / ``askpanda_epic._cache`` — likewise

``test_panda_http_sync.py`` already pins the first two against each other.
This module covers what that one cannot see, which is the ePIC copies, and it
exists because of a specific failure: when BigPanDA's filebrowser began
requiring a token, the fix had to be written five times, and three of the five
were only found by a test going red rather than by anyone remembering they
were there.

What is asserted is the *surface*, not the text. The copies diverge
legitimately — the ePIC cache has no binary-media section — so comparing
bodies would fail for the wrong reason. A credential that reaches one copy and
not another produces exactly the silent half-fixed state this guards against.
"""
from __future__ import annotations

import importlib
from typing import Any

import pytest

#: A URL that accepts the credential, for the header-shape tests.
_FILEBROWSER_URL: str = (
    "https://bigpanda.cern.ch/filebrowser/?pandaid=1&json&filename=pilotlog.txt"
)

#: The job metadata URL, which serves 200 unauthenticated and 401 when an
#: Authorization header is present.
_METADATA_URL: str = "https://bigpanda.cern.ch/job?pandaid=1&json"

#: Where a log download redirects; the header has to survive that hop.
_MEDIA_URL: str = (
    "https://bigpanda.cern.ch//media/filebrowser/806f3002/tarball/pilotlog.txt"
)

#: Modules that build BigPanDA request headers.
_HTTP_MODULES: tuple[str, ...] = (
    "bamboo.tools._panda_http",
    "askpanda_atlas._fallback_http",
    "askpanda_epic._fallback_http",
)

#: Modules that cache BigPanDA responses.
_CACHE_MODULES: tuple[str, ...] = (
    "askpanda_atlas._cache",
    "askpanda_epic._cache",
)


def _load(name: str) -> Any:
    """Import a module, skipping the test when the package is not installed.

    Args:
        name: Dotted module path.

    Returns:
        The imported module.
    """
    try:
        return importlib.import_module(name)
    except ImportError:  # pragma: no cover - plugin absent from this checkout
        pytest.skip(f"{name} is not importable in this checkout")
        raise AssertionError("unreachable")


@pytest.mark.parametrize("module_name", _HTTP_MODULES)
def test_every_http_copy_can_send_the_token(module_name: str) -> None:
    """Each copy exposes ``panda_monitor_headers`` with the same contract.

    Args:
        module_name: Dotted path of the module under test.
    """
    module = _load(module_name)
    assert hasattr(module, "panda_monitor_headers"), (
        f"{module_name} cannot send a BigPanDA token. Every copy of this "
        f"module must, or the plugin that uses it silently loses log access."
    )
    assert module.ENV_MONITOR_TOKEN == "PANDA_MONITOR_TOKEN"
    assert module.DEFAULT_MONITOR_TOKEN_SCHEME == "Token"


@pytest.mark.parametrize("module_name", _HTTP_MODULES)
def test_an_unset_token_adds_no_header(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unconfigured deployment behaves exactly as it did before.

    Args:
        module_name: Dotted path of the module under test.
        monkeypatch: Pytest fixture.
    """
    module = _load(module_name)
    monkeypatch.delenv("PANDA_MONITOR_TOKEN", raising=False)
    assert module.panda_monitor_headers(_FILEBROWSER_URL) == {}


@pytest.mark.parametrize("module_name", _HTTP_MODULES)
def test_a_set_token_becomes_a_token_header(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default scheme is ``Token`` in every copy.

    Not ``Bearer``: under that scheme BigPanDA expects an ATLAS IAM JWT and
    answers ``401 {"detail": "Invalid ATLAS IAM token: Not enough segments"}``
    for the opaque credential it actually issues.

    Args:
        module_name: Dotted path of the module under test.
        monkeypatch: Pytest fixture.
    """
    module = _load(module_name)
    monkeypatch.setenv("PANDA_MONITOR_TOKEN", "abc123")
    monkeypatch.delenv("PANDA_MONITOR_TOKEN_SCHEME", raising=False)
    assert module.panda_monitor_headers(_FILEBROWSER_URL) == {
        "Authorization": "Token abc123"
    }


@pytest.mark.parametrize("module_name", _HTTP_MODULES)
def test_an_empty_scheme_sends_the_raw_token(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The escape hatch for a deployment that wants the value unprefixed.

    The exact form BigPanDA expects is the one part of this that cannot be
    verified from the repository, so getting it wrong is a configuration
    change rather than a code change — in every copy, or the escape hatch is
    not an escape hatch.

    Args:
        module_name: Dotted path of the module under test.
        monkeypatch: Pytest fixture.
    """
    module = _load(module_name)
    monkeypatch.setenv("PANDA_MONITOR_TOKEN", "abc123")
    monkeypatch.setenv("PANDA_MONITOR_TOKEN_SCHEME", "")
    assert module.panda_monitor_headers(_FILEBROWSER_URL) == {
        "Authorization": "abc123"
    }


@pytest.mark.parametrize("module_name", _CACHE_MODULES)
def test_every_cache_copy_reports_and_retries_failures(module_name: str) -> None:
    """Each cache copy carries the reason surface and the bounded TTLs.

    Args:
        module_name: Dotted path of the module under test.
    """
    module = _load(module_name)

    assert hasattr(module, "LogFetch")
    assert hasattr(module, "cached_fetch_log_detailed")
    assert hasattr(module, "last_log_failure")

    # A 404 expires; success does not.  An unbounded 404 freezes a premature
    # look into the answer for the lifetime of the process.
    assert module.LOG_MISSING_TTL > 0
    assert module.LOG_MISSING_TTL != float("inf")
    assert module.LOG_TTL == float("inf")


@pytest.mark.parametrize("module_name", _HTTP_MODULES)
def test_the_metadata_endpoint_never_receives_the_token(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Job and task metadata must be requested without an Authorization header.

    This is not caution, it is a requirement. The metadata endpoint serves
    ``200`` unauthenticated and answers ``401`` when the header is present, so
    a credential helper that attaches the token to everything breaks every
    analysis before it reaches a log file — which is exactly what happened.

    Args:
        module_name: Dotted path of the module under test.
        monkeypatch: Pytest fixture.
    """
    module = _load(module_name)
    monkeypatch.setenv("PANDA_MONITOR_TOKEN", "abc123")

    assert module.panda_monitor_headers(_METADATA_URL) == {}
    assert module.url_needs_token(_METADATA_URL) is False


@pytest.mark.parametrize("module_name", _HTTP_MODULES)
def test_the_download_and_its_redirect_both_receive_the_token(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The filebrowser redirects to /media/, and the header must survive it.

    Args:
        module_name: Dotted path of the module under test.
        monkeypatch: Pytest fixture.
    """
    module = _load(module_name)
    monkeypatch.setenv("PANDA_MONITOR_TOKEN", "abc123")

    assert module.url_needs_token(_FILEBROWSER_URL) is True
    assert module.url_needs_token(_MEDIA_URL) is True
    assert module.panda_monitor_headers(_MEDIA_URL)["Authorization"]
