"""Tests that bamboo.tools._panda_http and askpanda_atlas._fallback_http stay in sync.

``_fallback_http`` is a copy of the canonical ``_panda_http`` module, used by
the plugin when bamboo core is not installed.  These tests guard against the
two drifting apart by comparing their public APIs and verifying that each
exported function produces identical results for the same inputs.
"""
from __future__ import annotations

import inspect
import pytest

import bamboo.tools._panda_http as core_http
import askpanda_atlas._fallback_http as fallback_http

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PUBLIC_FUNCTIONS = [
    "get_base_url",
    "fetch_jsonish",
    "job_counts_from_payload",
    "datasets_summary",
    "panda_monitor_headers",
]


def _public_names(module) -> set[str]:
    """Return the set of public callable names exported by a module."""
    return {
        name
        for name, obj in inspect.getmembers(module, inspect.isfunction)
        if not name.startswith("_") and obj.__module__ == module.__name__
    }


# ---------------------------------------------------------------------------
# API parity
# ---------------------------------------------------------------------------

def test_public_api_identical():
    """Both modules must export exactly the same set of public functions."""
    core_api = _public_names(core_http)
    fallback_api = _public_names(fallback_http)
    assert core_api == fallback_api, (
        f"API mismatch between _panda_http and _fallback_http.\n"
        f"  Only in core:     {core_api - fallback_api}\n"
        f"  Only in fallback: {fallback_api - core_api}"
    )


@pytest.mark.parametrize("name", PUBLIC_FUNCTIONS)
def test_signatures_match(name):
    """Each shared function must have the same signature in both modules."""
    core_sig = inspect.signature(getattr(core_http, name))
    fallback_sig = inspect.signature(getattr(fallback_http, name))
    assert core_sig == fallback_sig, (
        f"{name}: signature differs.\n"
        f"  core:     {core_sig}\n"
        f"  fallback: {fallback_sig}"
    )


# ---------------------------------------------------------------------------
# Behavioural parity — get_base_url
# ---------------------------------------------------------------------------

def test_get_base_url_default(monkeypatch):
    """Both modules return the same default URL when the env var is unset."""
    monkeypatch.delenv("PANDA_BASE_URL", raising=False)
    assert core_http.get_base_url() == fallback_http.get_base_url()


def test_get_base_url_custom(monkeypatch):
    """Both modules honour the PANDA_BASE_URL env var identically."""
    monkeypatch.setenv("PANDA_BASE_URL", "https://custom.example.com/")
    assert core_http.get_base_url() == fallback_http.get_base_url() == "https://custom.example.com"


# ---------------------------------------------------------------------------
# Behavioural parity — fetch_jsonish
# ---------------------------------------------------------------------------

class _DummyResp:
    def __init__(self, status_code=200, headers=None, text="{}", json_data=None):
        self.status_code = status_code
        self.headers = headers or {"content-type": "application/json"}
        self._text = text
        self._json_data = json_data

    @property
    def text(self):
        return self._text

    def json(self):
        if self._json_data is not None:
            return self._json_data
        raise ValueError("no JSON")


@pytest.mark.parametrize("scenario,resp_kwargs,url", [
    (
        "success_json",
        {"status_code": 200, "json_data": {"task": {"status": "finished"}},
         "text": '{"task": {"status": "finished"}}'},
        "https://bigpanda.cern.ch/task/1234/?json",
    ),
    (
        "http_404",
        {"status_code": 404, "headers": {"content-type": "text/html"}, "text": ""},
        "https://bigpanda.cern.ch/task/0/?json",
    ),
    (
        "non_json_body",
        {"status_code": 200, "headers": {"content-type": "text/html"}, "text": "<html/>"},
        "https://bigpanda.cern.ch/task/1/?json",
    ),
])
def test_fetch_jsonish_parity(monkeypatch, scenario, resp_kwargs, url):
    """fetch_jsonish must return identical results from both modules."""
    dummy = _DummyResp(**resp_kwargs)

    def fake_get(*args, **kwargs):
        return dummy

    monkeypatch.setattr("requests.get", fake_get)

    core_result = core_http.fetch_jsonish(url)
    fallback_result = fallback_http.fetch_jsonish(url)
    assert core_result == fallback_result, (
        f"fetch_jsonish differs for scenario '{scenario}':\n"
        f"  core:     {core_result}\n"
        f"  fallback: {fallback_result}"
    )


# ---------------------------------------------------------------------------
# Behavioural parity — job_counts_from_payload
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", [
    {},
    {"jobs": []},
    {"jobs": [{"jobStatus": "finished"}, {"jobStatus": "failed"}, {"jobStatus": "finished"}]},
    {"jobList": [{"status": "running"}, {"status": "running"}]},
    {"jobs": [{"jobStatus": None}, {}, {"jobStatus": "finished"}]},
])
def test_job_counts_parity(payload):
    """job_counts_from_payload must return identical results from both modules."""
    assert core_http.job_counts_from_payload(payload) == fallback_http.job_counts_from_payload(payload)


# ---------------------------------------------------------------------------
# Behavioural parity — datasets_summary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", [
    {},
    {"datasets": []},
    {
        "datasets": [
            {"status": "finished", "nfilesfailed": 0, "nfilesfinished": 10,
             "nfileswaiting": 0, "nfilesmissing": 0, "datasetname": "ds1", "nfiles": 10},
            {"status": "failed", "nfilesfailed": 3, "nfilesfinished": 7,
             "nfileswaiting": 0, "nfilesmissing": 0, "datasetname": "ds2", "nfiles": 10},
        ]
    },
    {
        "datasets": [
            {"status": "running", "nfilesfailed": 5, "datasetname": f"ds{i}", "nfiles": 20}
            for i in range(10)
        ]
    },
])
def test_datasets_summary_parity(payload):
    """datasets_summary must return identical results from both modules."""
    assert core_http.datasets_summary(payload) == fallback_http.datasets_summary(payload)
