"""Minimal BigPanDA HTTP helpers for standalone (no bamboo core) use.

Used only when ``bamboo.tools._panda_http`` is not importable, i.e. when
the plugin is installed without bamboo core.  Keep this in sync with the
canonical helpers in ``bamboo.tools._panda_http``.
"""
from __future__ import annotations

import os
from collections import Counter
from typing import Any

import requests


def get_base_url() -> str:
    """Return the BigPanDA base URL from the environment.

    Reads the ``PANDA_BASE_URL`` environment variable, falling back to
    the public BigPanDA instance. Trailing slashes are stripped so
    callers can safely append path segments with ``/``.

    Returns:
        The base URL string, without a trailing slash.
    """
    return os.getenv("PANDA_BASE_URL", "https://bigpanda.cern.ch").rstrip("/")


#: Environment variable holding the BigPanDA access token.
ENV_MONITOR_TOKEN: str = "PANDA_MONITOR_TOKEN"

#: Environment variable overriding the Authorization scheme.  Set it to an
#: empty string to send the token value bare, with no scheme prefix.
ENV_MONITOR_TOKEN_SCHEME: str = "PANDA_MONITOR_TOKEN_SCHEME"

#: Scheme used when :data:`ENV_MONITOR_TOKEN_SCHEME` is unset.
DEFAULT_MONITOR_TOKEN_SCHEME: str = "Bearer"


def panda_monitor_headers() -> dict[str, str]:
    """Return the Authorization header for BigPanDA, or an empty mapping.

    BigPanDA's filebrowser endpoint used to serve its ``&json`` form
    unauthenticated.  It no longer does: an unauthenticated request now gets
    ``401`` with ``{"error": "No token provided"}``, which means no log file
    and no file listing can be read without a credential.

    Read from the environment on every call rather than captured at import,
    so a token can be supplied or rotated without reaching into module state,
    and so a test can set one with ``monkeypatch.setenv``.

    The scheme is configurable because the exact form BigPanDA expects is the
    one part of this that is not verifiable from here: ``Bearer <token>``
    covers the usual case, and setting
    :data:`ENV_MONITOR_TOKEN_SCHEME` to an empty string sends the raw value
    for a deployment that wants the token unprefixed.  Getting it wrong is
    then a configuration change rather than a code change.

    Returns:
        ``{"Authorization": ...}`` when a token is configured, otherwise an
        empty dict — so an unconfigured deployment behaves exactly as before
        and callers can always splat the result into their headers.
    """
    token: str = os.getenv(ENV_MONITOR_TOKEN, "").strip()
    if not token:
        return {}
    scheme: str = os.getenv(
        ENV_MONITOR_TOKEN_SCHEME, DEFAULT_MONITOR_TOKEN_SCHEME
    ).strip()
    return {"Authorization": f"{scheme} {token}" if scheme else token}


def fetch_jsonish(
    url: str,
    timeout: int = 30,
) -> tuple[int, str, str, dict[str, Any] | None]:
    """Fetch a URL and return parsed response components.

    Sends a GET request with JSON and AskPanDA user-agent headers.
    Non-2xx responses and non-JSON bodies are returned with
    ``json_or_none`` set to ``None`` so callers can handle them
    uniformly.

    Args:
        url: The URL to fetch.
        timeout: HTTP timeout in seconds.

    Returns:
        A tuple of ``(status_code, content_type, body_text,
        parsed_json_or_none)``. ``parsed_json_or_none`` is ``None``
        when the response is not a 2xx JSON object.
    """
    resp = requests.get(
        url,
        timeout=timeout,
        headers={
            "Accept": "application/json",
            "User-Agent": "AskPanDA/1.0",
            **panda_monitor_headers(),
        },
        allow_redirects=True,
    )
    status = resp.status_code
    ctype = (resp.headers.get("content-type") or "").lower()
    text = resp.text or ""
    if status < 200 or status >= 300:
        return status, ctype, text, None
    try:
        data = resp.json()
        return status, ctype, text, data if isinstance(data, dict) else {"_data": data}
    except Exception:  # pylint: disable=broad-exception-caught
        return status, ctype, text, None


def job_counts_from_payload(payload: dict[str, Any]) -> dict[str, int]:
    """Count job statuses in a BigPanDA task payload.

    Looks for a job list under the keys ``jobs``, ``jobList``, or
    ``joblist`` and tallies each job's status string. Jobs with a
    missing or non-string status are silently skipped.

    Args:
        payload: Parsed JSON payload returned by the BigPanDA API.

    Returns:
        A mapping of status string to occurrence count, e.g.
        ``{"finished": 42, "failed": 3}``. Empty dict if no job list
        is found.
    """
    jobs: list[Any] | None = next(
        (payload.get(k) for k in ("jobs", "jobList", "joblist")
         if isinstance(payload.get(k), list)),
        None,
    )
    if not jobs:
        return {}
    statuses = [
        j.get("jobStatus") or j.get("status") or j.get("job_status")
        for j in jobs
        if isinstance(j, dict)
    ]
    return dict(Counter(s for s in statuses if isinstance(s, str) and s))


def datasets_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Summarise dataset statuses and file counts in a BigPanDA task payload.

    Aggregates per-dataset file counters and collects the five datasets
    with the most failed files for quick triage.

    Args:
        payload: Parsed JSON payload returned by the BigPanDA API.

    Returns:
        A dict with the following keys:

        - ``dataset_count`` (int): Total number of datasets.
        - ``status_counts`` (dict[str, int]): Tally of dataset statuses.
        - ``nfilesfailed_total`` (int): Sum of failed files across all datasets.
        - ``nfilesfinished_total`` (int): Sum of finished files.
        - ``nfileswaiting_total`` (int): Sum of waiting files.
        - ``nfilesmissing_total`` (int): Sum of missing files.
        - ``worst_datasets`` (list[dict]): Up to five datasets with the
          highest ``nfilesfailed``, sorted descending.

        Returns an empty dict if ``payload`` contains no ``datasets`` list.
    """
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        return {}
    sc: Counter[str] = Counter()
    totals: dict[str, int] = {"failed": 0, "finished": 0, "waiting": 0, "missing": 0}
    key_map: dict[str, str] = {
        "nfilesfailed": "failed",
        "nfilesfinished": "finished",
        "nfileswaiting": "waiting",
        "nfilesmissing": "missing",
    }
    worst: list[dict[str, Any]] = []
    for ds in datasets:
        if not isinstance(ds, dict):
            continue
        st = ds.get("status") or ""
        if st:
            sc[st] += 1
        for k, t in key_map.items():
            v = ds.get(k)
            if isinstance(v, int):
                totals[t] += v
        nff = ds.get("nfilesfailed")
        if isinstance(nff, int) and nff > 0:
            worst.append({
                "datasetname": ds.get("datasetname"),
                "status": ds.get("status"),
                "nfilesfailed": nff,
                "nfiles": ds.get("nfiles"),
            })
    worst = sorted(worst, key=lambda x: x.get("nfilesfailed") or 0, reverse=True)[:5]
    return {
        "dataset_count": len(datasets),
        "status_counts": dict(sc),
        "nfilesfailed_total": totals["failed"],
        "nfilesfinished_total": totals["finished"],
        "nfileswaiting_total": totals["waiting"],
        "nfilesmissing_total": totals["missing"],
        "worst_datasets": worst,
    }
