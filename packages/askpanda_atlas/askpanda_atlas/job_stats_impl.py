"""Implementation of ``panda_job_stats`` — NL queries against job stats data.

Translates a natural-language question into OpenSearch aggregation parameters,
executes either a single-value metric aggregation or a terms+sub-aggregation
(group-by) against the ``atlas_panda_job_stats-*`` index, and returns a
compact evidence dict structured for LLM synthesis by the Bamboo executor.

Pipeline
--------
1. LLM call (async): NL question → structured query parameters (JSON).
2. Parameter validation and defaults (including ``group_by`` / ``top_n``).
3. Synchronous OpenSearch aggregation query (wrapped in
   ``asyncio.to_thread``).
4. Evidence dict returned as MCP content.

Environment variables
---------------------
ASKPANDA_OPENSEARCH
    Password for OpenSearch HTTP auth.  **Required.**
ASKPANDA_OPENSEARCH_HOST
    Base URL of the OpenSearch cluster.
    Default: ``https://os-atlas.cern.ch/os``
ASKPANDA_OPENSEARCH_USER
    HTTP auth username.  Default: ``pilot-monitor-agent``
ASKPANDA_OPENSEARCH_CA
    Path to the CA certificate bundle.
    Default: ``/etc/pki/tls/certs/CERN-bundle.pem``
ASKPANDA_OPENSEARCH_VERIFY_CERTS
    Set to ``"false"`` to disable TLS verification (local dev).

Public surface
--------------
- ``get_definition()``           — MCP tool definition dict
- ``PandaJobStatsTool``          — MCP tool class
- ``panda_job_stats_tool``       — module-level singleton
- ``fetch_job_stats(...)``       — synchronous OpenSearch query
- ``parse_llm_params(...)``      — validate/normalise LLM-extracted params
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenSearch client factory
# ---------------------------------------------------------------------------


def _create_os_client() -> Any:
    """Create an authenticated OpenSearch client from environment variables.

    Delegates to the shared factory in :mod:`bamboo.llm.opensearch_client`
    using the ``ASKPANDA_OPENSEARCH`` read password.  Kept as a module-level
    function so tests can patch it by name.

    Returns:
        An authenticated :class:`opensearchpy.OpenSearch` client.

    Raises:
        RuntimeError: If ``ASKPANDA_OPENSEARCH`` is not set.
        ImportError: If ``opensearch-py`` is not installed.
    """
    from bamboo.llm.opensearch_client import create_os_client as _shared  # deferred

    password = os.environ.get("ASKPANDA_OPENSEARCH", "")
    if not password:
        raise RuntimeError(
            "Environment variable ASKPANDA_OPENSEARCH is not set. "
            "Set it to your OpenSearch password to enable job stats queries."
        )
    return _shared(password)


# ---------------------------------------------------------------------------
# Default time window
# ---------------------------------------------------------------------------


def _default_window() -> tuple[str, str]:
    """Return ISO-8601 strings for the default look-back window ending now.

    The window length is :data:`~job_stats_schema.DEFAULT_WINDOW_HOURS` hours.

    Returns:
        ``(from_dt_iso, to_dt_iso)`` formatted as ``YYYY-MM-DDTHH:MM:SS``.
    """
    from askpanda_atlas.job_stats_schema import DEFAULT_WINDOW_HOURS  # deferred

    now = datetime.now(tz=timezone.utc).replace(microsecond=0)
    start = now - timedelta(hours=DEFAULT_WINDOW_HOURS)
    return (
        start.strftime("%Y-%m-%dT%H:%M:%S"),
        now.strftime("%Y-%m-%dT%H:%M:%S"),
    )


# ---------------------------------------------------------------------------
# LLM parameter extraction
# ---------------------------------------------------------------------------


async def _call_llm_for_params(question: str) -> str:
    """Call the configured Bamboo LLM and return its raw reply.

    Uses the default model profile at temperature 0.0 with a tight token
    cap to minimise hallucination.  The reply is expected to be a JSON
    object or the ``CANNOT_ANSWER`` sentinel.

    Args:
        question: Natural-language question from the user.

    Returns:
        Raw reply string from the LLM.

    Raises:
        RuntimeError: If the LLM manager or selector is not initialised.
    """
    from bamboo.llm.runtime import get_llm_manager, get_llm_selector  # deferred
    from bamboo.llm.types import GenerateParams, Message  # deferred
    from askpanda_atlas.job_stats_schema import build_query_prompt  # deferred

    selector = get_llm_selector()
    manager = get_llm_manager()
    registry = getattr(selector, "registry", None)
    if registry is None:
        raise RuntimeError("LLM selector does not expose a registry.")

    default_profile = getattr(selector, "default_profile", "default")
    model_spec = registry.get(default_profile)
    client = await manager.get_client(model_spec)

    messages_raw = build_query_prompt(question)
    messages: list[Message] = [
        {"role": m["role"], "content": m["content"]}
        for m in messages_raw
    ]

    resp = await client.generate(
        messages=messages,
        params=GenerateParams(temperature=0.0, max_tokens=256),
    )
    return resp.text


def _strip_llm_fences(text: str) -> str:
    r"""Remove markdown code fences from *text*, returning the inner content.

    Handles ``\`\`\`json``, ``\`\`\`JSON``, and plain ``\`\`\``` wrappers.

    Args:
        text: Raw stripped string from the LLM.

    Returns:
        Content with code fences removed and whitespace stripped.
    """
    for fence in ("```json", "```JSON", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
            if text.endswith("```"):
                text = text[:-3]
            return text.strip()
    return text


def _is_cannot_answer(text: str) -> bool:
    """Return ``True`` when *text* signals the LLM cannot produce parameters.

    Args:
        text: Stripped LLM reply text (fences already removed).

    Returns:
        ``True`` if the reply matches the cannot-answer sentinel or a
        natural-language refusal phrase.
    """
    from askpanda_atlas.job_stats_schema import CANNOT_ANSWER_SENTINEL  # deferred

    if text.upper() == CANNOT_ANSWER_SENTINEL:
        return True
    lower = text.lower()
    refusals = (
        "i cannot", "i can't", "cannot answer", "unable to",
        "not possible", "i don't know",
    )
    return any(p in lower for p in refusals)


def _str_or_none(parsed: dict[str, Any], key: str) -> str | None:
    """Extract a string value from *parsed* by *key*, or return ``None``.

    Args:
        parsed: Decoded JSON dict from the LLM.
        key: Key to look up.

    Returns:
        Non-empty stripped string, or ``None``.
    """
    v = parsed.get(key)
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _int_or_none(parsed: dict[str, Any], key: str) -> int | None:
    """Extract an integer value from *parsed* by *key*, or return ``None``.

    Args:
        parsed: Decoded JSON dict from the LLM.
        key: Key to look up.

    Returns:
        Integer value, or ``None`` if missing or unconvertible.
    """
    v = parsed.get(key)
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def parse_llm_params(raw: str) -> dict[str, Any] | None:
    """Parse and validate LLM-extracted query parameters.

    Strips markdown code fences, detects ``CANNOT_ANSWER`` replies, then
    parses the JSON object and validates all keys against the schema.
    Unknown keys are silently dropped; missing optional keys are filled with
    defaults from :mod:`job_stats_schema`.

    Args:
        raw: Raw reply string from the LLM (may contain JSON or a sentinel).

    Returns:
        Validated parameter dict with keys ``metric``, ``field``,
        ``site``, ``jobstatus``, ``jeditaskid``, ``python_version``,
        ``os_version``, ``from_dt``, ``to_dt``, ``group_by``, and
        ``top_n``, or ``None`` when the LLM signalled it cannot answer.
    """
    from askpanda_atlas.job_stats_schema import (  # deferred
        DEFAULT_FIELD,
        DEFAULT_METRIC,
        KEYWORD_GROUP_BY_FIELDS,
        NUMERIC_FIELDS,
        VALID_METRICS,
    )

    text = _strip_llm_fences(raw.strip())

    if _is_cannot_answer(text):
        return None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.debug("panda_job_stats: LLM reply is not valid JSON: %r", text[:200])
        return None

    if not isinstance(parsed, dict):
        return None

    # Validate and normalise metric.
    metric = str(parsed.get("metric") or DEFAULT_METRIC).strip().lower()
    if metric not in VALID_METRICS:
        logger.debug("panda_job_stats: unknown metric %r, using default", metric)
        metric = DEFAULT_METRIC

    # Validate and normalise field.
    field = str(parsed.get("field") or DEFAULT_FIELD).strip()
    if field not in NUMERIC_FIELDS:
        logger.debug("panda_job_stats: non-numeric field %r, using default", field)
        field = DEFAULT_FIELD

    # Validate group_by — only keyword fields are permitted.
    raw_group_by = _str_or_none(parsed, "group_by")
    if raw_group_by is not None and raw_group_by not in KEYWORD_GROUP_BY_FIELDS:
        logger.debug(
            "panda_job_stats: unsupported group_by %r, ignoring", raw_group_by
        )
        raw_group_by = None

    # Validate top_n — must be a positive integer ≤ 20; default 5.
    raw_top_n = _int_or_none(parsed, "top_n")
    if raw_top_n is None or raw_top_n < 1:
        raw_top_n = 5
    top_n: int = min(raw_top_n, 20)

    # Validate order — "asc" or "desc"; default "desc".
    raw_order = _str_or_none(parsed, "order")
    if raw_order not in ("asc", "desc"):
        raw_order = "desc"

    return {
        "metric": metric,
        "field": field,
        "site": _str_or_none(parsed, "site"),
        "jobstatus": _str_or_none(parsed, "jobstatus"),
        "jeditaskid": _int_or_none(parsed, "jeditaskid"),
        "python_version": _str_or_none(parsed, "python_version"),
        "os_version": _str_or_none(parsed, "os_version"),
        "from_dt": _str_or_none(parsed, "from_dt"),
        "to_dt": _str_or_none(parsed, "to_dt"),
        "group_by": raw_group_by,
        "top_n": top_n,
        "order": raw_order,
    }


# ---------------------------------------------------------------------------
# Core OpenSearch query (synchronous)
# ---------------------------------------------------------------------------


def _total_hits(response: Any) -> int:
    """Return the total hit count of an OpenSearch response.

    ``hits.total`` arrives in one of two shapes.  Modern OpenSearch sends an
    object — ``{"value": N, "relation": "eq"}`` — but a cluster or client
    configured with ``rest_total_hits_as_int`` sends a bare integer, and
    opensearch-dsl surfaces whichever it got as a dynamic ``AttrDict``.  A
    ``hasattr(total, "value")`` test therefore reports **zero documents** for
    the integer form rather than the count, which looks like an empty result
    set rather than a parsing problem.

    Args:
        response: The object ``Search.execute()`` returned.

    Returns:
        The total hit count, or ``0`` when it cannot be read.
    """
    total: Any = getattr(getattr(response, "hits", None), "total", None)
    raw: Any = getattr(total, "value", total)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def fetch_job_stats(
    metric: str,
    field: str,
    site: str | None = None,
    jobstatus: str | None = None,
    jeditaskid: int | None = None,
    from_dt: str | None = None,
    to_dt: str | None = None,
    group_by: str | None = None,
    top_n: int = 5,
    order: str = "desc",
    python_version: str | None = None,
    os_version: str | None = None,
) -> dict[str, Any]:
    """Execute a metric aggregation against the job stats index.

    Supports two execution paths:

    * **Scalar path** (``group_by=None``): executes a single-value metric
      aggregation and returns a ``value`` key with the scalar result.
    * **Terms path** (``group_by`` is a keyword field): executes a terms
      aggregation bucketed by *group_by*, with a metric sub-aggregation per
      bucket sorted by *order*.  Returns a ``buckets`` key (list of
      ``{"key": ..., "value": ..., "doc_count": ...}`` dicts) instead of a
      scalar ``value``.

    Creates a fresh OpenSearch client per call.  Checks the module-level
    cache first; on a miss executes the query and caches the result for
    :data:`~job_stats_schema.CACHE_TTL_SECS` seconds.

    The query applies optional ``term``/``wildcard`` filters for
    ``computingsite``, ``jobstatus``, ``jeditaskid``, ``python_version``,
    and ``os_version``, and an optional ``range`` filter on ``@timestamp``.

    Args:
        metric: OpenSearch metric aggregation type (``avg``, ``sum``,
            ``min``, ``max``, or ``value_count``).
        field: Numeric field name to aggregate.
        site: Optional ``computingsite`` filter.  A wildcard query is used
            so partial site names work (e.g. ``"BNL"`` matches
            ``"BNL_ATLAS_1"``).
        jobstatus: Optional ``jobstatus`` filter, e.g. ``"finished"``.
        jeditaskid: Optional integer JEDI task ID filter.
        from_dt: Optional ISO-8601 lower bound on ``@timestamp``.
        to_dt: Optional ISO-8601 upper bound on ``@timestamp``.
        group_by: Optional keyword field to bucket by (terms aggregation).
            When set, the evidence dict contains a ``"buckets"`` list
            instead of a scalar ``"value"``.
        top_n: Number of buckets to return when *group_by* is set.
            Clamped to the range ``[1, 20]``; default is ``5``.
        order: Sort direction for group-by buckets: ``"desc"`` (default,
            highest-first) or ``"asc"`` (lowest-first, for "worst" /
            "lowest" queries).  Ignored when *group_by* is ``None``.
        python_version: Optional Python version prefix filter, e.g.
            ``"3.7"``.  Matches jobs whose ``python_version`` field starts
            with this prefix (e.g. ``"3.7"`` matches ``"3.7.9"``), since
            the stored value is a full version string but users typically
            refer to just the major/minor version.
        os_version: Optional OS version prefix filter, e.g. ``"7"`` or
            ``"9.7"``.  Same prefix-match semantics as *python_version*.

    Returns:
        Evidence dict with keys ``metric``, ``field``, ``group_by``,
        ``top_n``, ``order``, ``value`` (scalar path) or ``buckets``
        (terms path), ``doc_count``, ``site_filter``, ``jobstatus_filter``,
        ``jeditaskid_filter``, ``python_version_filter``, ``os_version_filter``,
        ``from_dt``, ``to_dt``, ``endpoint``, and ``error``.

    Raises:
        RuntimeError: If ``ASKPANDA_OPENSEARCH`` is not set or the query
            fails unrecoverably.
        ImportError: If ``opensearch-py`` or ``opensearch-dsl`` are not
            installed.
    """
    from askpanda_atlas._cache import _MISS, _get, _set  # deferred
    from askpanda_atlas.job_stats_schema import CACHE_PREFIX, CACHE_TTL_SECS, INDEX_PATTERN  # deferred

    top_n = max(1, min(top_n, 20))
    order = order if order in ("asc", "desc") else "desc"

    cache_key = (
        f"{CACHE_PREFIX}{metric}|{field}|{site or ''}|"
        f"{jobstatus or ''}|{jeditaskid or ''}|"
        f"{python_version or ''}|{os_version or ''}|"
        f"{from_dt or ''}|{to_dt or ''}|"
        f"{group_by or ''}|{top_n}|{order}"
    )
    cached = _get(cache_key)
    if cached is not _MISS:
        logger.debug("panda_job_stats: cache hit for %s", cache_key)
        return cached  # type: ignore[return-value]

    client = _create_os_client()

    from opensearch_dsl import Search  # type: ignore[import]  # deferred

    s = Search(using=client, index=INDEX_PATTERN).extra(size=0)

    # Time range filter on @timestamp (= statechangetime).
    if from_dt or to_dt:
        time_range: dict[str, str] = {}
        if from_dt:
            time_range["gte"] = from_dt
        if to_dt:
            time_range["lte"] = to_dt
        s = s.filter("range", **{"@timestamp": time_range})

    # Keyword filters.
    if jobstatus:
        s = s.filter("term", **{"jobstatus.keyword": jobstatus.lower()})
    if jeditaskid is not None:
        s = s.filter("term", jeditaskid=jeditaskid)

    # Version filters: prefix match (not substring) so "3.7" matches
    # "3.7.9" but not "13.7.0" or "9.3.7" — the stored value is a full
    # version string (e.g. "3.9.22") but users typically refer to just the
    # major/minor version they care about.
    if python_version:
        s = s.filter("wildcard", **{"python_version.keyword": f"{python_version}*"})
    if os_version:
        s = s.filter("wildcard", **{"os_version.keyword": f"{os_version}*"})

    # Site filter: wildcard on computingsite.keyword so partial names work.
    if site:
        s = s.filter("wildcard", **{"computingsite.keyword": f"*{site}*"})

    if group_by:
        # ── Terms + sub-aggregation path ─────────────────────────────────
        # Bucket by group_by field (uses the .keyword sub-field for keyword
        # fields).  Sort by sub-metric in the requested direction so
        # "highest" (desc) and "worst/lowest" (asc) both work correctly.
        group_field = f"{group_by}.keyword"
        # opensearch-dsl builds aggregations through __getattr__, so `s.aggs`
        # is typed as a plain AttrDict and the fluent bucket().metric() chain
        # reads as calling a mapping.  Bind it as Any rather than scatter
        # ignores across the chain.
        aggs: Any = s.aggs
        (
            aggs
            .bucket("by_group", "terms",
                    field=group_field,
                    size=top_n,
                    order={"sub_metric": order})
            .metric("sub_metric", metric, field=field)
        )

        response = s.execute()

        doc_count: int = _total_hits(response)
        buckets = [
            {
                "key": b.key,
                "value": b.sub_metric.value,
                "doc_count": b.doc_count,
            }
            for b in response.aggregations.by_group.buckets
        ]

        evidence: dict[str, Any] = {
            "metric": metric,
            "field": field,
            "group_by": group_by,
            "top_n": top_n,
            "order": order,
            "buckets": buckets,
            "value": None,          # not used in group_by path
            "doc_count": doc_count,
            "site_filter": site,
            "jobstatus_filter": jobstatus,
            "jeditaskid_filter": jeditaskid,
            "python_version_filter": python_version,
            "os_version_filter": os_version,
            "from_dt": from_dt,
            "to_dt": to_dt,
            "endpoint": INDEX_PATTERN,
            "error": None,
        }

        _set(cache_key, evidence, CACHE_TTL_SECS)
        logger.debug(
            "panda_job_stats: %s(%s) grouped by %s  order=%s  top_n=%d  "
            "buckets=%d  doc_count=%d",
            metric, field, group_by, order, top_n, len(buckets), doc_count,
        )
        return evidence

    # ── Single-value scalar path ──────────────────────────────────────────
    agg_name = "stats_value"
    s.aggs.metric(agg_name, metric, field=field)

    response = s.execute()

    doc_count = _total_hits(response)
    agg = response.aggregations[agg_name]
    value: float | int | None = getattr(agg, "value", None)

    evidence = {
        "metric": metric,
        "field": field,
        "group_by": None,
        "top_n": None,
        "order": None,
        "buckets": None,
        "value": value,
        "doc_count": doc_count,
        "site_filter": site,
        "jobstatus_filter": jobstatus,
        "jeditaskid_filter": jeditaskid,
        "python_version_filter": python_version,
        "os_version_filter": os_version,
        "from_dt": from_dt,
        "to_dt": to_dt,
        "endpoint": INDEX_PATTERN,
        "error": None,
    }

    _set(cache_key, evidence, CACHE_TTL_SECS)
    logger.debug(
        "panda_job_stats: %s(%s) = %s  doc_count=%d  site=%r  status=%r",
        metric, field, value, doc_count, site, jobstatus,
    )
    return evidence


# ---------------------------------------------------------------------------
# OpenSearch exception → user-facing message
# ---------------------------------------------------------------------------


def _os_error_message(exc: BaseException) -> str:
    """Translate an OpenSearch exception into a concise user-facing message.

    Inspects the exception type and extracts HTTP status / reason where
    available so the user sees a specific, actionable error rather than the
    generic "check that ASKPANDA_OPENSEARCH is set" fallback.

    Args:
        exc: Exception raised by opensearch-py during query execution.

    Returns:
        A single-sentence string suitable for display in the Bamboo UI.
    """
    cls_name = type(exc).__name__

    # AuthorizationException (HTTP 403) ― permissions problem, not connectivity.
    if cls_name == "AuthorizationException":
        return (
            "Permission denied on atlas_panda_job_stats-* (HTTP 403). "
            "Ask your OpenSearch admin to grant indices:data/read/search "
            "for the pilot-monitor-agent account on this index pattern."
        )

    # NotFoundError (HTTP 404) ― index does not exist yet.
    if cls_name == "NotFoundError":
        return (
            "Index atlas_panda_job_stats-* not found (HTTP 404). "
            "The index may not have been created yet — check with Sasha."
        )

    # ConnectionError / ConnectionTimeout ― network or VPN issue.
    if cls_name in ("ConnectionError", "ConnectionTimeout"):
        return (
            "Could not reach the OpenSearch cluster. "
            "Check that you are on the CERN VPN and that "
            "ASKPANDA_OPENSEARCH_HOST is correct."
        )

    # Generic TransportError ― include the HTTP status and reason if present.
    if cls_name == "TransportError" or "TransportError" in cls_name:
        status = getattr(exc, "status_code", None)
        error = getattr(exc, "error", None)
        if status and error:
            return (
                f"OpenSearch returned HTTP {status}: {error}. "
                "Check cluster health and index availability."
            )
        if status:
            return (
                f"OpenSearch returned HTTP {status}. "
                "Check cluster health and index availability."
            )

    # Fallback for anything else (e.g. SSLError, unexpected exceptions).
    return (
        "Could not retrieve job stats data. "
        "Check that ASKPANDA_OPENSEARCH is set and the OpenSearch "
        "cluster is reachable."
    )

# ---------------------------------------------------------------------------
# Structured error constructor
# ---------------------------------------------------------------------------


def _error_evidence(
    metric: str,
    field: str,
    site: str | None,
    jobstatus: str | None,
    jeditaskid: int | None,
    from_dt: str | None,
    to_dt: str | None,
    detail: str,
    user_message: str | None = None,
    group_by: str | None = None,
    top_n: int | None = None,
    order: str | None = None,
    python_version: str | None = None,
    os_version: str | None = None,
) -> dict[str, Any]:
    """Return a structured evidence dict representing a fetch failure.

    The internal *detail* is logged at DEBUG level but never exposed to
    the user.

    Args:
        metric: Requested aggregation metric.
        field: Requested aggregation field.
        site: Requested site filter, or ``None``.
        jobstatus: Requested job status filter, or ``None``.
        jeditaskid: Requested JEDI task ID filter, or ``None``.
        from_dt: Requested time-range lower bound, or ``None``.
        to_dt: Requested time-range upper bound, or ``None``.
        detail: Internal error message for logging (never shown to user).
        user_message: Optional user-facing error string.  When omitted,
            falls back to a generic connectivity message.
        group_by: Requested group-by field, or ``None``.
        top_n: Requested bucket count, or ``None``.
        order: Requested sort order (``"asc"`` / ``"desc"``), or ``None``.
        python_version: Requested Python version prefix filter, or ``None``.
        os_version: Requested OS version prefix filter, or ``None``.

    Returns:
        Evidence dict with ``error`` populated and ``value`` set to ``None``.
    """
    from askpanda_atlas.job_stats_schema import INDEX_PATTERN  # deferred

    logger.debug("panda_job_stats error: %s", detail)
    if user_message is None:
        user_message = (
            "Could not retrieve job stats data. "
            "Check that ASKPANDA_OPENSEARCH is set and the OpenSearch "
            "cluster is reachable."
        )
    return {
        "metric": metric,
        "field": field,
        "group_by": group_by,
        "top_n": top_n,
        "order": order,
        "buckets": None,
        "value": None,
        "doc_count": 0,
        "site_filter": site,
        "jobstatus_filter": jobstatus,
        "jeditaskid_filter": jeditaskid,
        "python_version_filter": python_version,
        "os_version_filter": os_version,
        "from_dt": from_dt,
        "to_dt": to_dt,
        "endpoint": INDEX_PATTERN,
        "error": user_message,
    }


def _cannot_answer_evidence(question: str) -> dict[str, Any]:
    """Return a structured evidence dict when the LLM cannot extract params.

    Args:
        question: The original user question.

    Returns:
        Evidence dict with a user-safe error message.
    """
    from askpanda_atlas.job_stats_schema import INDEX_PATTERN  # deferred

    return {
        "metric": None,
        "field": None,
        "group_by": None,
        "top_n": None,
        "order": None,
        "buckets": None,
        "value": None,
        "doc_count": 0,
        "site_filter": None,
        "jobstatus_filter": None,
        "jeditaskid_filter": None,
        "python_version_filter": None,
        "os_version_filter": None,
        "from_dt": None,
        "to_dt": None,
        "endpoint": INDEX_PATTERN,
        "error": (
            "I wasn't able to translate that question into a job stats query. "
            "Try asking about a specific field — for example: "
            "'What is the average stage-in time at BNL?', "
            "'What is the average RSS memory usage at CERN?', "
            "'What is the CPU efficiency at IN2P3 today?', "
            "or 'How many jobs ran at BNL today?'."
        ),
        "question": question,
    }


# ---------------------------------------------------------------------------
# MCP tool definition
# ---------------------------------------------------------------------------


def get_definition() -> dict[str, Any]:
    """Return the MCP tool definition for ``panda_job_stats``.

    Returns:
        Tool definition dict compatible with MCP discovery.
    """
    return {
        "name": "panda_job_stats",
        "description": (
            "Answer natural-language questions about PanDA job performance "
            "and statistics by querying the OpenSearch atlas_panda_job_stats-* "
            "index.  Use this tool when the user asks about job timing, memory "
            "usage, CPU efficiency, HS06 accounting, I/O throughput, pilot or "
            "execution errors, task/campaign context, carbon footprint, or "
            "software environment (Python version, OS version) — "
            "for example: "
            "'What is the average stage-in time at BNL?', "
            "'What is the average RSS memory usage at CERN today?', "
            "'What is the CPU efficiency at IN2P3 today?', "
            "'What is the total HS06-seconds at TRIUMF today?', "
            "'What is the average write throughput at CERN?', "
            "'Which sites are still running python 3.7?', "
            "or 'How many jobs ran at BNL today?'. "
            "Requires ASKPANDA_OPENSEARCH to be set."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "Natural-language question about PanDA job performance, "
                        "e.g. 'What is the average stage-in time at BNL?'"
                    ),
                },
                "site": {
                    "type": "string",
                    "description": (
                        "Optional computing site filter, e.g. 'BNL', 'CERN'. "
                        "Overrides any site extracted from the question."
                    ),
                },
                "python_version": {
                    "type": "string",
                    "description": (
                        "Optional Python version prefix filter, e.g. '3.7'. "
                        "Matches jobs whose python_version starts with this "
                        "prefix. Overrides any version extracted from the "
                        "question."
                    ),
                },
                "os_version": {
                    "type": "string",
                    "description": (
                        "Optional OS version prefix filter, e.g. '7', '9.7'. "
                        "Matches jobs whose os_version starts with this "
                        "prefix. Overrides any version extracted from the "
                        "question."
                    ),
                },
                "from_dt": {
                    "type": "string",
                    "description": (
                        "ISO-8601 lower bound on @timestamp "
                        "(e.g. '2026-06-01T00:00:00').  "
                        "Overrides any time range extracted from the question."
                    ),
                },
                "to_dt": {
                    "type": "string",
                    "description": (
                        "ISO-8601 upper bound on @timestamp.  "
                        "Overrides any time range extracted from the question."
                    ),
                },
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    }


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------


class PandaJobStatsTool:
    """MCP tool that answers NL job-stats questions via OpenSearch aggregations.

    Translates the user's natural-language question into OpenSearch
    aggregation parameters using a single LLM call, then executes a
    single-value metric aggregation against ``atlas_panda_job_stats-*``
    and returns a compact evidence dict for Bamboo's central synthesiser.
    """

    def __init__(self) -> None:
        """Initialise with the cached tool definition."""
        self._def: dict[str, Any] = get_definition()

    def get_definition(self) -> dict[str, Any]:
        """Return the MCP tool definition.

        Returns:
            Tool definition dictionary.
        """
        return self._def

    async def call(self, arguments: dict[str, Any]) -> list[Any]:
        """Extract query parameters from the question and return evidence.

        Steps:

        1. Validate and normalise the ``question`` argument.
        2. Call the LLM (async) to extract structured query parameters.
        3. Apply any argument-level overrides (``site``, ``python_version``,
           ``os_version``, ``from_dt``, ``to_dt``).
        4. Fall back to the default time window when no time range is
           specified.
        5. Execute the OpenSearch aggregation synchronously via
           ``asyncio.to_thread``.
        6. Return the evidence dict as MCP content.

        ``bamboo.tools.base`` is imported here (deferred) so the rest of
        this module remains importable when bamboo core is not installed.

        Args:
            arguments: Dict with required ``"question"`` (str) and optional
                ``"site"`` (str), ``"python_version"`` (str),
                ``"os_version"`` (str), ``"from_dt"`` (str), ``"to_dt"`` (str).

        Returns:
            One-element MCP content list containing the JSON-serialised
            evidence dict, or an error payload on failure.
        """
        from bamboo.tools.base import text_content  # deferred

        question: str = (arguments.get("question") or "").strip()
        if not question:
            return text_content(json.dumps({
                "evidence": {"error": "question argument is required."},
            }))
        if len(question) > 2000:
            return text_content(json.dumps({
                "evidence": {"error": "Question is too long (max 2000 characters)."},
            }))

        # Argument-level overrides (take precedence over LLM extraction).
        arg_site: str | None = (arguments.get("site") or "").strip() or None
        arg_python_version: str | None = (arguments.get("python_version") or "").strip() or None
        arg_os_version: str | None = (arguments.get("os_version") or "").strip() or None
        arg_from_dt: str | None = (arguments.get("from_dt") or "").strip() or None
        arg_to_dt: str | None = (arguments.get("to_dt") or "").strip() or None

        logger.debug("panda_job_stats: question=%r", question)

        # ── Stage 1: LLM parameter extraction ─────────────────────────────
        try:
            raw_reply = await _call_llm_for_params(question)
        except Exception as exc:  # noqa: BLE001
            logger.exception("panda_job_stats: LLM call failed")
            from askpanda_atlas.job_stats_schema import DEFAULT_FIELD, DEFAULT_METRIC  # deferred
            ev = _error_evidence(
                DEFAULT_METRIC, DEFAULT_FIELD,
                arg_site, None, None, arg_from_dt, arg_to_dt,
                detail=f"LLM call failed: {exc}",
                python_version=arg_python_version,
                os_version=arg_os_version,
            )
            return text_content(json.dumps({"evidence": ev}))

        params = parse_llm_params(raw_reply)
        if params is None:
            logger.debug("panda_job_stats: LLM could not extract params")
            ev = _cannot_answer_evidence(question)
            return text_content(json.dumps({"evidence": ev}))

        # Apply argument-level overrides.
        if arg_site:
            params["site"] = arg_site
        if arg_python_version:
            params["python_version"] = arg_python_version
        if arg_os_version:
            params["os_version"] = arg_os_version
        if arg_from_dt:
            params["from_dt"] = arg_from_dt
        if arg_to_dt:
            params["to_dt"] = arg_to_dt

        # Fall back to the default time window when no range at all.
        if not params["from_dt"] and not params["to_dt"]:
            default_from, default_to = _default_window()
            params["from_dt"] = default_from
            params["to_dt"] = default_to

        logger.debug(
            "panda_job_stats: params=%s", json.dumps(params, default=str)
        )

        # ── Stage 2: OpenSearch aggregation ───────────────────────────────
        try:
            evidence = await asyncio.to_thread(
                fetch_job_stats,
                params["metric"],
                params["field"],
                params["site"],
                params["jobstatus"],
                params["jeditaskid"],
                params["from_dt"],
                params["to_dt"],
                params.get("group_by"),
                params.get("top_n", 5),
                params.get("order", "desc"),
                params.get("python_version"),
                params.get("os_version"),
            )
            return text_content(json.dumps({"evidence": evidence}))
        except Exception as exc:  # noqa: BLE001
            logger.exception("panda_job_stats: OpenSearch query failed")
            ev = _error_evidence(
                params["metric"], params["field"],
                params["site"], params["jobstatus"], params["jeditaskid"],
                params["from_dt"], params["to_dt"],
                detail=repr(exc),
                user_message=_os_error_message(exc),
                group_by=params.get("group_by"),
                top_n=params.get("top_n"),
                order=params.get("order"),
                python_version=params.get("python_version"),
                os_version=params.get("os_version"),
            )
            return text_content(json.dumps({"evidence": ev}))


panda_job_stats_tool = PandaJobStatsTool()

__all__ = [
    "PandaJobStatsTool",
    "_cannot_answer_evidence",
    "_default_window",
    "_error_evidence",
    "_os_error_message",
    "_int_or_none",
    "_is_cannot_answer",
    "_str_or_none",
    "_strip_llm_fences",
    "fetch_job_stats",
    "get_definition",
    "panda_job_stats_tool",
    "parse_llm_params",
]
