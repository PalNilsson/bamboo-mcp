"""Blue/green ChromaDB collection name resolver.

The ``bamboo-mcp-services`` document-monitor agent stores vectors in a pair of
physical ChromaDB collections (``<name>__a`` and ``<name>__b``) and rotates
between them atomically on each update cycle.  Which slot is currently live is
recorded in a JSON sidecar file:

.. code-block:: json

    {
        "atlas_docs": "atlas_docs__a",
        "epic_docs":  "epic_docs__b"
    }

The sidecar is written by the agent using ``os.replace`` (POSIX atomic rename)
so it is never partially written from a reader's perspective.

This module provides two **read-only** helpers:

:func:`resolve_collection`
    Translates a logical collection name (e.g. ``"atlas_docs"``) to the
    currently live physical slot name (e.g. ``"atlas_docs__a"``).  This is the
    single point of blue/green slot resolution and must always be called before
    opening a ChromaDB collection.

:func:`resolve_collection_for_topic`
    Translates an abstract *topic* string (e.g. ``"rucio"``) to a logical
    collection name, then delegates to :func:`resolve_collection` for slot
    resolution.  Topic-to-logical-name mapping is driven by the
    ``BAMBOO_CHROMA_COLLECTION_MAP`` environment variable (a JSON object), with
    ``BAMBOO_CHROMA_COLLECTION`` as the ultimate scalar fallback.

Both helpers are intentionally standalone — they do not import from
``bamboo_mcp_services`` — so Bamboo MCP remains independent of the services
package.

Multi-collection configuration
-------------------------------
Set ``BAMBOO_CHROMA_COLLECTION_MAP`` to a JSON object mapping topic keys to
logical collection names::

    export BAMBOO_CHROMA_COLLECTION_MAP='{
        "panda":           "panda_docs",
        "atlas":           "atlas_docs",
        "bamboo":          "bamboo_docs",
        "bamboo_mcp":      "bamboo_mcp_docs",
        "bamboo_services": "bamboo_services_docs",
        "rucio":           "rucio_docs",
        "root":            "root_docs",
        "epic":            "epic_docs",
        "cgsim":           "cgsim_docs"
    }'

The ``bamboo`` key is a legacy alias for single-collection deployments that
store all Bamboo documentation together.  Deployments that have split the
collection into ``bamboo_mcp_docs`` (bamboo-mcp repository) and
``bamboo_services_docs`` (bamboo-mcp-services repository) should add
``bamboo_mcp`` and ``bamboo_services`` entries — this prevents install and
setup documentation for one component polluting answers about the other.

Adding a new collection requires only updating the JSON string — no code
changes and no new environment variables.

Deployments that do not set ``BAMBOO_CHROMA_COLLECTION_MAP`` fall back to the
scalar ``BAMBOO_CHROMA_COLLECTION`` for all topics (pre-multi-collection
behaviour is fully preserved).

Fallback behaviour
------------------
If the sidecar is absent, unreadable, or has no entry for the requested logical
name, :func:`resolve_collection` returns the logical name unchanged.  This
means:

- Deployments that have not yet upgraded to the blue/green agent continue to
  work without any configuration change.
- A transient sidecar read error (e.g. during a write) degrades gracefully
  rather than taking the RAG tool offline.

Live re-resolution
------------------
:func:`resolve_collection` re-reads the sidecar on **every call**.  This is
intentional: the sidecar is a tiny JSON file (< 1 KB) and reading it is
cheap compared with a ChromaDB query.  Re-reading on every call means that
when the document-monitor agent swaps the active slot the MCP tools pick up
the new collection on the next query without requiring a server restart.

Callers that cache a ``chromadb.Collection`` handle should compare the
resolved physical name against the name of the cached handle and invalidate
the cache when they differ.  :class:`~bamboo.tools.doc_rag.PandaDocSearchTool`
does this via its ``_resolved_physical`` attribute.
"""
from __future__ import annotations

import json
import logging
import os

LOG = logging.getLogger(__name__)

#: Name of the routing sidecar file, relative to the ChromaDB directory.
ROUTING_SIDECAR = "collection_routing.json"

#: Environment variable that holds the topic → logical-collection-name map as
#: a JSON object.  Example value (single line for shell export)::
#:
#:     '{"panda":"panda_docs","atlas":"atlas_docs",
#:       "bamboo":"bamboo_docs",
#:       "bamboo_mcp":"bamboo_mcp_docs","bamboo_services":"bamboo_services_docs",
#:       "rucio":"rucio_docs","root":"root_docs","epic":"epic_docs",
#:       "cgsim":"cgsim_docs"}'
#:
#: If the variable is absent or unparseable, :func:`resolve_collection_for_topic`
#: falls back to :data:`COLLECTION_DEFAULT_ENV`.
COLLECTION_MAP_ENV = "BAMBOO_CHROMA_COLLECTION_MAP"

#: Scalar fallback env var used when ``BAMBOO_CHROMA_COLLECTION_MAP`` is not
#: set or does not contain an entry for the requested topic.
COLLECTION_DEFAULT_ENV = "BAMBOO_CHROMA_COLLECTION"

#: Topics already warned about, so the scalar-override notice is logged once
#: per topic rather than on every question.
_WARNED_TOPICS: set[str] = set()

#: Built-in default logical collection names keyed by topic string.  These are
#: used when neither ``BAMBOO_CHROMA_COLLECTION_MAP`` nor
#: ``BAMBOO_CHROMA_COLLECTION`` is set.
#:
#: The ``"bamboo"`` key is a legacy alias retained for backward compatibility
#: with single-collection deployments that have not yet split ``bamboo_docs``
#: into ``bamboo_mcp_docs`` and ``bamboo_services_docs``.  New deployments
#: should use ``"bamboo_mcp"`` and ``"bamboo_services"`` explicitly.
_BUILTIN_DEFAULTS: dict[str, str] = {
    "panda": "panda_docs",
    "atlas": "atlas_docs",
    "bamboo": "bamboo_docs",       # legacy alias — kept for backward compat
    "bamboo_mcp": "bamboo_mcp_docs",
    "bamboo_services": "bamboo_services_docs",
    "rucio": "rucio_docs",
    "root": "root_docs",
    "epic": "epic_docs",
    "cgsim": "cgsim_docs",
}


def resolve_collection(chroma_path: str, logical_name: str) -> str:
    """Resolve a logical ChromaDB collection name to its current live physical name.

    Reads ``<chroma_path>/collection_routing.json`` written by the
    ``bamboo-mcp-services`` document-monitor agent.  If the sidecar is absent,
    unreadable, or contains no entry for *logical_name*, the logical name is
    returned unchanged (graceful fallback for deployments that do not yet use
    blue/green rotation).

    This function re-reads the sidecar file on every call so that a live swap
    by the document-monitor agent is picked up immediately without a server
    restart.

    Args:
        chroma_path: Path to the ChromaDB persistent directory (the same value
            as ``BAMBOO_CHROMA_PATH``).
        logical_name: The logical collection name as configured via
            ``BAMBOO_CHROMA_COLLECTION`` (e.g. ``"atlas_docs"``).

    Returns:
        The physical ChromaDB collection name to open (e.g. ``"atlas_docs__a"``),
        or *logical_name* unchanged if no routing record is found.
    """
    sidecar = os.path.join(chroma_path, ROUTING_SIDECAR)
    try:
        with open(sidecar, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        # Normal for deployments not yet on the blue/green agent.
        return logical_name
    except Exception:  # pylint: disable=broad-exception-caught
        # Corrupt sidecar, permission error, etc. — degrade gracefully.
        LOG.warning(
            "_chroma_routing: failed to read sidecar '%s'; using logical name '%s'",
            sidecar, logical_name,
        )
        return logical_name

    physical = data.get(logical_name)
    if not physical:
        # No entry for this logical name — new corpus not yet swapped, or a
        # deployment with a custom collection name that predates the sidecar.
        return logical_name

    if physical != logical_name:
        LOG.debug(
            "_chroma_routing: '%s' → '%s' (via sidecar)",
            logical_name, physical,
        )
    return physical


def resolve_collection_for_topic(chroma_path: str, topic: str) -> str:
    """Resolve a topic string to the current live physical ChromaDB slot name.

    Performs a two-step lookup:

    1. **Topic → logical name** via ``BAMBOO_CHROMA_COLLECTION_MAP`` (a JSON
       object mapping topic keys to logical collection names).  Falls back to
       ``BAMBOO_CHROMA_COLLECTION`` when the map is absent or has no entry for
       *topic*, and finally to the built-in default for the topic (e.g.
       ``"panda_docs"`` for topic ``"panda"``).

    2. **Logical name → physical slot** via :func:`resolve_collection`, which
       reads the blue/green sidecar written by ``bamboo-mcp-services``.

    This function is the recommended entry point for all RAG tools.  Callers
    should pass the topic string supplied in the tool's ``arguments`` dict
    (e.g. ``"atlas"``, ``"rucio"``).  An unknown or empty topic falls back
    silently to ``"panda_docs"`` (or whatever ``BAMBOO_CHROMA_COLLECTION`` is
    set to).

    Adding support for a new collection requires only updating
    ``BAMBOO_CHROMA_COLLECTION_MAP`` — no code changes are needed.

    Args:
        chroma_path: Path to the ChromaDB persistent directory (same as
            ``BAMBOO_CHROMA_PATH``).
        topic: Abstract topic key, e.g. ``"panda"``, ``"atlas"``, ``"rucio"``,
            ``"root"``, ``"bamboo"``, ``"epic"``, ``"cgsim"``.  Case-insensitive.

    Returns:
        Physical ChromaDB collection name to open (e.g. ``"panda_docs__b"``).
    """
    topic_key = (topic or "").strip().lower()

    # Step 1a — try BAMBOO_CHROMA_COLLECTION_MAP
    map_raw = os.getenv(COLLECTION_MAP_ENV, "").strip()
    logical_name: str = ""
    if map_raw:
        try:
            collection_map: dict[str, str] = json.loads(map_raw)
            logical_name = str(collection_map.get(topic_key) or "")
        except Exception:  # pylint: disable=broad-exception-caught
            LOG.warning(
                "_chroma_routing: failed to parse %s; falling back to %s",
                COLLECTION_MAP_ENV, COLLECTION_DEFAULT_ENV,
            )

    # Step 1b — fall back to scalar BAMBOO_CHROMA_COLLECTION
    if not logical_name:
        logical_name = os.getenv(COLLECTION_DEFAULT_ENV, "").strip()

        # The scalar is checked BEFORE the per-topic built-in below, so setting
        # it collapses every topic onto one collection. That is deliberate for
        # single-collection deployments, and silent everywhere else: on
        # aipanda033 the store held panda_docs (389), atlas_docs (356),
        # root_docs (8064) and rucio_docs (236), while every query — including
        # ones the planner correctly tagged topic="atlas" — was answered from
        # bamboo_docs (83 chunks of this project's own service READMEs). The
        # answers were thin rather than wrong, so nothing looked broken.
        #
        # Routing is left as it is; changing the precedence would silently
        # re-point existing single-collection deployments. But a topic that has
        # a built-in default and is being overridden by the scalar is worth
        # saying out loud, once per topic.
        if logical_name and topic_key in _BUILTIN_DEFAULTS:
            if topic_key not in _WARNED_TOPICS:
                _WARNED_TOPICS.add(topic_key)
                LOG.warning(
                    "_chroma_routing: topic '%s' would resolve to '%s', but %s='%s' "
                    "overrides it, so this and every other topic is answered from "
                    "'%s'. Set %s to route topics to their own collections, or "
                    "unset %s to use the built-in per-topic defaults.",
                    topic_key, _BUILTIN_DEFAULTS[topic_key],
                    COLLECTION_DEFAULT_ENV, logical_name, logical_name,
                    COLLECTION_MAP_ENV, COLLECTION_DEFAULT_ENV,
                )

    # Step 1c — built-in per-topic default
    if not logical_name:
        logical_name = _BUILTIN_DEFAULTS.get(topic_key, "panda_docs")

    LOG.debug(
        "_chroma_routing: topic '%s' → logical '%s'", topic_key, logical_name
    )

    # Step 2 — blue/green slot resolution
    return resolve_collection(chroma_path, logical_name)
