# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Authors
# - Paul Nilsson, paul.nilsson@cern.ch, 2026

"""Disk-backed store for asynchronous failure analyses.

Why on disk
-----------
A log analysis takes tens of seconds, which is longer than a browser request
should wait behind an nginx proxy, so the REST facade starts the work and hands
back an identifier to poll.  That identifier has to resolve to something, and a
process-global dict is the wrong something for three reasons: a restart would
strand every client mid-poll with no way to say what happened; multiple uvicorn
workers would round-robin a poll to a worker that never saw the request; and
the single-flight claim has to be visible to whoever else is asking for the
same job at the same moment.  A directory of small JSON files answers all
three, and the failure mode when it goes wrong is a re-run rather than a lie.

Layout::

    $BAMBOO_REST_STORE_ROOT/
        analyses/<analysis_id>.json     one record per request
        cache/<cache_key>.json          pointer to a completed analysis
        inflight/<cache_key>.json       claim held while one is running

Three separate concerns
-----------------------
*Records* are the state machine: queued, running, then complete or failed.
Written by one owner, read by many pollers.

*Cache* answers "has this exact question already been answered".  Job logs are
immutable once uploaded, so a completed analysis stays valid for a long time —
but only for the same prompt and model, which is why both are folded into the
cache key.  The exception is an analysis that found no log at all: a job that
has just failed may still be uploading, so that result gets a short TTL
(:data:`NO_LOG_CACHE_TTL_S`) rather than being cached for a week.

*Inflight claims* stop twenty people clicking the same button from starting
twenty gdb-free-but-still-expensive analyses.  The claim is taken with
``O_CREAT | O_EXCL``, which is atomic on POSIX filesystems, so exactly one
caller wins and the rest are handed the winner's id.

Crash reconciliation
--------------------
Both records and claims carry the owning process id.  A record found in a
non-terminal state whose owner is gone is reported as failed rather than left
pending forever, and a claim whose owner is gone is taken over rather than
blocking the job until someone clears the directory by hand.  This is the same
approach the core-dump analyzer takes with its workspace manifests.

Environment variables
---------------------
``BAMBOO_REST_STORE_ROOT``
    Root directory (default: ``/tmp/bamboo/rest-analysis``).  ``/tmp`` does not
    survive a reboot; set a persistent path where losing cached answers on
    reboot matters.

``BAMBOO_ANALYSIS_CACHE_TTL_S``
    Lifetime of a cached answer (default: 604800, one week).

``BAMBOO_ANALYSIS_RETENTION_S``
    Age at which :func:`sweep` deletes records and pointers (default:
    1209600, two weeks).

``BAMBOO_ANALYSIS_PROMPT_VERSION``
    Folded into the cache key.  Bump it to invalidate every cached answer
    without deleting anything.

``BAMBOO_ANALYSIS_MAX_RECORD_CHARS``
    Cap on the serialised evidence held in one record (default: 1000000).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Root directory used when the environment is unset.  Mirrors the
#: ``/tmp/bamboo/...`` convention of the core-dump analyzer.
DEFAULT_STORE_ROOT: str = "/tmp/bamboo/rest-analysis"

#: Default lifetime of a cached answer: one week.
DEFAULT_CACHE_TTL_S: float = 604800.0

#: Cache lifetime for an analysis that found no log.  Short, because a job that
#: has just failed may still be uploading and the next caller should get a real
#: answer rather than this one.
NO_LOG_CACHE_TTL_S: float = 300.0

#: Default age at which :func:`sweep` removes records: two weeks.
DEFAULT_RETENTION_S: float = 1209600.0

#: Cap on the serialised evidence stored in one record.
DEFAULT_MAX_RECORD_CHARS: int = 1_000_000

#: Prompt version folded into the cache key when the environment is unset.
#: Bump this when the synthesis prompt changes in a way that should invalidate
#: previously cached answers.
DEFAULT_PROMPT_VERSION: str = "1"

#: Reads attempted before an unreadable claim file is declared abandoned, and
#: the pause between them.  Insurance for the hard-link fallback path, where a
#: claim can briefly exist with no body yet.
_CLAIM_READ_ATTEMPTS: int = 5
_CLAIM_READ_DELAY_S: float = 0.05


class AnalysisState:
    """The states a record moves through.

    Attributes:
        QUEUED: Accepted, not yet started.
        RUNNING: Work in progress.
        COMPLETE: Finished with an answer.
        FAILED: Finished without one.
    """

    QUEUED: str = "queued"
    RUNNING: str = "running"
    COMPLETE: str = "complete"
    FAILED: str = "failed"

    #: States from which no further transition happens.
    TERMINAL: frozenset[str] = frozenset({"complete", "failed"})


@dataclass
class AnalysisRecord:
    """One analysis request and whatever is known about it so far.

    Attributes:
        analysis_id: Opaque identifier handed to the client.
        job_id: PanDA job being analysed.
        mode: Analysis flavour, e.g. ``"failure"``.
        state: One of :class:`AnalysisState`.
        cache_key: Key this analysis publishes to on success.
        pid: Process that owns the work, for crash reconciliation.
        created_utc: ISO-8601 creation timestamp.
        updated_utc: ISO-8601 timestamp of the last write.
        elapsed_s: Seconds from creation to the terminal state.
        answer_markdown: The answer, when complete.
        evidence: Structured evidence backing the answer.
        promptlog: ``{"index": ..., "doc_id": ...}`` so a rating reaches the
            right document without relying on process-global state.
        error: Failure reason, when failed.
        no_log: Whether the analysis ran but found no log to read.
        requested_by: Identity forwarded by the caller, for attribution only.
    """

    analysis_id: str
    job_id: int
    mode: str
    state: str
    cache_key: str = ""
    pid: int = 0
    created_utc: str = ""
    updated_utc: str = ""
    elapsed_s: float = 0.0
    answer_markdown: str | None = None
    evidence: dict[str, Any] | None = None
    promptlog: dict[str, str] | None = None
    error: str | None = None
    no_log: bool = False
    requested_by: str = ""

    #: Set when the record was served from cache rather than freshly computed.
    #: Not persisted — it describes this response, not the analysis.
    cached: bool = field(default=False, compare=False)

    def is_terminal(self) -> bool:
        """Report whether the analysis has finished.

        Returns:
            ``True`` when the state is complete or failed.
        """
        return self.state in AnalysisState.TERMINAL

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation.

        Returns:
            Dict of every field, including ``cached``.
        """
        return asdict(self)


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        Timestamp with a trailing ``Z``.
    """
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _parse_utc(text: str) -> float:
    """Parse an ISO-8601 timestamp into a POSIX timestamp.

    Args:
        text: Timestamp produced by :func:`_utc_now`.

    Returns:
        Seconds since the epoch, or ``0.0`` when unparsable.
    """
    if not text:
        return 0.0
    try:
        cleaned = text[:-1] if text.endswith("Z") else text
        return datetime.fromisoformat(cleaned).replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return 0.0


def _env_float(name: str, default: float) -> float:
    """Read a positive float from the environment.

    Args:
        name: Environment variable name.
        default: Value returned when unset, unparsable, or not positive.

    Returns:
        The parsed value, or *default*.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_int(name: str, default: int) -> int:
    """Read a positive integer from the environment.

    Args:
        name: Environment variable name.
        default: Value returned when unset, unparsable, or not positive.

    Returns:
        The parsed value, or *default*.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def pid_alive(pid: int | None) -> bool:
    """Report whether a process id still exists.

    ``os.kill(pid, 0)`` raising :class:`PermissionError` means the process is
    there but owned by somebody else, which for this purpose is alive.

    Args:
        pid: Process id, or ``None``.

    Returns:
        ``True`` when the process exists.
    """
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (TypeError, ValueError, OSError):
        return False
    return True


def store_root() -> Path:
    """Return the root directory of the store.

    Returns:
        ``$BAMBOO_REST_STORE_ROOT``, or :data:`DEFAULT_STORE_ROOT`.
    """
    return Path(os.getenv("BAMBOO_REST_STORE_ROOT") or DEFAULT_STORE_ROOT)


def records_dir() -> Path:
    """Return the directory holding analysis records.

    Returns:
        ``<root>/analyses``.
    """
    return store_root() / "analyses"


def cache_dir() -> Path:
    """Return the directory holding cache pointers.

    Returns:
        ``<root>/cache``.
    """
    return store_root() / "cache"


def inflight_dir() -> Path:
    """Return the directory holding single-flight claims.

    Returns:
        ``<root>/inflight``.
    """
    return store_root() / "inflight"


def cache_ttl_s() -> float:
    """Return the configured lifetime of a cached answer.

    Returns:
        ``$BAMBOO_ANALYSIS_CACHE_TTL_S``, or :data:`DEFAULT_CACHE_TTL_S`.
    """
    return _env_float("BAMBOO_ANALYSIS_CACHE_TTL_S", DEFAULT_CACHE_TTL_S)


def retention_s() -> float:
    """Return the age at which :func:`sweep` deletes records.

    Returns:
        ``$BAMBOO_ANALYSIS_RETENTION_S``, or :data:`DEFAULT_RETENTION_S`.
    """
    return _env_float("BAMBOO_ANALYSIS_RETENTION_S", DEFAULT_RETENTION_S)


def prompt_version() -> str:
    """Return the prompt version folded into cache keys.

    Returns:
        ``$BAMBOO_ANALYSIS_PROMPT_VERSION``, or
        :data:`DEFAULT_PROMPT_VERSION`.
    """
    return os.getenv("BAMBOO_ANALYSIS_PROMPT_VERSION", "").strip() or DEFAULT_PROMPT_VERSION


def cache_key_for(job_id: int, mode: str, model: str, version: str | None = None) -> str:
    """Return the cache key for one question.

    The model and prompt version are part of the key, not metadata beside it:
    a cached answer produced by a different model or a since-revised prompt is
    a different answer, and serving it would make a model change invisible.

    Args:
        job_id: PanDA job id.
        mode: Analysis flavour.
        model: Model string the answer was or will be produced with.
        version: Prompt version, or ``None`` for :func:`prompt_version`.

    Returns:
        Hex digest usable as a filename.
    """
    raw = f"{int(job_id)}|{mode}|{model}|{version or prompt_version()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON to *path* atomically.

    The temporary file is created in the destination directory so that
    ``os.replace`` is a rename within one filesystem, which is what makes it
    atomic — a concurrent reader sees either the old file or the new one, never
    a half-written mixture.

    Args:
        path: Destination path.
        payload: JSON-serialisable content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path, quiet: bool = False) -> dict[str, Any] | None:
    """Read JSON from *path*, tolerating absence and corruption.

    Args:
        path: File to read.
        quiet: Log at debug rather than warning. Used by the claim-read retry,
            where a momentarily unreadable file is expected rather than a
            problem worth reporting three times.

    Returns:
        The parsed object, or ``None`` when missing or unreadable.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        if quiet:
            logger.debug("analysis_store: cannot read %s yet: %s", path, exc)
        else:
            logger.warning("analysis_store: cannot read %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def _record_path(analysis_id: str) -> Path:
    """Return the manifest path for an analysis id.

    Args:
        analysis_id: Identifier to locate.

    Returns:
        Path under ``<root>/analyses``.

    Raises:
        ValueError: If *analysis_id* is not a bare identifier.  Ids reach this
            module straight from a URL path, so anything that could climb out
            of the directory is refused rather than sanitised.
    """
    if not analysis_id or not all(ch.isalnum() or ch in "-_" for ch in analysis_id):
        raise ValueError(f"invalid analysis id: {analysis_id!r}")
    return records_dir() / f"{analysis_id}.json"


def _truncate_evidence(evidence: dict[str, Any] | None) -> dict[str, Any] | None:
    """Cap the serialised size of an evidence dict.

    Args:
        evidence: Evidence to store, or ``None``.

    Returns:
        The evidence unchanged, or a marker dict when it exceeds the cap.
    """
    if evidence is None:
        return None
    cap = _env_int("BAMBOO_ANALYSIS_MAX_RECORD_CHARS", DEFAULT_MAX_RECORD_CHARS)
    try:
        size = len(json.dumps(evidence, default=str))
    except (TypeError, ValueError):
        return {"truncated": True, "reason": "evidence is not serialisable"}
    if size <= cap:
        return evidence
    return {
        "truncated": True,
        "reason": f"evidence of {size} characters exceeds the {cap}-character cap",
    }


def create(job_id: int, mode: str, cache_key: str, requested_by: str = "") -> AnalysisRecord:
    """Create and persist a queued record.

    Args:
        job_id: PanDA job id.
        mode: Analysis flavour.
        cache_key: Key this analysis will publish to on success.
        requested_by: Identity forwarded by the caller, for attribution only.

    Returns:
        The persisted record.
    """
    now = _utc_now()
    record = AnalysisRecord(
        analysis_id=uuid.uuid4().hex,
        job_id=int(job_id),
        mode=mode,
        state=AnalysisState.QUEUED,
        cache_key=cache_key,
        pid=os.getpid(),
        created_utc=now,
        updated_utc=now,
        requested_by=requested_by,
    )
    _write_json(_record_path(record.analysis_id), record.as_dict())
    return record


def save(record: AnalysisRecord) -> AnalysisRecord:
    """Persist a record, stamping its update time.

    Args:
        record: Record to write.

    Returns:
        The record as written.
    """
    record.updated_utc = _utc_now()
    record.evidence = _truncate_evidence(record.evidence)
    _write_json(_record_path(record.analysis_id), record.as_dict())
    return record


def load(analysis_id: str) -> AnalysisRecord | None:
    """Load a record, reconciling one abandoned by a dead process.

    A record left in a non-terminal state by a process that no longer exists
    is reported as failed.  Leaving it pending would have the client poll until
    it gives up, with nothing anywhere saying why.

    Args:
        analysis_id: Identifier to load.

    Returns:
        The record, or ``None`` when unknown.

    Raises:
        ValueError: If *analysis_id* is not a bare identifier.
    """
    data = _read_json(_record_path(analysis_id))
    if data is None:
        return None

    data.pop("cached", None)
    known = {f.name for f in fields(AnalysisRecord)}
    unknown = set(data) - known
    if unknown:
        # A record written by a newer version must stay readable: the store
        # lives across restarts and upgrades, and refusing the whole record
        # over one unrecognised key would strand a client mid-poll.
        logger.debug(
            "analysis_store: ignoring unknown field(s) %s in record %s",
            sorted(unknown),
            analysis_id,
        )
        data = {key: value for key, value in data.items() if key in known}

    try:
        record = AnalysisRecord(**data)
    except TypeError as exc:
        logger.warning("analysis_store: record %s has bad fields: %s", analysis_id, exc)
        return None

    if record.is_terminal() or pid_alive(record.pid):
        return record

    record.state = AnalysisState.FAILED
    record.error = "The server restarted while this analysis was running."
    record.elapsed_s = max(0.0, _parse_utc(record.updated_utc) - _parse_utc(record.created_utc))
    return save(record)


def mark_running(record: AnalysisRecord) -> AnalysisRecord:
    """Move a record into the running state.

    Args:
        record: Record to update.

    Returns:
        The persisted record.
    """
    record.state = AnalysisState.RUNNING
    record.pid = os.getpid()
    return save(record)


def mark_complete(
    record: AnalysisRecord,
    answer_markdown: str,
    evidence: dict[str, Any] | None = None,
    promptlog: dict[str, str] | None = None,
    no_log: bool = False,
) -> AnalysisRecord:
    """Finish a record with an answer and publish it to the cache.

    Args:
        record: Record to update.
        answer_markdown: The answer text.
        evidence: Structured evidence backing the answer.
        promptlog: Coordinates of the prompt-log document, so a later rating
            reaches the right one.
        no_log: Whether the analysis found no log to read.

    Returns:
        The persisted record.
    """
    record.state = AnalysisState.COMPLETE
    record.answer_markdown = answer_markdown
    record.evidence = evidence
    record.promptlog = promptlog
    record.no_log = no_log
    record.elapsed_s = max(0.0, time.time() - _parse_utc(record.created_utc))
    saved = save(record)
    publish_cache(saved)
    return saved


def mark_failed(record: AnalysisRecord, error: str) -> AnalysisRecord:
    """Finish a record without an answer.

    Failures are deliberately not cached: the next caller should get a real
    attempt, since most failures here are transient (a log not yet uploaded, a
    BigPanDA timeout) rather than properties of the job.

    Args:
        record: Record to update.
        error: Human-readable failure reason.

    Returns:
        The persisted record.
    """
    record.state = AnalysisState.FAILED
    record.error = error
    record.elapsed_s = max(0.0, time.time() - _parse_utc(record.created_utc))
    return save(record)


def publish_cache(record: AnalysisRecord) -> None:
    """Point a cache key at a completed analysis.

    Args:
        record: A completed record carrying a cache key.
    """
    if record.state != AnalysisState.COMPLETE or not record.cache_key:
        return
    ttl = NO_LOG_CACHE_TTL_S if record.no_log else cache_ttl_s()
    _write_json(
        cache_dir() / f"{record.cache_key}.json",
        {
            "analysis_id": record.analysis_id,
            "completed_utc": _utc_now(),
            "expires_at": time.time() + ttl,
            "job_id": record.job_id,
        },
    )


def lookup_cache(cache_key: str) -> AnalysisRecord | None:
    """Return a cached completed analysis for *cache_key*, if any.

    An expired pointer is deleted on the way past, so the directory does not
    accumulate entries nobody will ever read again.

    Args:
        cache_key: Key to look up.

    Returns:
        The cached record with ``cached`` set, or ``None``.
    """
    pointer_path = cache_dir() / f"{cache_key}.json"
    pointer = _read_json(pointer_path)
    if pointer is None:
        return None

    if float(pointer.get("expires_at", 0.0)) <= time.time():
        pointer_path.unlink(missing_ok=True)
        return None

    analysis_id = str(pointer.get("analysis_id", ""))
    if not analysis_id:
        return None

    try:
        record = load(analysis_id)
    except ValueError:
        return None

    if record is None or record.state != AnalysisState.COMPLETE:
        # The record was swept while the pointer survived.
        pointer_path.unlink(missing_ok=True)
        return None

    record.cached = True
    return record


def try_claim(cache_key: str, analysis_id: str) -> str | None:
    """Claim the right to run an analysis, or report who already has it.

    The claim is written to a temporary file first and published with
    ``os.link``, which on POSIX both fails when the destination exists and
    makes the file visible complete.  ``O_CREAT | O_EXCL`` alone is not enough:
    it creates a zero-length file and the body lands on the next line, so a
    caller arriving inside that window reads an empty file, concludes the claim
    is corrupt, takes it over, and there are two winners.  That is not
    theoretical — it is what CI caught.

    A claim whose owner process is gone is taken over: the alternative is a job
    that can never be analysed again until somebody clears the directory by
    hand.

    Args:
        cache_key: Key identifying the question.
        analysis_id: Identifier of the caller's own record.

    Returns:
        ``None`` when the claim was taken, or the analysis id of the holder.
    """
    inflight_dir().mkdir(parents=True, exist_ok=True)
    path = inflight_dir() / f"{cache_key}.json"
    payload = {
        "analysis_id": analysis_id,
        "pid": os.getpid(),
        "started_utc": _utc_now(),
    }

    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    try:
        try:
            os.link(tmp, path)
            return None
        except FileExistsError:
            return _take_over_or_report(path, payload)
        except OSError as exc:
            # Some filesystems refuse hard links.  Fall back to the exclusive
            # create, accepting the narrow window it reopens; the retry in
            # _take_over_or_report is what covers it.
            logger.debug("analysis_store: os.link unavailable (%s); using O_EXCL", exc)
            return _claim_without_link(path, payload)
    finally:
        tmp.unlink(missing_ok=True)


def _claim_without_link(path: Path, payload: dict[str, Any]) -> str | None:
    """Claim using an exclusive create, for filesystems without hard links.

    Args:
        path: The claim file.
        payload: The caller's claim.

    Returns:
        ``None`` when the claim was taken, or the holder's analysis id.
    """
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return _take_over_or_report(path, payload)

    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return None


def _take_over_or_report(path: Path, payload: dict[str, Any]) -> str | None:
    """Resolve a claim collision.

    An unreadable claim is retried before being declared abandoned.  An empty
    or truncated file is far more likely to be a claim being written this
    instant than one left by a crash, and stealing it would produce exactly the
    duplicate work the claim exists to prevent.

    Args:
        path: The existing claim file.
        payload: The caller's claim, written if the holder is gone.

    Returns:
        ``None`` when the claim was taken over, or the holder's analysis id.
    """
    holder = _read_claim_with_retry(path)

    if holder is None:
        logger.info("analysis_store: claim %s is unreadable; taking it over", path.name)
        return _publish_takeover(path, payload)

    if pid_alive(int(holder.get("pid", 0) or 0)):
        return str(holder.get("analysis_id", "")) or None

    logger.info(
        "analysis_store: taking over claim %s abandoned by pid %s",
        path.name,
        holder.get("pid"),
    )
    return _publish_takeover(path, payload)


def _read_claim_with_retry(path: Path) -> dict[str, Any] | None:
    """Read a claim file, allowing for one being written right now.

    Args:
        path: The claim file.

    Returns:
        The parsed claim, or ``None`` if it stays unreadable.
    """
    for attempt in range(_CLAIM_READ_ATTEMPTS):
        holder = _read_json(path, quiet=True)
        if holder is not None and holder.get("analysis_id"):
            return holder
        if not path.exists():
            # The holder released it while we looked; nothing to report.
            return None
        if attempt < _CLAIM_READ_ATTEMPTS - 1:
            time.sleep(_CLAIM_READ_DELAY_S)
    return None


def _publish_takeover(path: Path, payload: dict[str, Any]) -> str | None:
    """Replace an abandoned claim, confirming the result by reading it back.

    Two callers can decide to take over the same abandoned claim at the same
    moment, and ``os.replace`` would let both believe they succeeded.  Reading
    back settles it: whoever's id survives owns the claim.

    Args:
        path: The claim file.
        payload: The caller's claim.

    Returns:
        ``None`` when the caller owns the claim, or the winner's analysis id.
    """
    _write_json(path, payload)
    settled = _read_json(path) or {}
    winner = str(settled.get("analysis_id", ""))
    if winner and winner != payload["analysis_id"]:
        return winner
    return None


def release_claim(cache_key: str, analysis_id: str) -> None:
    """Release a claim, if it is still the caller's.

    Checking ownership matters: a claim taken over after a crash belongs to
    somebody else now, and the original owner returning from the dead must not
    delete it.

    Args:
        cache_key: Key identifying the question.
        analysis_id: Identifier of the caller's own record.
    """
    path = inflight_dir() / f"{cache_key}.json"
    holder = _read_json(path)
    if holder is None or str(holder.get("analysis_id", "")) == analysis_id:
        path.unlink(missing_ok=True)


def sweep(max_age_s: float | None = None) -> int:
    """Delete records and pointers older than the retention window.

    Args:
        max_age_s: Age in seconds, or ``None`` for :func:`retention_s`.

    Returns:
        Number of files deleted.
    """
    cutoff = time.time() - (max_age_s if max_age_s is not None else retention_s())
    deleted = 0

    for directory in (records_dir(), cache_dir(), inflight_dir()):
        if not directory.is_dir():
            continue
        for path in directory.glob("*.json"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
                    deleted += 1
            except OSError as exc:  # pragma: no cover - racing sweeper
                logger.debug("analysis_store: cannot sweep %s: %s", path, exc)
    return deleted


__all__ = [
    "DEFAULT_CACHE_TTL_S",
    "DEFAULT_MAX_RECORD_CHARS",
    "DEFAULT_PROMPT_VERSION",
    "DEFAULT_RETENTION_S",
    "DEFAULT_STORE_ROOT",
    "NO_LOG_CACHE_TTL_S",
    "AnalysisRecord",
    "AnalysisState",
    "cache_dir",
    "cache_key_for",
    "cache_ttl_s",
    "create",
    "inflight_dir",
    "load",
    "lookup_cache",
    "mark_complete",
    "mark_failed",
    "mark_running",
    "pid_alive",
    "prompt_version",
    "publish_cache",
    "records_dir",
    "release_claim",
    "retention_s",
    "save",
    "store_root",
    "sweep",
    "try_claim",
]
