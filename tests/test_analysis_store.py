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

"""Tests for :mod:`bamboo.analysis_store`.

The store exists to survive things a dict would not: a restart mid-analysis, a
second worker process polling a record it did not create, and twenty people
clicking the same button at once.  These tests exercise those, plus the cache
semantics that decide whether a click costs an LLM call or nothing.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from bamboo import analysis_store as store

_JOB: int = 7272161793
_MODEL: str = "claude-sonnet-4-6"


@pytest.fixture(autouse=True)
def _store_root(tmp_path: Path) -> Any:
    """Point the store at a temporary directory.

    Args:
        tmp_path: pytest-provided temporary directory.

    Yields:
        The root in use.
    """
    with patch.dict(os.environ, {"BAMBOO_REST_STORE_ROOT": str(tmp_path)}):
        yield tmp_path


def _new(job_id: int = _JOB, mode: str = "failure") -> store.AnalysisRecord:
    """Create a queued record with a matching cache key.

    Args:
        job_id: PanDA job id.
        mode: Analysis flavour.

    Returns:
        The persisted record.
    """
    key = store.cache_key_for(job_id, mode, _MODEL)
    return store.create(job_id, mode, key)


def _claim_from_subprocess(root: str, cache_key: str, analysis_id: str, out: str) -> None:
    """Take a claim from a separate process and write the outcome.

    Args:
        root: Store root.
        cache_key: Key to claim.
        analysis_id: Caller's analysis id.
        out: File to write the result into.
    """
    os.environ["BAMBOO_REST_STORE_ROOT"] = root
    from bamboo import analysis_store as st

    result = st.try_claim(cache_key, analysis_id)
    Path(out).write_text(json.dumps({"result": result}), encoding="utf-8")


class TestCacheKey:
    """What is and is not part of the cache identity."""

    def test_same_question_same_key(self) -> None:
        """Identical inputs produce an identical key."""
        assert store.cache_key_for(_JOB, "failure", _MODEL) == store.cache_key_for(
            _JOB, "failure", _MODEL
        )

    def test_job_changes_the_key(self) -> None:
        """A different job is a different question."""
        assert store.cache_key_for(_JOB, "failure", _MODEL) != store.cache_key_for(
            _JOB + 1, "failure", _MODEL
        )

    def test_model_changes_the_key(self) -> None:
        """A model switch must not serve the old model's answers."""
        assert store.cache_key_for(_JOB, "failure", "claude-haiku-4-5") != store.cache_key_for(
            _JOB, "failure", _MODEL
        )

    def test_prompt_version_changes_the_key(self) -> None:
        """Bumping the prompt version invalidates every cached answer."""
        first = store.cache_key_for(_JOB, "failure", _MODEL)
        with patch.dict(os.environ, {"BAMBOO_ANALYSIS_PROMPT_VERSION": "2"}):
            second = store.cache_key_for(_JOB, "failure", _MODEL)
        assert first != second

    def test_key_is_filename_safe(self) -> None:
        """The key goes straight into a path, so it must be inert."""
        key = store.cache_key_for(_JOB, "failure", "vendor/model:weird name")
        assert key.isalnum()


class TestLifecycle:
    """Records move queued to running to terminal, and persist."""

    def test_create_persists_a_queued_record(self, _store_root: Path) -> None:
        """A new record is on disk and queued."""
        record = _new()

        assert record.state == store.AnalysisState.QUEUED
        assert (_store_root / "analyses" / f"{record.analysis_id}.json").exists()

    def test_load_round_trips(self) -> None:
        """A record reads back with its fields intact."""
        record = _new()
        loaded = store.load(record.analysis_id)

        assert loaded is not None
        assert loaded.analysis_id == record.analysis_id
        assert loaded.job_id == _JOB

    def test_unknown_id_is_none(self) -> None:
        """A missing record is absent, not an error."""
        assert store.load("0" * 32) is None

    def test_running_then_complete(self) -> None:
        """The happy path reaches a terminal state with an answer."""
        record = store.mark_running(_new())
        assert record.state == store.AnalysisState.RUNNING

        done = store.mark_complete(
            record,
            answer_markdown="The payload ran out of memory.",
            evidence={"pilot_error": 1212},
            promptlog={"index": "idx", "doc_id": "abc"},
        )

        assert done.is_terminal()
        assert done.promptlog == {"index": "idx", "doc_id": "abc"}
        assert store.load(done.analysis_id) is not None

    def test_failure_records_the_reason(self) -> None:
        """A failed analysis says why."""
        failed = store.mark_failed(_new(), "BigPanDA timed out.")

        assert failed.state == store.AnalysisState.FAILED
        assert "timed out" in (failed.error or "")

    def test_elapsed_is_recorded(self) -> None:
        """Terminal records carry a duration."""
        record = _new()
        time.sleep(0.01)
        done = store.mark_complete(record, answer_markdown="ok")
        assert done.elapsed_s > 0

    def test_corrupt_record_is_not_fatal(self, _store_root: Path) -> None:
        """A truncated manifest reads as absent rather than raising."""
        record = _new()
        path = _store_root / "analyses" / f"{record.analysis_id}.json"
        path.write_text('{"analysis_id": ', encoding="utf-8")

        assert store.load(record.analysis_id) is None

    def test_record_with_unknown_fields_still_loads(self, _store_root: Path) -> None:
        """A manifest from a newer version stays readable.

        The store outlives the process that wrote it, so a rolling upgrade will
        have old code reading new records. Refusing the whole record over one
        unrecognised key would strand a client mid-poll.
        """
        record = _new()
        path = _store_root / "analyses" / f"{record.analysis_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["field_from_the_future"] = True
        path.write_text(json.dumps(data), encoding="utf-8")

        loaded = store.load(record.analysis_id)

        assert loaded is not None
        assert loaded.job_id == _JOB

    def test_path_traversal_is_refused(self) -> None:
        """Ids arrive from a URL path and must not escape the directory."""
        with pytest.raises(ValueError):
            store.load("../../etc/passwd")

    def test_oversized_evidence_is_truncated(self) -> None:
        """A runaway evidence dict cannot fill the disk."""
        record = _new()
        with patch.dict(os.environ, {"BAMBOO_ANALYSIS_MAX_RECORD_CHARS": "500"}):
            done = store.mark_complete(
                record, answer_markdown="ok", evidence={"log": "x" * 5000}
            )

        assert done.evidence is not None
        assert done.evidence.get("truncated") is True


class TestCrashReconciliation:
    """A record abandoned by a dead process must not poll forever."""

    def test_dead_owner_marks_the_record_failed(self, _store_root: Path) -> None:
        """A running record with no live owner is reported as failed."""
        record = store.mark_running(_new())
        path = _store_root / "analyses" / f"{record.analysis_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["pid"] = 999_999_999
        path.write_text(json.dumps(data), encoding="utf-8")

        loaded = store.load(record.analysis_id)

        assert loaded is not None
        assert loaded.state == store.AnalysisState.FAILED
        assert "restarted" in (loaded.error or "")

    def test_reconciliation_is_persisted(self, _store_root: Path) -> None:
        """The verdict is written, so every later poll agrees."""
        record = store.mark_running(_new())
        path = _store_root / "analyses" / f"{record.analysis_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["pid"] = 999_999_999
        path.write_text(json.dumps(data), encoding="utf-8")

        store.load(record.analysis_id)
        again = json.loads(path.read_text(encoding="utf-8"))

        assert again["state"] == store.AnalysisState.FAILED

    def test_completed_record_is_untouched(self, _store_root: Path) -> None:
        """A finished analysis is not retroactively failed."""
        done = store.mark_complete(_new(), answer_markdown="answer")
        path = _store_root / "analyses" / f"{done.analysis_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["pid"] = 999_999_999
        path.write_text(json.dumps(data), encoding="utf-8")

        loaded = store.load(done.analysis_id)

        assert loaded is not None
        assert loaded.state == store.AnalysisState.COMPLETE
        assert loaded.answer_markdown == "answer"


class TestCache:
    """Whether a click costs an LLM call or nothing."""

    def test_miss_before_anything_completes(self) -> None:
        """An unasked question is not cached."""
        assert store.lookup_cache(store.cache_key_for(_JOB, "failure", _MODEL)) is None

    def test_hit_after_completion(self) -> None:
        """A completed analysis is served again without recomputing."""
        key = store.cache_key_for(_JOB, "failure", _MODEL)
        store.mark_complete(store.create(_JOB, "failure", key), answer_markdown="answer")

        hit = store.lookup_cache(key)

        assert hit is not None
        assert hit.answer_markdown == "answer"
        assert hit.cached is True

    def test_failure_is_not_cached(self) -> None:
        """A transient failure must not be served to the next caller."""
        key = store.cache_key_for(_JOB, "failure", _MODEL)
        store.mark_failed(store.create(_JOB, "failure", key), "timeout")

        assert store.lookup_cache(key) is None

    def test_expired_pointer_is_a_miss_and_is_removed(self, _store_root: Path) -> None:
        """A stale entry does not linger in the directory."""
        key = store.cache_key_for(_JOB, "failure", _MODEL)
        with patch.dict(os.environ, {"BAMBOO_ANALYSIS_CACHE_TTL_S": "0.05"}):
            store.mark_complete(store.create(_JOB, "failure", key), answer_markdown="answer")
            time.sleep(0.1)

            assert store.lookup_cache(key) is None

        assert not (_store_root / "cache" / f"{key}.json").exists()

    def test_no_log_result_gets_a_short_ttl(self, _store_root: Path) -> None:
        """A job still uploading its log must not be cached for a week."""
        key = store.cache_key_for(_JOB, "failure", _MODEL)
        store.mark_complete(
            store.create(_JOB, "failure", key), answer_markdown="No log yet.", no_log=True
        )

        pointer = json.loads((_store_root / "cache" / f"{key}.json").read_text(encoding="utf-8"))
        remaining = pointer["expires_at"] - time.time()

        assert remaining <= store.NO_LOG_CACHE_TTL_S
        assert remaining < store.cache_ttl_s()

    def test_swept_record_leaves_no_dangling_pointer(self, _store_root: Path) -> None:
        """A pointer to a deleted record is a miss and is cleaned up."""
        key = store.cache_key_for(_JOB, "failure", _MODEL)
        done = store.mark_complete(store.create(_JOB, "failure", key), answer_markdown="answer")
        (_store_root / "analyses" / f"{done.analysis_id}.json").unlink()

        assert store.lookup_cache(key) is None
        assert not (_store_root / "cache" / f"{key}.json").exists()


class TestSingleFlight:
    """Twenty clicks on one job must produce one analysis."""

    def test_first_caller_wins(self) -> None:
        """An uncontested claim succeeds."""
        key = store.cache_key_for(_JOB, "failure", _MODEL)
        assert store.try_claim(key, "aaa") is None

    def test_second_caller_is_told_the_winner(self) -> None:
        """The loser gets the winner's id rather than starting work."""
        key = store.cache_key_for(_JOB, "failure", _MODEL)
        store.try_claim(key, "aaa")

        assert store.try_claim(key, "bbb") == "aaa"

    def test_release_frees_the_key(self) -> None:
        """The next question on the same job can run once the first is done."""
        key = store.cache_key_for(_JOB, "failure", _MODEL)
        store.try_claim(key, "aaa")
        store.release_claim(key, "aaa")

        assert store.try_claim(key, "bbb") is None

    def test_release_by_a_non_owner_is_ignored(self) -> None:
        """A superseded owner must not free somebody else's claim."""
        key = store.cache_key_for(_JOB, "failure", _MODEL)
        store.try_claim(key, "aaa")
        store.release_claim(key, "bbb")

        assert store.try_claim(key, "ccc") == "aaa"

    def test_claim_from_a_dead_process_is_taken_over(self, _store_root: Path) -> None:
        """A crash must not lock a job out of analysis permanently."""
        key = store.cache_key_for(_JOB, "failure", _MODEL)
        store.inflight_dir().mkdir(parents=True, exist_ok=True)
        (store.inflight_dir() / f"{key}.json").write_text(
            json.dumps({"analysis_id": "ghost", "pid": 999_999_999, "started_utc": "x"}),
            encoding="utf-8",
        )

        assert store.try_claim(key, "live") is None

    def test_unreadable_claim_is_taken_over(self, _store_root: Path) -> None:
        """A corrupt claim file must not deadlock the key."""
        key = store.cache_key_for(_JOB, "failure", _MODEL)
        store.inflight_dir().mkdir(parents=True, exist_ok=True)
        (store.inflight_dir() / f"{key}.json").write_text("{{{", encoding="utf-8")

        assert store.try_claim(key, "live") is None

    def test_exactly_one_winner_across_processes(self, _store_root: Path, tmp_path: Path) -> None:
        """The O_EXCL claim holds against genuinely parallel callers.

        Multi-process rather than multi-threaded on purpose: threads in one
        interpreter would be serialised tightly enough by the GIL that the race
        never happens, and processes are the real case once more than one
        uvicorn worker is running.
        """
        import multiprocessing

        key = store.cache_key_for(_JOB, "failure", _MODEL)
        context = multiprocessing.get_context("spawn")
        outputs = [tmp_path / f"out{i}.json" for i in range(4)]
        procs = [
            context.Process(
                target=_claim_from_subprocess,
                args=(str(_store_root), key, f"id{i}", str(outputs[i])),
            )
            for i in range(4)
        ]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join(timeout=60)

        results = [json.loads(path.read_text(encoding="utf-8"))["result"] for path in outputs]
        winners = [r for r in results if r is None]

        assert all(proc.exitcode == 0 for proc in procs)
        assert len(winners) == 1
        assert all(r.startswith("id") for r in results if r is not None)


class TestSweep:
    """Retention, so an unattended deployment does not fill its disk."""

    def test_old_files_are_deleted(self, _store_root: Path) -> None:
        """Anything past the retention window goes."""
        record = _new()
        path = _store_root / "analyses" / f"{record.analysis_id}.json"
        old = time.time() - 100
        os.utime(path, (old, old))

        deleted = store.sweep(max_age_s=10)

        assert deleted == 1
        assert not path.exists()

    def test_recent_files_are_kept(self, _store_root: Path) -> None:
        """A live analysis is not swept out from under its poller."""
        record = _new()

        assert store.sweep(max_age_s=3600) == 0
        assert (_store_root / "analyses" / f"{record.analysis_id}.json").exists()

    def test_sweep_on_an_empty_store_is_zero(self) -> None:
        """Sweeping before anything exists is not an error."""
        assert store.sweep(max_age_s=1) == 0
