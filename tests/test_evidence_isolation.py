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

"""Cross-session isolation of conversational state.

The evidence store and the prompt-log document coordinates are written during
one turn and read during a later one.  On the shared HTTP server a single
process serves every client, so a process-global store means one client's turn
can be answered — or rated — against another client's data.

The two gates that made this concrete before :mod:`bamboo.session_scope`
existed:

* ``get_last_core_dump_offer`` decides whether a bare "yes" means "analyse the
  core dump", and recovers the job id from the stored evidence rather than
  from the user's message.  Read process-globally, user B's "yes" could start
  a gdb run against user A's job.
* ``get_last_traceback_evidence`` gates the rule 1b pilot-source route, so a
  question could be answered against another user's traceback.

These tests are the regression fence.  They are written to fail if the store
is ever restored to a plain module-level dict, which is the mutation used to
verify they are load-bearing rather than vacuous.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from bamboo import session_scope as ss
from bamboo.llm import prompt_log as pl
from bamboo.tools import bamboo_executor as ex_mod
from bamboo.tools.bamboo_executor import (
    bamboo_last_evidence_tool,
    get_last_core_dump_offer,
    get_last_traceback_evidence,
)

_JOB_A: int = 7272161793
_JOB_B: int = 6799893074


@pytest.fixture(autouse=True)
def _clean_state() -> Any:
    """Clear session buckets and the default store around each test.

    Yields:
        None.
    """
    ss.reset_all()
    ex_mod._last_evidence_store.clear()
    yield
    ss.reset_all()
    ex_mod._last_evidence_store.clear()


def _store_core_dump_offer(job_id: int) -> None:
    """Store a ``panda_log_analysis`` result carrying a core-dump offer.

    Args:
        job_id: Job the offer refers to.
    """
    ex_mod._last_evidence_store["panda_log_analysis"] = {
        "evidence": {
            "job_id": job_id,
            "failure_type": "looping_job",
            "core_dump_offer_md": "\n\nA core dump is present. Analyse it?",
        },
        "text": "Job was killed as a looping job.",
    }
    ex_mod._last_evidence_store["last_tool"] = "panda_log_analysis"


def _store_traceback(job_id: int) -> None:
    """Store a ``panda_log_analysis`` result carrying a pilot traceback.

    Args:
        job_id: Job the traceback belongs to.
    """
    ex_mod._last_evidence_store["panda_log_analysis"] = {
        "evidence": {
            "job_id": job_id,
            "traceback_available": True,
            "deepest_pilot_frame": {
                "path": "pilot/control/payload.py",
                "line": 214,
                "function": "run_payload",
            },
            "log_excerpt": "Traceback (most recent call last): ...",
        },
    }
    ex_mod._last_evidence_store["last_tool"] = "panda_log_analysis"


class TestCoreDumpOfferGate:
    """The bare-affirmative gate must not cross sessions."""

    def test_offer_is_visible_inside_its_own_session(self) -> None:
        """The session that produced the offer can act on it."""
        with ss.session_scope("mcp:a"):
            _store_core_dump_offer(_JOB_A)
            offer = get_last_core_dump_offer()

        assert offer is not None
        assert offer["job_id"] == _JOB_A

    def test_offer_does_not_leak_to_another_session(self) -> None:
        """A second client's "yes" must not inherit the first client's offer."""
        with ss.session_scope("mcp:a"):
            _store_core_dump_offer(_JOB_A)

        with ss.session_scope("mcp:b"):
            assert get_last_core_dump_offer() is None

    def test_each_session_keeps_its_own_job_id(self) -> None:
        """Two concurrent offers stay attached to their own jobs."""
        with ss.session_scope("mcp:a"):
            _store_core_dump_offer(_JOB_A)
        with ss.session_scope("mcp:b"):
            _store_core_dump_offer(_JOB_B)

        with ss.session_scope("mcp:a"):
            offer_a = get_last_core_dump_offer()
        with ss.session_scope("mcp:b"):
            offer_b = get_last_core_dump_offer()

        assert offer_a is not None and offer_a["job_id"] == _JOB_A
        assert offer_b is not None and offer_b["job_id"] == _JOB_B

    def test_offer_is_dropped_when_the_session_ends(self) -> None:
        """Clearing a session releases its evidence."""
        with ss.session_scope("mcp:a"):
            _store_core_dump_offer(_JOB_A)
        ss.clear_session("mcp:a")
        with ss.session_scope("mcp:a"):
            assert get_last_core_dump_offer() is None


class TestTracebackGate:
    """The rule 1b pilot-source gate must not cross sessions."""

    def test_traceback_is_visible_inside_its_own_session(self) -> None:
        """The producing session can route to pilot-source analysis."""
        with ss.session_scope("mcp:a"):
            _store_traceback(_JOB_A)
            evidence = get_last_traceback_evidence()

        assert evidence is not None
        assert evidence["job_id"] == _JOB_A

    def test_traceback_does_not_leak_to_another_session(self) -> None:
        """Another client's question must not see this traceback."""
        with ss.session_scope("mcp:a"):
            _store_traceback(_JOB_A)

        with ss.session_scope("mcp:b"):
            assert get_last_traceback_evidence() is None


class TestLastEvidenceTool:
    """``bamboo_last_evidence`` reports the caller's own session only."""

    def test_tool_reports_nothing_for_a_fresh_session(self) -> None:
        """A new client sees an empty store, not the previous client's."""
        with ss.session_scope("mcp:a"):
            _store_core_dump_offer(_JOB_A)

        with ss.session_scope("mcp:b"):
            result = asyncio.run(bamboo_last_evidence_tool.call({}))

        payload = json.loads(result[0]["text"])
        assert "error" in payload or not payload.get("evidence")

    def test_tool_reports_the_caller_session_evidence(self) -> None:
        """The producing session gets its own evidence back."""
        with ss.session_scope("mcp:a"):
            _store_core_dump_offer(_JOB_A)
            result = asyncio.run(bamboo_last_evidence_tool.call({}))

        payload = json.loads(result[0]["text"])
        assert payload["tool"] == "panda_log_analysis"
        assert payload["evidence"]["evidence"]["job_id"] == _JOB_A


class TestUnscopedBehaviourUnchanged:
    """stdio must behave exactly as it did before scoping existed."""

    def test_unscoped_write_and_read_round_trip(self) -> None:
        """With no scope bound the store behaves like the old global dict."""
        _store_core_dump_offer(_JOB_A)
        offer = get_last_core_dump_offer()

        assert offer is not None
        assert offer["job_id"] == _JOB_A

    def test_unscoped_state_is_the_default_bucket(self) -> None:
        """Unscoped writes are addressable as the default session."""
        _store_core_dump_offer(_JOB_A)
        bucket = ss.bucket(ss.EVIDENCE_BUCKET, session_id=ss.DEFAULT_SESSION_ID)

        assert bucket["last_tool"] == "panda_log_analysis"

    def test_unscoped_and_scoped_do_not_mix(self) -> None:
        """A scoped client cannot see stdio state and vice versa."""
        _store_core_dump_offer(_JOB_A)

        with ss.session_scope("mcp:a"):
            assert get_last_core_dump_offer() is None
            _store_core_dump_offer(_JOB_B)

        offer = get_last_core_dump_offer()
        assert offer is not None
        assert offer["job_id"] == _JOB_A


class TestPromptLogCoordinates:
    """Ratings must reach the document the caller's own turn produced."""

    def test_coordinates_do_not_leak_between_sessions(self) -> None:
        """Session B must not be able to rate session A's document."""
        with ss.session_scope("mcp:a"):
            pl.record_last_doc("idx-2026.08.31", "doc-a", session_id="mcp:a")

        with ss.session_scope("mcp:b"):
            assert pl.get_last_doc_id() is None

        with ss.session_scope("mcp:a"):
            assert pl.get_last_doc_id() == ("idx-2026.08.31", "doc-a")

    def test_unscoped_falls_back_to_the_process_store(self) -> None:
        """stdio keeps the single-slot process-wide behaviour."""
        pl.record_last_doc("idx-2026.08.31", "doc-stdio")
        assert pl.get_last_doc_id() == ("idx-2026.08.31", "doc-stdio")

    def test_write_document_result_is_attributed(self) -> None:
        """The scheduling wrapper files coordinates under the right session."""

        def _fake_write(doc: dict[str, Any]) -> tuple[str, str]:
            return ("idx-2026.08.31", "doc-42")

        async def _run() -> None:
            with ss.session_scope("mcp:a"):
                await pl._index_and_record({"turn_number": 1}, "mcp:a")

        original = pl._write_document
        pl._write_document = _fake_write  # type: ignore[assignment]
        try:
            asyncio.run(_run())
        finally:
            pl._write_document = original  # type: ignore[assignment]

        with ss.session_scope("mcp:a"):
            assert pl.get_last_doc_id() == ("idx-2026.08.31", "doc-42")
        with ss.session_scope("mcp:b"):
            assert pl.get_last_doc_id() is None

    def test_document_records_the_bound_session_id(self) -> None:
        """Indexed documents carry the transport session, not the process id."""
        captured: list[dict[str, Any]] = []

        def _fake_write(doc: dict[str, Any]) -> None:
            captured.append(doc)

        async def _run() -> None:
            with ss.session_scope("mcp:zz"):
                await pl.log_prompt(
                    system_prompt="s",
                    user_prompt="u",
                    response="r",
                    tools_used=[],
                    provider="anthropic",
                    model="claude-sonnet-4-6",
                    max_tokens=2048,
                )
                await asyncio.sleep(0.05)

        original = pl._write_document
        pl._write_document = _fake_write  # type: ignore[assignment]
        try:
            import os
            from unittest.mock import patch

            with patch.dict(os.environ, {"BAMBOO_OPENSEARCH_PROMPTLOG": "testpw"}):
                asyncio.run(_run())
        finally:
            pl._write_document = original  # type: ignore[assignment]

        assert len(captured) == 1
        assert captured[0]["session_id"] == "mcp:zz"
