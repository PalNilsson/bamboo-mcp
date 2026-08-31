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

"""The REST facade's question must keep routing to log analysis.

``bamboo.entrypoints.rest`` asks ``bamboo_answer`` a fixed sentence — "Analyze
job N and explain the failure" — rather than calling ``panda_log_analysis``
directly, so that the button and a chat user exercise one code path.  The cost
of that choice is a dependency on a routing heuristic: if ``_LOG_PATTERN`` or
the rule ordering in ``_build_deterministic_plan`` changes, the button silently
starts answering a different question.

These tests are the fence around that dependency.  They fail loudly on a
routing change instead of letting the monitor quietly degrade, and they pin the
exact string the facade sends rather than a paraphrase of it, because a
paraphrase would pass while the real request broke.
"""

from __future__ import annotations

import re

import pytest

from bamboo.entrypoints import rest
from bamboo.tools.bamboo_answer import (
    _build_deterministic_plan,
    _extract_job_id,
    _is_log_analysis_request,
)
from bamboo.tools.planner import PlanRoute

_JOB: int = 7272161793

#: The exact sentence the facade sends, reconstructed the same way.
_CANONICAL: str = f"Analyze job {_JOB} and explain the failure"


def _facade_question(job_id: int) -> str:
    """Return the question the REST facade builds for a job.

    Kept as a helper so the assertion below compares against the same shape the
    facade produces rather than a hand-written copy.

    Args:
        job_id: PanDA job id.

    Returns:
        The question string.
    """
    return f"Analyze job {job_id} and explain the failure"


class TestFacadeQuestionIsUnchanged:
    """The string itself is part of the contract."""

    def test_facade_source_uses_the_canonical_phrasing(self) -> None:
        """The sentence in rest.py matches what these tests pin.

        Read from the module source, so editing the f-string in ``_execute``
        without updating this file fails here rather than in production.
        """
        import inspect

        source = inspect.getsource(rest._execute)
        match = re.search(r'f"([^"]*\{record\.job_id\}[^"]*)"', source)

        assert match is not None, "could not find the question f-string in rest._execute"
        rendered = match.group(1).replace("{record.job_id}", str(_JOB))
        assert rendered == _CANONICAL

    def test_helper_matches_the_canonical_string(self) -> None:
        """The helper used below is the same sentence."""
        assert _facade_question(_JOB) == _CANONICAL


class TestCanonicalQuestionRouting:
    """The sentence must reach panda_log_analysis, deterministically."""

    def test_job_id_is_extracted(self) -> None:
        """Routing needs the job id out of the sentence first."""
        assert _extract_job_id(_CANONICAL) == _JOB

    def test_recognised_as_a_log_analysis_request(self) -> None:
        """Rule 1's predicate accepts the phrasing."""
        assert _is_log_analysis_request(_CANONICAL) is True

    def test_deterministic_plan_selects_log_analysis(self) -> None:
        """The fast path routes it without consulting an LLM."""
        plan = _build_deterministic_plan(_CANONICAL, task_id=None, job_id=_JOB)

        assert plan is not None
        assert plan.route == PlanRoute.FAST_PATH
        assert [call.tool for call in plan.tool_calls] == ["panda_log_analysis"]

    def test_not_diverted_to_core_dump_analysis(self) -> None:
        """Rule 1c sits above rule 1 and must not claim this sentence.

        Core-dump analysis holds a single global slot and serialises, so a
        button on every failed job page must never land there by accident.
        """
        plan = _build_deterministic_plan(_CANONICAL, task_id=None, job_id=_JOB)

        assert plan is not None
        assert all("core_dump" not in call.tool for call in plan.tool_calls)

    def test_not_diverted_to_job_status(self) -> None:
        """Rule 2 is the fallback for a bare job id and must not win here."""
        plan = _build_deterministic_plan(_CANONICAL, task_id=None, job_id=_JOB)

        assert plan is not None
        assert "panda_job_status" not in [call.tool for call in plan.tool_calls]

    def test_job_id_is_passed_to_the_tool(self) -> None:
        """The plan carries the job, not just the tool name."""
        plan = _build_deterministic_plan(_CANONICAL, task_id=None, job_id=_JOB)

        assert plan is not None
        job_argument = plan.tool_calls[0].arguments.get("job_id")
        assert job_argument is not None
        assert int(job_argument) == _JOB

    @pytest.mark.parametrize("job_id", [1234, 7272161793, 999999999999])
    def test_routing_holds_across_job_id_widths(self, job_id: int) -> None:
        """``_LOG_PATTERN`` bounds the digit count; the real range must fit."""
        question = _facade_question(job_id)
        plan = _build_deterministic_plan(question, task_id=None, job_id=job_id)

        assert plan is not None
        assert [call.tool for call in plan.tool_calls] == ["panda_log_analysis"]


class TestFastPathIsPinnedOn:
    """The facade must not inherit an interface's debugging switch."""

    def test_rest_passes_bypass_fast_path_false(self) -> None:
        """``BAMBOO_FAST_PATH`` is read only by Streamlit and Textual.

        Those interfaces translate it into ``bypass_fast_path``; core never
        consults the variable. A REST caller that left the argument unset would
        still get ``False`` today, but pinning it explicitly means the
        deterministic route cannot be turned off from the environment.
        """
        import inspect

        source = inspect.getsource(rest._execute)

        assert '"bypass_fast_path": False' in source
