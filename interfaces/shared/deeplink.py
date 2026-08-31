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

"""Turn URL query parameters into an opening chat question.

Why this exists
---------------
The cheapest possible version of the PanDA monitor's "Analyze failure" button
is a link: the monitor renders an anchor to the Bamboo UI carrying the job id,
and the chat opens with the question already asked.  No REST call, no proxy
view, no polling.  It is worth having even once the REST facade is live,
because it is also the "continue in Bamboo" link from the answer panel and the
fallback when the API is down.

Why the logic lives here rather than in ``chat.py``
---------------------------------------------------
``interfaces/streamlit/chat.py`` imports Streamlit, which is an optional
dependency and is not installed in the test environment.  Keeping the parsing
here means it can be tested directly, and the Textual interface can reuse it
without a second implementation.

Why ``q`` is off by default
---------------------------
Accepting ``job_id`` means a link can ask one fixed, harmless question about a
job.  Accepting arbitrary ``q`` text means anybody who can get a person to
click a link can put words in their mouth: the question appears in their
history as though they typed it, and it spends tokens against the deployment's
budget.  For a pilot behind CERN SSO that is a small risk, but it is not zero
and it is not needed by the monitor button, so free text requires
``BAMBOO_DEEPLINK_ALLOW_QUESTION=1``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

#: The question a ``job_id`` parameter expands to.  Deliberately identical to
#: the sentence ``bamboo.entrypoints.rest`` sends, so the link and the button
#: produce the same routing and the same answer.
FAILURE_QUESTION_TEMPLATE: str = "Analyze job {job_id} and explain the failure"

#: Longest free-text question accepted from a URL.
MAX_QUESTION_CHARS: int = 500

#: Job ids accepted from a URL.  The bound matches ``_LOG_PATTERN`` in
#: ``bamboo_answer``: an id outside it would not route to log analysis, so
#: refusing it here gives a clean no-op instead of a confusing answer.
_JOB_ID_RE = re.compile(r"^[0-9]{4,12}$")

#: Control characters are collapsed rather than kept, so a crafted link cannot
#: inject newlines that would read as separate instructions in a prompt.
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")


def free_text_enabled() -> bool:
    """Report whether a ``q`` parameter is honoured.

    Returns:
        ``True`` when ``BAMBOO_DEEPLINK_ALLOW_QUESTION`` is truthy.
    """
    raw = os.getenv("BAMBOO_DEEPLINK_ALLOW_QUESTION", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _first(value: Any) -> str:
    """Return a single string from a query-parameter value.

    Streamlit hands back a bare string, ``urllib.parse.parse_qs`` a list; both
    shapes reach this module depending on the caller.

    Args:
        value: The raw parameter value.

    Returns:
        The first value as a string, or an empty string.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and value:
        return str(value[0])
    return ""


def sanitise_question(text: str) -> str | None:
    """Clean a free-text question from a URL.

    Args:
        text: The raw parameter value.

    Returns:
        The cleaned question, or ``None`` when it is empty or too long.
    """
    cleaned = _CONTROL_RE.sub(" ", text).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    if not cleaned or len(cleaned) > MAX_QUESTION_CHARS:
        return None
    return cleaned


def question_for_job(job_id: str | int) -> str | None:
    """Return the canonical failure question for a job id.

    Args:
        job_id: Job id as text or int.

    Returns:
        The question, or ``None`` when the id is not a plausible PanDA job id.
    """
    text = str(job_id).strip()
    if not _JOB_ID_RE.match(text):
        return None
    return FAILURE_QUESTION_TEMPLATE.format(job_id=text)


def question_from_params(params: Mapping[str, Any]) -> str | None:
    """Build the opening question from URL query parameters.

    ``job_id`` takes precedence over ``q``: the monitor sends the former, and
    preferring it means a link carrying both cannot use the job id as cover for
    free text that would otherwise be refused.

    Args:
        params: Query parameters, values as strings or one-element sequences.

    Returns:
        The question to ask on open, or ``None`` when nothing applies.
    """
    job_id = _first(params.get("job_id"))
    if job_id:
        return question_for_job(job_id)

    raw_question = _first(params.get("q"))
    if raw_question and free_text_enabled():
        return sanitise_question(raw_question)

    return None


__all__ = [
    "FAILURE_QUESTION_TEMPLATE",
    "MAX_QUESTION_CHARS",
    "free_text_enabled",
    "question_for_job",
    "question_from_params",
    "sanitise_question",
]
