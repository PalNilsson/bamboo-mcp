"""Log-analysis scenarios shared by the equivalence walkthrough.

One scenario is everything needed to drive both log-analysis paths over the
same job: the BigPanDA ``job`` dict, the text each file downloads as, the
file-size listing, and what the two paths are expected to do with it.
``tests/test_log_equivalence.py`` runs ``fetch_and_analyse`` and the
``atlas.log.*`` primitive loop over each one and compares them.

Data, not pytest fixtures, and a plain module rather than ``conftest.py``, so
that the Track A granularity study can import the table without running under
pytest.  The import pattern is the one ``tests/test_plugin_mirror_parity.py``
already uses for ``tests/plugin_mirror_spec.py``::

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from log_scenarios import SCENARIOS  # noqa: E402

Why the expectations are in the table
-------------------------------------
:attr:`LogScenario.expect_fetched` and :attr:`LogScenario.expect_failure_type`
are not needed to compare the two paths against each other, and that is
exactly why they are here.  An equivalence assertion alone cannot notice the
two paths drifting together — a rule changed in ``_fetch_logs_payload`` and
transcribed faithfully into ``plan_fetch`` would keep them equal and change
what Bamboo tells an operator.  The expected fetch order and verdict pin the
behaviour itself, so such a change has to be made deliberately, in this file,
where it is reviewable.

Adding a scenario
-----------------
Append to :data:`SCENARIOS`.  ``sizes`` of ``None`` means the file listing
could not be fetched, which both paths treat fail-open; a file absent from
``texts`` downloads as ``None``, i.e. "exists in the listing but could not be
read".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "PAYLOAD_STDERR",
    "PAYLOAD_STDOUT",
    "PILOT_LOG",
    "SCENARIOS",
    "SETUP_LOG",
    "LogScenario",
    "by_name",
]

SETUP_LOG = "setup.stdout"
PAYLOAD_STDOUT = "payload.stdout"
PAYLOAD_STDERR = "payload.stderr"
PILOT_LOG = "pilotlog.txt"

# ---------------------------------------------------------------------------
# File contents
# ---------------------------------------------------------------------------

#: An asetup log that ``_setup_log_has_error`` does not fire on.  It mentions
#: Athena, as a real one does, which is what the substring table classifies a
#: job on when the setup log is the only evidence available.
_SETUP_CLEAN = (
    "Using AtlasSetup version 03-00-17\n"
    "Setting up Athena 21.0.15 for x86_64-centos7-gcc8-opt\n"
    "asetup completed in 4.2 seconds\n"
)

_SETUP_ERROR = (
    "Using AtlasSetup version 03-00-17\n"
    "!!!ERROR!!! No matched release is found for 21.0.15\n"
    "asetup failed\n"
)

#: A traceback whose deepest frame is in pilot code, so
#: ``_classify_from_exception`` reports ``pilot_exception`` rather than
#: letting the substring table blame the user's payload.
_PILOT_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/cvmfs/atlas.cern.ch/pilot/control/job.py", line 2841, in run_job\n'
    "    _stage_in(args, job)\n"
    '  File "/cvmfs/atlas.cern.ch/pilot/control/job.py", line 1102, in _stage_in\n'
    "    raise ValueError(message)\n"
    "ValueError: no replicas found for the input file\n"
)

_PAYLOAD_CHATTER = "AthenaMP INFO processing event {}\n"

#: Carries a version line of its own.  ``parse_pilot_version`` matches its
#: pattern anywhere, but only ``pilotlog.txt`` may answer for the pilot
#: version, so both paths must ignore this one and report the pilotid version
#: instead.
_PAYLOAD_SEGFAULT = (
    _PAYLOAD_CHATTER.format(998)
    + "echo: running under pilot version 9.9.9.9\n"
    + _PAYLOAD_CHATTER.format(999)
    + "Segmentation fault (core dumped)\n"
)

_PILOT_LOG_TIMEOUT = "\n".join([
    "2026-05-08 05:50:33 | INFO | pilot version 3.14.0.22",
    "2026-05-08 05:50:34 | INFO | stage-in starting",
    "2026-05-08 10:32:18 | INFO | handle_rucio_error | TimeoutException: "
    "Timeout reached, timeout=6842 seconds",
    "2026-05-08 10:32:18 | WARNING | failed to transfer_files: "
    "File transfer timed out during stage-in",
    "2026-05-08 10:32:19 | ERROR | pilot error set to 1151",
    "2026-05-08 10:32:20 | INFO | job ended",
])

#: The same log with the start-up banner cut off, as happens when the pilot
#: log is truncated.  The version then has to come from ``pilotid``.
_PILOT_LOG_NO_VERSION = "\n".join(_PILOT_LOG_TIMEOUT.splitlines()[1:])

_PILOT_LOG_NETWORK = "\n".join([
    "2026-05-08 05:50:33 | INFO | pilot version 3.14.0.22",
    "2026-05-08 05:51:02 | ERROR | curl: connection refused by rucio server",
    "2026-05-08 05:51:02 | ERROR | giving up",
])

#: Long enough that the stderr reservation is observable.  A short stderr
#: excerpts identically under a 2000-character budget and an 8000-character
#: one, so a scenario built on one cannot tell the two apart.
#: Wide for the same reason ``_PILOT_LOG_WIDE_LINE`` is: the context window
#: either side of a traceback is capped at a line count first, so narrow
#: lines make the character budget irrelevant.
_PAYLOAD_WIDE_LINE = (
    "AthenaMP WARNING AthenaEventLoopMgr.EventSelector "
    "failed to open input collection AOD.44556677._000123.pool.root.1, "
    "retrying with the next replica\n"
)
_BIG_PAYLOAD_STDERR = (
    _PAYLOAD_WIDE_LINE * 120
    + _PILOT_TRACEBACK
    + _PAYLOAD_WIDE_LINE * 40
)

#: Likewise for the pilot log, against the full budget rather than the
#: payload path's reduced one.  The preceding-context window is capped at a
#: line count, so the lines have to be long for the *character* budget to be
#: what binds — a narrow log excerpts identically under 6000 characters and
#: 8000.
_PILOT_LOG_WIDE_LINE = (
    "2026-05-08 05:50:34 | INFO | copying input file "
    + "root://eos.cern.ch//eos/atlas/data/mc23_13p6TeV/AOD.44556677._000123.pool.root.1 "
    + "to the work directory, attempt 1 of 3\n"
)
_BIG_PILOT_LOG = (
    "2026-05-08 05:50:33 | INFO | pilot version 3.14.0.22\n"
    + _PILOT_LOG_WIDE_LINE * 200
    + _PILOT_TRACEBACK
    + "2026-05-08 10:32:20 | INFO | job ended\n" * 20
)

#: A setup error log longer than the budget.  The marker is near the top, so
#: keeping the head of the file (what an erroring setup log gets) and keeping
#: the tail (what plain extraction would give) disagree on both the excerpt
#: and the verdict.
_BIG_SETUP_ERROR = _SETUP_ERROR + "asetup: probing cvmfs path\n" * 400

#: A ``pilotid`` in BigPanDA's pipe-delimited form.
_PILOTID = "https://aipanda.cern.ch/logs/pilot.tgz|PR|3.14.0.22"

#: A ``pilotid`` disagreeing with the version in the pilot log, so that a
#: scenario can tell "read from the log" apart from "fell back to metadata".
_PILOTID_STALE = "https://aipanda.cern.ch/logs/pilot.tgz|PR|3.9.1.4"


# ---------------------------------------------------------------------------
# The scenario record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LogScenario:
    """One job, as both log-analysis paths will see it.

    Attributes:
        name: Identifier, used as the pytest parameter id.
        pins: What this scenario exists to hold in place.  Read it before
            changing any field below.
        job: The ``job`` dict as BigPanDA's metadata endpoint returns it.
        texts: Mapping of filename to downloaded text.  A filename absent
            from the mapping downloads as ``None``: present in the listing,
            unreadable in practice.
        sizes: Mapping of filename to size in bytes, as the filebrowser
            listing reports it, or ``None`` when the listing itself could not
            be fetched — which both paths treat fail-open.
        expect_fetched: Filenames both paths must download, in order.
        expect_failure_type: The verdict both paths must reach.
    """

    name: str
    pins: str
    job: dict[str, Any]
    texts: dict[str, str | None]
    sizes: dict[str, int] | None
    expect_fetched: tuple[str, ...]
    expect_failure_type: str

    def listing(self) -> list[dict[str, Any]] | None:
        """Return the file listing as ``_fetch_file_listing`` would.

        Returns:
            Normalised listing records with all five keys
            ``_normalise_listing_entry`` guarantees, or ``None`` when
            :attr:`sizes` is ``None``.
        """
        if self.sizes is None:
            return None
        return [
            {
                "relative_path": name,
                "name": name,
                "dirname": "",
                "size_bytes": size,
                "modification": "2026-05-08 10:35",
            }
            for name, size in self.sizes.items()
        ]

    def text_for(self, filename: str) -> str | None:
        """Return the text one file downloads as.

        Args:
            filename: Log filename relative to the job directory.

        Returns:
            The configured text, or ``None`` when the file cannot be read.
        """
        return self.texts.get(filename)


def _job(
    code: Any = 1305,
    status: str = "failed",
    diag: str = "Payload failed",
    **overrides: Any,
) -> dict[str, Any]:
    """Build a BigPanDA ``job`` dict with the fields both paths read.

    Args:
        code: ``piloterrorcode``.  Deliberately typed ``Any`` so a scenario
            can supply the unparseable value BigPanDA occasionally returns.
        status: ``jobstatus``.
        diag: ``piloterrordiag``.
        **overrides: Any further job fields.

    Returns:
        The job dict.
    """
    job: dict[str, Any] = {
        "pandaid": 6799893074,
        "jobstatus": status,
        "piloterrorcode": code,
        "piloterrordiag": diag,
        "computingsite": "CERN-PROD",
        "cloud": "CERN",
        "jeditaskid": 44556677,
        "attemptnr": 2,
        "maxattempt": 5,
        "transformation": "Sim_tf.py",
        "pilotid": _PILOTID,
    }
    job.update(overrides)
    return job


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

SCENARIOS: tuple[LogScenario, ...] = (
    LogScenario(
        name="payload_1305_stdout_and_stderr",
        pins=(
            "The full payload path: setup.stdout first, then both payload "
            "logs, joined with the stderr separator, with the stderr "
            "traceback preferred over anything in stdout."
        ),
        job=_job(),
        texts={
            SETUP_LOG: _SETUP_CLEAN,
            PAYLOAD_STDOUT: _PAYLOAD_CHATTER * 200,
            PAYLOAD_STDERR: _PILOT_TRACEBACK,
        },
        sizes={SETUP_LOG: 120, PAYLOAD_STDOUT: 6400, PAYLOAD_STDERR: 420},
        expect_fetched=(SETUP_LOG, PAYLOAD_STDOUT, PAYLOAD_STDERR),
        expect_failure_type="pilot_exception",
    ),
    LogScenario(
        name="payload_1305_setup_error",
        pins=(
            "The early return.  A setup error means the payload never ran, "
            "so neither path may download the payload logs even though the "
            "listing says they have content."
        ),
        job=_job(),
        texts={
            SETUP_LOG: _SETUP_ERROR,
            PAYLOAD_STDOUT: _PAYLOAD_CHATTER * 200,
            PAYLOAD_STDERR: _PILOT_TRACEBACK,
        },
        sizes={SETUP_LOG: 130, PAYLOAD_STDOUT: 6400, PAYLOAD_STDERR: 420},
        expect_fetched=(SETUP_LOG,),
        expect_failure_type="setup_release_not_found",
    ),
    LogScenario(
        name="payload_1305_clean_setup_no_payload_logs",
        pins=(
            "The clean-setup fallback fixed in B7a.  Both payload files are "
            "zero-length, so the error-free setup log is the only evidence "
            "and the verdict is taken from it — it used to be taken from an "
            "empty excerpt."
        ),
        job=_job(),
        texts={SETUP_LOG: _SETUP_CLEAN},
        sizes={SETUP_LOG: 120, PAYLOAD_STDOUT: 0, PAYLOAD_STDERR: 0},
        expect_fetched=(SETUP_LOG,),
        expect_failure_type="payload_error",
    ),
    LogScenario(
        name="payload_1305_stdout_only",
        pins=(
            "A zero-length payload.stderr is skipped by both paths, and the "
            "excerpt is then payload.stdout alone with no separator.  The "
            "version line in payload.stdout must be ignored by both: only "
            "pilotlog.txt answers for the pilot version."
        ),
        job=_job(),
        texts={SETUP_LOG: _SETUP_CLEAN, PAYLOAD_STDOUT: _PAYLOAD_SEGFAULT},
        sizes={SETUP_LOG: 120, PAYLOAD_STDOUT: 900, PAYLOAD_STDERR: 0},
        expect_fetched=(SETUP_LOG, PAYLOAD_STDOUT),
        expect_failure_type="segfault",
    ),
    LogScenario(
        name="payload_1305_stderr_only",
        pins=(
            "The mirror image: payload.stdout is zero-length, so the "
            "combined excerpt is an empty stdout section, the separator, "
            "then stderr.  The separator is written either way."
        ),
        job=_job(),
        texts={SETUP_LOG: _SETUP_CLEAN, PAYLOAD_STDERR: _PILOT_TRACEBACK},
        sizes={SETUP_LOG: 120, PAYLOAD_STDOUT: 0, PAYLOAD_STDERR: 420},
        expect_fetched=(SETUP_LOG, PAYLOAD_STDERR),
        expect_failure_type="pilot_exception",
    ),
    LogScenario(
        name="payload_1305_large_stderr",
        pins=(
            "The stderr reservation.  payload.stderr is excerpted against "
            "2000 characters and payload.stdout against the budget minus "
            "2000, so that the joined excerpt cannot exceed it.  Both files "
            "here are larger than their share."
        ),
        job=_job(),
        texts={
            SETUP_LOG: _SETUP_CLEAN,
            PAYLOAD_STDOUT: _PAYLOAD_CHATTER * 300,
            PAYLOAD_STDERR: _BIG_PAYLOAD_STDERR,
        },
        sizes={SETUP_LOG: 120, PAYLOAD_STDOUT: 9600, PAYLOAD_STDERR: 8000},
        expect_fetched=(SETUP_LOG, PAYLOAD_STDOUT, PAYLOAD_STDERR),
        expect_failure_type="pilot_exception",
    ),
    LogScenario(
        name="payload_1305_large_setup_error",
        pins=(
            "An erroring setup.stdout longer than the budget keeps the head "
            "of the file, not the tail extraction would give it: setup "
            "failures are shell output, and the asetup diagnostics are at "
            "the top.  Losing that rule changes the verdict here, not just "
            "the excerpt."
        ),
        job=_job(),
        texts={SETUP_LOG: _BIG_SETUP_ERROR},
        sizes={SETUP_LOG: 11000, PAYLOAD_STDOUT: 6400, PAYLOAD_STDERR: 420},
        expect_fetched=(SETUP_LOG,),
        expect_failure_type="setup_release_not_found",
    ),
    LogScenario(
        name="pilotlog_larger_than_the_budget",
        pins=(
            "The pilot log is excerpted against the whole budget — it is "
            "never joined to a second file — so this scenario separates the "
            "full budget from the payload path's reduced one."
        ),
        job=_job(code=1151, diag="File transfer timed out during stage-in"),
        texts={PILOT_LOG: _BIG_PILOT_LOG},
        sizes={PILOT_LOG: 18000},
        expect_fetched=(PILOT_LOG,),
        expect_failure_type="pilot_exception",
    ),
    LogScenario(
        name="payload_1305_no_usable_logs",
        pins=(
            "Every file zero-length: nothing is downloaded, and both paths "
            "classify from metadata alone rather than reporting an error."
        ),
        job=_job(diag="Payload failed with an unknown error"),
        texts={},
        sizes={SETUP_LOG: 0, PAYLOAD_STDOUT: 0, PAYLOAD_STDERR: 0},
        expect_fetched=(),
        expect_failure_type="unknown",
    ),
    LogScenario(
        name="listing_unavailable_fails_open",
        pins=(
            "An unavailable listing offers every file rather than "
            "suppressing it.  payload.stderr is then attempted and turns out "
            "to be unreadable, which is not the same as zero-length."
        ),
        job=_job(),
        texts={SETUP_LOG: _SETUP_CLEAN, PAYLOAD_STDOUT: _PAYLOAD_SEGFAULT},
        sizes=None,
        expect_fetched=(SETUP_LOG, PAYLOAD_STDOUT, PAYLOAD_STDERR),
        expect_failure_type="segfault",
    ),
    LogScenario(
        name="pilotlog_stagein_timeout",
        pins=(
            "The non-1305 path: one file, pilotlog.txt, anchored on the "
            "1151 pattern, with the pilot version read from the start-up "
            "banner far above the failure — and preferred over the differing "
            "version in this job's pilotid."
        ),
        job=_job(
            code=1151,
            diag="File transfer timed out during stage-in",
            pilotid=_PILOTID_STALE,
        ),
        texts={PILOT_LOG: _PILOT_LOG_TIMEOUT},
        sizes={PILOT_LOG: 5200},
        expect_fetched=(PILOT_LOG,),
        expect_failure_type="stagein_timeout",
    ),
    LogScenario(
        name="pilotlog_version_falls_back_to_pilotid",
        pins=(
            "A truncated pilot log carries no version banner, so both paths "
            "fall back to the pilotid metadata field."
        ),
        job=_job(code=1151, diag="File transfer timed out during stage-in"),
        texts={PILOT_LOG: _PILOT_LOG_NO_VERSION},
        sizes={PILOT_LOG: 5100},
        expect_fetched=(PILOT_LOG,),
        expect_failure_type="stagein_timeout",
    ),
    LogScenario(
        name="unparseable_pilot_error_code",
        pins=(
            "A non-numeric piloterrorcode coerces to 0 on both paths, which "
            "puts the job on the pilotlog strategy rather than the payload "
            "one.  A disagreement here would send the two paths at different "
            "files."
        ),
        job=_job(code="not-a-number", diag="Pilot failed"),
        texts={PILOT_LOG: _PILOT_LOG_NETWORK},
        sizes={PILOT_LOG: 800},
        expect_fetched=(PILOT_LOG,),
        expect_failure_type="network",
    ),
    LogScenario(
        name="pilotlog_zero_length",
        pins=(
            "The pilot log is confirmed empty, so the non-1305 path "
            "downloads nothing at all."
        ),
        job=_job(code=1151, diag="File transfer timed out during stage-in"),
        texts={},
        sizes={PILOT_LOG: 0},
        expect_fetched=(),
        expect_failure_type="stagein_timeout",
    ),
    LogScenario(
        name="metadata_only_finished_job",
        pins=(
            "A job outside failed/holding/cancelled has no logs fetched at "
            "all — the monolith skips the download step and plan_fetch "
            "reports the metadata_only strategy."
        ),
        job=_job(code=0, status="finished", diag=""),
        texts={PILOT_LOG: _PILOT_LOG_TIMEOUT},
        sizes={PILOT_LOG: 5200},
        expect_fetched=(),
        expect_failure_type="unknown",
    ),
    # --- the metadata subset against _build_search_text -------------------
    # Each of these classifies on one job field that fetch_metadata must
    # carry.  Where a log is present it says something else, so a dropped
    # field changes the verdict rather than going unnoticed.
    LogScenario(
        name="metadata_commandtopilot_outranks_the_log",
        pins=(
            "commandtopilot is in the metadata subset although "
            "fetch_and_analyse never promotes it to evidence.  A job "
            "reassigned by JEDI never really failed, so the traceback in its "
            "log is incidental and must not win."
        ),
        job=_job(commandtopilot="toreassign"),
        texts={PAYLOAD_STDOUT: _PILOT_TRACEBACK},
        sizes={SETUP_LOG: 0, PAYLOAD_STDOUT: 420, PAYLOAD_STDERR: 0},
        expect_fetched=(PAYLOAD_STDOUT,),
        expect_failure_type="reassigned_by_jedi",
    ),
    LogScenario(
        name="metadata_jobsubstatus_outranks_the_log",
        pins="jobsubstatus is read by _build_search_text; same test, other field.",
        job=_job(jobsubstatus="toreassign"),
        texts={PAYLOAD_STDOUT: _PILOT_TRACEBACK},
        sizes={SETUP_LOG: 0, PAYLOAD_STDOUT: 420, PAYLOAD_STDERR: 0},
        expect_fetched=(PAYLOAD_STDOUT,),
        expect_failure_type="reassigned_by_jedi",
    ),
    LogScenario(
        name="metadata_taskbuffererrordiag_classifies",
        pins=(
            "taskbuffererrordiag is searched ahead of piloterrordiag and "
            "classifies a job with no readable log."
        ),
        job=_job(
            code=0,
            diag="",
            taskbuffererrordiag="The job was killed: walltime exceeded",
        ),
        texts={},
        sizes={PILOT_LOG: 0},
        expect_fetched=(),
        expect_failure_type="timeout",
    ),
    LogScenario(
        name="metadata_exeerrordiag_classifies",
        pins="exeerrordiag is searched too; a dropped field would read unknown.",
        job=_job(
            code=0,
            diag="",
            exeerrordiag="Segmentation fault (core dumped) in AthenaMP",
        ),
        texts={},
        sizes={PILOT_LOG: 0},
        expect_fetched=(),
        expect_failure_type="segfault",
    ),
)


def by_name(name: str) -> LogScenario:
    """Return one scenario by name.

    Args:
        name: The scenario's :attr:`LogScenario.name`.

    Returns:
        The matching scenario.

    Raises:
        KeyError: If no scenario has that name.
    """
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise KeyError(name)
