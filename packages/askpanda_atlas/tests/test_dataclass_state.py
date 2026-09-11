"""Tests for the ``to_state``/``from_state`` codecs on the log-analysis dataclasses.

These codecs exist so that state can survive a round trip out of the process
and back — through a recorded fixture, or through a caller that composes tool
calls itself and carries intermediate results between them.  That is a
different job from the ``as_dict`` methods, which project onto evidence keys
for the LLM and are deliberately lossy: ``as_dict`` renames ``exc_type`` to
``type``, drops ``raw`` entirely and adds a derived ``deepest_pilot_frame``.

The central property asserted here is therefore round-trip identity —
``from_state(x.to_state()) == x`` — plus the tolerance rules that let state
written by a different version, or by hand, still load.  A separate class
asserts that adding these methods did not disturb ``as_dict``, since the
evidence path and the regression-gated ``test_log_analysis.py`` both depend on
its exact keys.
"""
from __future__ import annotations

from typing import Any

import pytest

from askpanda_atlas._traceback_parse import (
    ExceptionInfo,
    Frame,
    coerce_bool,
    coerce_int,
    coerce_mapping,
    coerce_optional_str,
    coerce_str,
)
from askpanda_atlas.log_analysis_impl import FailureContext, _LogFetchResult


def _frame() -> Frame:
    """Build a pilot frame with every field populated.

    Returns:
        A :class:`Frame` whose ``pilot_path`` is set, so ``is_pilot`` is true.
    """
    return Frame(
        file="/tmp/atlas_1234/pilot3/pilot/util/https.py",
        lineno=412,
        func="request",
        pilot_path="pilot/util/https.py",
    )


def _exception() -> ExceptionInfo:
    """Build an exception with frames, a level and raw traceback text.

    Returns:
        A fully-populated :class:`ExceptionInfo`.
    """
    return ExceptionInfo(
        exc_type="StageInFailure",
        exc_type_full="pilot.common.exception.StageInFailure",
        message="Failed to stage in file",
        frames=[
            Frame(file="/usr/lib/python3.9/socket.py", lineno=704, func="readinto"),
            _frame(),
        ],
        level="ERROR",
        raw="Traceback (most recent call last):\n  ...\nStageInFailure: boom",
    )


# --------------------------------------------------------------------------
# Coercion helpers
# --------------------------------------------------------------------------


class TestCoercions:
    """The tolerant conversions applied when reading a state mapping."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("x", "x"), (None, ""), (42, "42"), (True, "True")],
    )
    def test_coerce_str(self, value: Any, expected: str) -> None:
        """Anything becomes a string; ``None`` becomes empty.

        Args:
            value: Input value.
            expected: Expected coercion.
        """
        assert coerce_str(value) == expected

    def test_coerce_optional_str_preserves_none(self) -> None:
        """``None`` must survive: it means "never fetched", not "empty"."""
        assert coerce_optional_str(None) is None
        assert coerce_optional_str("") == ""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (42, 42),
            ("42", 42),
            ("  42 ", 42),
            ("not a number", 0),
            (None, 0),
            (412.0, 412),
            (3.9, 3),
            (float("nan"), 0),
            (float("inf"), 0),
        ],
    )
    def test_coerce_int(self, value: Any, expected: int) -> None:
        """A JSON number that arrived as text still yields a line number.

        Args:
            value: Input value.
            expected: Expected coercion.
        """
        assert coerce_int(value) == expected

    @pytest.mark.parametrize("value", [True, False])
    def test_coerce_int_rejects_bool(self, value: bool) -> None:
        """``True`` as a line number is a caller error, not line 1.

        ``bool`` is a subclass of ``int``, so without an explicit guard this
        would silently produce 1 or 0 and look like a plausible line number.

        Args:
            value: Boolean input.
        """
        assert coerce_int(value) == 0

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (True, True),
            (False, False),
            ("true", True),
            ("TRUE", True),
            ("yes", True),
            ("1", True),
            ("false", False),
            ("no", False),
            ("", False),
            (None, False),
            (1, True),
        ],
    )
    def test_coerce_bool(self, value: Any, expected: bool) -> None:
        """``"false"`` must be false, not merely a non-empty string.

        This is the case a plain ``bool(value)`` gets wrong, and hand-written
        JSON is exactly where it would arrive.

        Args:
            value: Input value.
            expected: Expected coercion.
        """
        assert coerce_bool(value) is expected

    def test_coerce_mapping(self) -> None:
        """Non-mappings yield empty; integer keys are stringified."""
        assert coerce_mapping({"a": 1}) == {"a": 1}
        assert coerce_mapping({1: "x"}) == {"1": "x"}
        assert coerce_mapping("not a mapping") == {}
        assert coerce_mapping(None) == {}


# --------------------------------------------------------------------------
# Round trips
# --------------------------------------------------------------------------


class TestFrameState:
    """``Frame`` round trip and tolerance."""

    def test_round_trip_is_identity(self) -> None:
        """The defining property of the codec."""
        frame = _frame()
        assert Frame.from_state(frame.to_state()) == frame

    def test_round_trip_of_a_non_pilot_frame(self) -> None:
        """An empty ``pilot_path`` must not become a pilot frame."""
        frame = Frame(file="/usr/lib/python3.9/socket.py", lineno=704, func="readinto")
        restored = Frame.from_state(frame.to_state())
        assert restored == frame
        assert restored.is_pilot is False

    def test_is_pilot_is_not_serialised(self) -> None:
        """Round-tripping a derived flag would let it contradict its field.

        ``as_dict`` carries ``is_pilot`` because evidence consumers read it;
        the codec omits it because it is recomputed from ``pilot_path``.
        """
        assert "is_pilot" not in _frame().to_state()

    def test_unknown_keys_are_ignored(self) -> None:
        """State from a newer version must still load."""
        state = _frame().to_state()
        state["some_future_field"] = "ignored"
        assert Frame.from_state(state) == _frame()

    def test_missing_keys_fall_back_to_defaults(self) -> None:
        """State from an older version must still load."""
        assert Frame.from_state({}) == Frame(file="", lineno=0, func="")

    def test_lineno_arriving_as_text_is_coerced(self) -> None:
        """The realistic failure when a caller hand-writes the state."""
        state = _frame().to_state()
        state["lineno"] = "412"
        assert Frame.from_state(state) == _frame()

    def test_non_mapping_state_yields_defaults(self) -> None:
        """Reachable from caller-supplied input, so it must not raise."""
        assert Frame.from_state("garbage") == Frame(file="", lineno=0, func="")


class TestExceptionInfoState:
    """``ExceptionInfo`` round trip, nesting and tolerance."""

    def test_round_trip_is_identity(self) -> None:
        """Including nested frames and the raw traceback text."""
        exc = _exception()
        assert ExceptionInfo.from_state(exc.to_state()) == exc

    def test_raw_survives_the_round_trip(self) -> None:
        """``as_dict`` drops ``raw``; the codec must not.

        This is the field most likely to be wanted downstream and the reason
        the codec could not simply reuse the evidence projection.
        """
        exc = _exception()
        assert ExceptionInfo.from_state(exc.to_state()).raw == exc.raw
        assert "raw" not in exc.as_dict()

    def test_state_uses_field_names_not_evidence_names(self) -> None:
        """The two representations must be distinguishable at a glance."""
        state = _exception().to_state()
        assert "exc_type" in state and "type" not in state
        assert "exc_type_full" in state and "type_full" not in state

    def test_derived_deepest_frame_is_not_serialised(self) -> None:
        """It is recomputed, and round-tripping it invites contradiction."""
        assert "deepest_pilot_frame" not in _exception().to_state()

    def test_derived_properties_survive_the_round_trip(self) -> None:
        """Restoring the fields is enough to restore the behaviour."""
        restored = ExceptionInfo.from_state(_exception().to_state())
        assert len(restored.pilot_frames) == 1
        assert restored.deepest_pilot_frame == _frame()

    def test_empty_exception_round_trips(self) -> None:
        """The all-defaults instance is a legitimate value."""
        assert ExceptionInfo.from_state(ExceptionInfo().to_state()) == ExceptionInfo()

    def test_a_malformed_frame_does_not_discard_the_others(self) -> None:
        """One bad entry must not cost the whole traceback."""
        state = _exception().to_state()
        state["frames"] = [state["frames"][0], "not a frame", state["frames"][1]]
        restored = ExceptionInfo.from_state(state)
        assert len(restored.frames) == 2

    def test_non_list_frames_yields_no_frames(self) -> None:
        """Tolerated rather than raised, as elsewhere in the codec."""
        state = _exception().to_state()
        state["frames"] = "not a list"
        assert ExceptionInfo.from_state(state).frames == []

    def test_non_mapping_state_yields_defaults(self) -> None:
        """Caller-supplied input must not raise."""
        assert ExceptionInfo.from_state(None) == ExceptionInfo()


class TestFailureContextState:
    """``FailureContext`` round trip, including the nested exception."""

    def test_round_trip_with_an_exception(self) -> None:
        """The nested codec must compose."""
        ctx = FailureContext(excerpt="log text", exception=_exception(), traceback_count=3)
        assert FailureContext.from_state(ctx.to_state()) == ctx

    def test_round_trip_without_an_exception(self) -> None:
        """No traceback is the common case and must survive."""
        ctx = FailureContext(excerpt="log text", exception=None, traceback_count=0)
        restored = FailureContext.from_state(ctx.to_state())
        assert restored == ctx
        assert restored.exception is None

    def test_none_exception_is_distinct_from_an_empty_one(self) -> None:
        """"No traceback" and "a traceback with no detail" differ.

        Collapsing them would make ``traceback_available`` in the evidence
        bundle report a traceback that was never found.
        """
        assert FailureContext(exception=None).to_state()["exception"] is None

        empty = FailureContext(exception=ExceptionInfo()).to_state()["exception"]
        assert empty is not None
        assert empty == ExceptionInfo().to_state()

        assert FailureContext.from_state({"exception": None}).exception is None
        assert (
            FailureContext.from_state({"exception": empty}).exception == ExceptionInfo()
        )

    def test_explicit_null_exception_restores_as_none(self) -> None:
        """Both a missing key and an explicit null mean no traceback."""
        assert FailureContext.from_state({"exception": None}).exception is None
        assert FailureContext.from_state({}).exception is None

    def test_non_mapping_state_yields_defaults(self) -> None:
        """Caller-supplied input must not raise."""
        assert FailureContext.from_state(42) == FailureContext()


class TestLogFetchResultState:
    """``_LogFetchResult`` round trip across its nine fields."""

    @staticmethod
    def _populated() -> _LogFetchResult:
        """Build a result with every field set to a non-default value.

        Returns:
            A fully-populated :class:`_LogFetchResult`.
        """
        return _LogFetchResult(
            log_excerpt="excerpt text",
            log_url="https://example.invalid/pilotlog.txt",
            log_available=True,
            stderr_url="https://example.invalid/payload.stderr",
            setup_log_url="https://example.invalid/setup.stdout",
            setup_log_excerpt="setup text",
            exception=_exception(),
            traceback_count=2,
            pilot_version="3.14.0.22",
        )

    def test_round_trip_is_identity(self) -> None:
        """Every field, including the nested exception."""
        result = self._populated()
        assert _LogFetchResult.from_state(result.to_state()) == result

    def test_default_result_round_trips(self) -> None:
        """The instance ``fetch_and_analyse`` starts from."""
        assert _LogFetchResult.from_state(_LogFetchResult().to_state()) == _LogFetchResult()

    def test_none_urls_stay_none(self) -> None:
        """``None`` means never fetched; ``""`` would claim otherwise.

        Normalising these to ``""`` would make the links section render an
        empty URL rather than omitting the file.
        """
        state = _LogFetchResult().to_state()
        assert state["log_url"] is None
        restored = _LogFetchResult.from_state(state)
        assert restored.log_url is None
        assert restored.stderr_url is None
        assert restored.setup_log_url is None
        assert restored.setup_log_excerpt is None

    def test_empty_string_url_is_distinct_from_none(self) -> None:
        """The codec preserves the distinction in both directions."""
        restored = _LogFetchResult.from_state({"log_url": ""})
        assert restored.log_url == ""
        assert restored.log_url is not None

    def test_log_available_false_survives(self) -> None:
        """A string ``"false"`` must not restore as ``True``.

        This is where a plain truthiness test would flip the meaning of the
        flag that decides whether an excerpt is shown at all.
        """
        assert _LogFetchResult.from_state({"log_available": "false"}).log_available is False
        assert _LogFetchResult.from_state({"log_available": "true"}).log_available is True

    def test_unknown_keys_are_ignored(self) -> None:
        """State from a newer version must still load."""
        state = self._populated().to_state()
        state["future_field"] = ["anything"]
        assert _LogFetchResult.from_state(state) == self._populated()

    def test_non_mapping_state_yields_defaults(self) -> None:
        """Caller-supplied input must not raise."""
        assert _LogFetchResult.from_state([]) == _LogFetchResult()


# --------------------------------------------------------------------------
# Non-interference with the evidence projection
# --------------------------------------------------------------------------


class TestAsDictIsUnchanged:
    """The evidence projection must be exactly what it was.

    ``_build_exception_evidence`` indexes ``as_dict()`` by ``type``,
    ``message``, ``frames`` and ``deepest_pilot_frame``, and the
    regression-gated ``test_log_analysis.py`` asserts the resulting evidence
    keys.  B3 is a pure addition; if any of these fail, it was not.
    """

    def test_frame_as_dict_keys(self) -> None:
        """Five keys, including the derived flag."""
        assert set(_frame().as_dict()) == {
            "file",
            "lineno",
            "func",
            "pilot_path",
            "is_pilot",
        }

    def test_exception_as_dict_keys(self) -> None:
        """Six keys, using the evidence names rather than the field names."""
        assert set(_exception().as_dict()) == {
            "type",
            "type_full",
            "message",
            "level",
            "frames",
            "deepest_pilot_frame",
        }

    def test_as_dict_and_to_state_are_different_shapes(self) -> None:
        """If these ever converge, one of the two has lost its purpose."""
        exc = _exception()
        assert exc.as_dict() != exc.to_state()
        assert set(exc.as_dict()) != set(exc.to_state())
