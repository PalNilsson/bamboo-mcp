"""Tests for the _normalise_latex helper in interfaces/streamlit/chat.py.

Imports the function directly from the module without starting Streamlit.
Streamlit itself is mocked out via sys.modules so the module can be imported
in a headless test environment.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _import_normalise_latex():
    """Import _normalise_latex from streamlit chat without starting Streamlit.

    Stubs are injected only for the duration of this import and are removed
    from ``sys.modules`` afterwards so they do not interfere with other test
    modules (in particular ``test_superuser_guard.py`` which imports the real
    ``interfaces.shared.superuser_guard``).

    Returns:
        The _normalise_latex callable.
    """
    import importlib.util
    from pathlib import Path

    st_mock = MagicMock()
    st_mock.session_state = {}

    # Track which names we are *adding* so we can remove them afterwards.
    # Names already present are left untouched (and not removed).
    stub_names = [
        "streamlit",
        "streamlit.components",
        "streamlit.components.v1",
        "plotly",
        "plotly.graph_objects",
        "plotly.express",
        "interfaces",
        "interfaces.shared",
        "interfaces.shared.mcp_client",
        "interfaces.shared.deeplink",
        "interfaces.shared.superuser_guard",
    ]
    stubs = {
        "streamlit": st_mock,
        "streamlit.components": MagicMock(),
        "streamlit.components.v1": MagicMock(),
        "plotly": MagicMock(),
        "plotly.graph_objects": MagicMock(),
        "plotly.express": MagicMock(),
        "interfaces": types.ModuleType("interfaces"),
        "interfaces.shared": types.ModuleType("interfaces.shared"),
        "interfaces.shared.mcp_client": MagicMock(),
        "interfaces.shared.deeplink": MagicMock(),
        "interfaces.shared.superuser_guard": MagicMock(),
    }

    added: list[str] = []
    saved: dict[str, types.ModuleType] = {}
    for name in stub_names:
        if name in sys.modules:
            saved[name] = sys.modules[name]
        else:
            added.append(name)
        sys.modules[name] = stubs[name]  # type: ignore[assignment]

    try:
        spec = importlib.util.spec_from_file_location(
            "streamlit_chat",
            Path(__file__).parent.parent
            / "interfaces"
            / "streamlit"
            / "chat.py",
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod._normalise_latex
    finally:
        # Restore sys.modules to its pre-stub state so other test modules
        # that import interfaces.shared.superuser_guard get the real module.
        for name in added:
            sys.modules.pop(name, None)
        for name, original in saved.items():
            sys.modules[name] = original
        # Also evict the dynamically loaded chat module so it does not
        # retain references to the stubs.
        sys.modules.pop("streamlit_chat", None)


class TestNormaliseLatex:
    """Unit tests for _normalise_latex."""

    @classmethod
    def setup_class(cls) -> None:
        """Import the function once for the whole class."""
        cls._fn = staticmethod(_import_normalise_latex())

    def _n(self, text: str) -> str:
        return self.__class__._fn(text)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Display math: \[ ... \]
    # ------------------------------------------------------------------

    def test_backslash_bracket_display_math(self) -> None:
        r"""'\[ expr \]' is converted to '$$expr$$'."""
        result = self._n(r"\[ M = \sqrt{E^2} \]")
        assert result == r"$$ M = \sqrt{E^2} $$"

    def test_backslash_bracket_multiline(self) -> None:
        r"""Multiline '\[ ... \]' blocks are converted correctly."""
        inp = "\\[\nM = \\sqrt{E^2}\n\\]"
        result = self._n(inp)
        assert result.startswith("$$")
        assert result.endswith("$$")

    # ------------------------------------------------------------------
    # Inline math: \( ... \)
    # ------------------------------------------------------------------

    def test_backslash_paren_inline_math(self) -> None:
        r"""'\( expr \)' is converted to '$expr$'."""
        result = self._n(r"The mass is \( m = E/c^2 \) in natural units.")
        assert "$" in result
        assert r"\(" not in result
        assert r"\)" not in result

    # ------------------------------------------------------------------
    # Bare bracket heuristic: [ \latex ] → $$\latex$$
    # ------------------------------------------------------------------

    def test_bare_bracket_with_latex_command(self) -> None:
        r"""'[ \sqrt{...} ]' (containing backslash) is converted to '$$...$$'."""
        inp = r"[ M = \sqrt{\left(\sum_i E_i\right)^2} ]"
        result = self._n(inp)
        assert result.startswith("$$")
        assert result.endswith("$$")

    def test_bare_bracket_plain_prose_not_converted(self) -> None:
        """'[see also: docs]' (no backslash) is left unchanged."""
        inp = "See [see also: docs] for more details."
        result = self._n(inp)
        assert result == inp

    def test_bare_bracket_url_not_converted(self) -> None:
        """'[link text]' without backslash is left unchanged."""
        inp = "Click [here] for more info."
        result = self._n(inp)
        assert result == inp

    # ------------------------------------------------------------------
    # Real formula from the failing response
    # ------------------------------------------------------------------

    def test_invariant_mass_formula(self) -> None:
        r"""The actual formula from the screenshot is converted correctly."""
        inp = (
            r"[ M = \sqrt{\left(\sum_i E_i\right)^2"
            r" - \left|\sum_i \vec{p}_i\right|^2} ]"
        )
        result = self._n(inp)
        assert result.startswith("$$")
        assert result.endswith("$$")
        assert r"\sqrt" in result

    # ------------------------------------------------------------------
    # No-op cases — plain text must not be modified
    # ------------------------------------------------------------------

    def test_plain_text_unchanged(self) -> None:
        """Plain prose with no LaTeX is returned unchanged."""
        inp = "The invariant mass is a Lorentz scalar."
        assert self._n(inp) == inp

    def test_already_dollar_delimited_unchanged(self) -> None:
        """Text already using $$ delimiters is returned unchanged."""
        inp = r"$$M = \sqrt{E^2 - p^2}$$"
        result = self._n(inp)
        assert result == inp

    def test_already_inline_dollar_unchanged(self) -> None:
        """Text already using $ delimiters is returned unchanged."""
        inp = r"The mass $m$ is related to energy by $E = mc^2$."
        result = self._n(inp)
        assert result == inp

    def test_empty_string(self) -> None:
        """Empty string is returned unchanged."""
        assert self._n("") == ""

    def test_mixed_content(self) -> None:
        r"""Text with prose, a formula, and more prose is handled correctly."""
        inp = (
            "The invariant mass formula is:\n"
            r"\[ M = \sqrt{E^2 - p^2} \]"
            "\nwhere E is energy and p is momentum."
        )
        result = self._n(inp)
        assert "$$" in result
        assert r"\[" not in result
        assert r"\]" not in result
        assert "where E is energy" in result
