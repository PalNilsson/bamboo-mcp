"""Module entry point for ``python -m interfaces.agent.job_agent``.

Kept separate from :mod:`interfaces.agent.job_agent.cli` so that importing the
CLI module — which tests do, to exercise the parser and the renderers — never
runs anything.
"""
from __future__ import annotations

from interfaces.agent.job_agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
