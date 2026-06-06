"""Entrypoint: `uv run llm-eval-dashboard` → Streamlit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    app = Path(__file__).resolve().parent / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))
