import subprocess
import sys
from pathlib import Path


def test_cli_help() -> None:
    repo = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, "-m", "llm_evaluation.cli", "--help"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    assert "llm_evaluation" in r.stdout or "LLM" in r.stdout or "config" in r.stdout
