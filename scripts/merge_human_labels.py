#!/usr/bin/env python3
"""Merge CSV HITL → predictions.jsonl e reprocessa summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_evaluation.run_reprocess import reprocess_run_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("csv", type=Path)
    args = p.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    csv_path = args.csv.expanduser().resolve()
    reprocess_run_dir(run_dir, hitl_csv=csv_path)
    print(f"OK: {run_dir}")


if __name__ == "__main__":
    main()
