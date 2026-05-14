#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ute_pipeline.config import project_root
from ute_pipeline.reporting import write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate docs/experiment_report.md from current outputs.")
    parser.parse_args()

    root = project_root()
    write_report(root)
    print(f"[REPORT] wrote {root / 'docs' / 'experiment_report.md'}")


if __name__ == "__main__":
    main()
