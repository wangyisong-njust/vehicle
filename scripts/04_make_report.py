#!/usr/bin/env python
from __future__ import annotations

from ute_pipeline.config import project_root
from ute_pipeline.reporting import write_report


def main() -> None:
    root = project_root()
    write_report(root)
    print(f"[REPORT] wrote {root / 'docs' / 'experiment_report.md'}")


if __name__ == "__main__":
    main()
