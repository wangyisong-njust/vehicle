from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_header(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return next(csv.reader(f))


def iter_dict_rows(path: str | Path) -> Iterable[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def write_csv(path: str | Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

