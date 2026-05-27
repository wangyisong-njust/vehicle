#!/usr/bin/env python
"""第二步：基于 K-Means + 物理顺序兜底生成四类交通状态标签。

输入：
  - outputs/features/all_windows.csv

输出：
  - outputs/labels/state_labels.csv      ← 每窗一标签 + 风险分数
  - outputs/reports/labels_summary.json
  - outputs/figures/state_distribution.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from state_labels import make_state_labels, STATE_NAMES, STATE_NAMES_EN
from plotting import plot_state_distribution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "datasets.json"))
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    feature_cfg = cfg["feature"]
    main_ds = cfg["experiment"].get("main_dataset", "xamn6")

    in_csv = ROOT / "outputs" / "features" / "all_windows.csv"
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing {in_csv}. Run 01_extract_features.py first.")

    with in_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    dataset_all = np.asarray([r["dataset"] for r in rows])
    start_s_all = np.asarray([float(r["start_s"]) for r in rows], dtype=np.float32)

    n_states = int(feature_cfg.get("n_states", 4))
    quantiles = tuple(feature_cfg.get("quantiles", [0.25, 0.50, 0.75]))
    labels, score, thresholds = make_state_labels(
        rows_all=rows,
        dataset_all=dataset_all,
        start_s_all=start_s_all,
        main_dataset=main_ds,
        n_states=n_states,
        quantiles=quantiles,
        random_seed=int(cfg["experiment"].get("random_seed", 42)),
        smoothing_window=int(feature_cfg.get("label_smoothing_window", 5)),
    )

    # Write labels CSV
    out_csv = ROOT / "outputs" / "labels" / "state_labels.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "window_id", "start_s", "label", "label_name", "score"])
        for i, r in enumerate(rows):
            label_idx = int(labels[i])
            writer.writerow([r["dataset"], r["window_id"], r["start_s"],
                             label_idx, STATE_NAMES[label_idx], f"{float(score[i]):.6f}"])
    print(f"[LABEL] wrote {len(rows)} labels -> {out_csv}")

    # Summary
    counts_main = np.bincount(labels[dataset_all == main_ds], minlength=n_states).tolist()
    summary = {
        "main_dataset": main_ds,
        "n_states": n_states,
        "state_names": STATE_NAMES[:n_states],
        "thresholds": thresholds.tolist(),
        "label_counts_main": {STATE_NAMES[c]: int(counts_main[c]) for c in range(n_states)},
        "label_counts_per_dataset": {
            ds: {
                STATE_NAMES[c]: int(np.sum((dataset_all == ds) & (labels == c)))
                for c in range(n_states)
            }
            for ds in sorted(set(dataset_all.tolist()))
        },
    }
    summary_path = ROOT / "outputs" / "reports" / "labels_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[LABEL] summary -> {summary_path}")

    # Figure
    main_labels = labels[dataset_all == main_ds]
    fig_path = ROOT / "outputs" / "figures" / "state_distribution.png"
    plot_state_distribution(main_labels, STATE_NAMES_EN[:n_states], fig_path)
    print(f"[LABEL] figure -> {fig_path}")


if __name__ == "__main__":
    main()
