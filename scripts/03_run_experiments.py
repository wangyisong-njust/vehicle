#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ute_pipeline.config import load_config, project_root
from ute_pipeline.experiments import (
    NUMERIC_FEATURES_OBB,
    compute_composite_mgti,
    make_state_labels,
    pkdd_generalization,
    read_feature_table,
    run_ablation,
    run_classification,
    run_deterioration_prediction,
    run_parameter_sensitivity,
    run_prediction,
    run_robustness,
    run_time_series_cv,
    save_results,
)


def write_updated_feature_tables(root, rows: list[dict[str, str]]) -> None:
    feature_dir = root / "outputs" / "features"
    fieldnames = list(rows[0].keys())
    by_dataset: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)

    for dataset, dataset_rows in by_dataset.items():
        path = feature_dir / f"{dataset}_windows.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dataset_rows)

    merged_path = feature_dir / "all_windows.csv"
    with merged_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets.json")
    args = parser.parse_args()

    root = project_root()
    cfg = load_config(args.config)
    table = read_feature_table(root / "outputs" / "features" / "all_windows.csv")
    n_states = int(cfg["feature"].get("n_states", 3))

    labels, thresholds = make_state_labels(table, cfg, main_dataset="xamn6")
    print("[EXP] state thresholds:", thresholds.tolist())

    # Compute corrected composite MGTI and update table so all experiments use it
    mgti_composite = compute_composite_mgti(table, cfg)
    mgti_col_idx = NUMERIC_FEATURES_OBB.index("mgti")
    table.x_obb[:, mgti_col_idx] = mgti_composite
    for i, row in enumerate(table.rows):
        row["mgti"] = str(float(mgti_composite[i]))
    write_updated_feature_tables(root, table.rows)

    classification = run_classification(table, cfg)
    ablation = run_ablation(table, cfg)
    sensitivity = run_parameter_sensitivity(table, cfg)
    prediction = run_prediction(table, cfg)
    generalization = pkdd_generalization(table, cfg)
    cross_validation = run_time_series_cv(table, cfg)
    deterioration = run_deterioration_prediction(table, cfg)
    robustness = run_robustness(table, cfg)

    main_mask = table.dataset == "xamn6"
    main_labels = labels[main_mask]
    payload = {
        "label_note": "State labels are physically constrained four-class labels built from speed ratio, density and HF-GO occupancy on XAM-N-6. R/F participate in candidate clustering, while final ordering is constrained by traffic-flow monotonicity. Mean headway, acceleration interference and MGTI are excluded from label construction to reduce leakage in ablation experiments.",
        "n_states": n_states,
        "state_thresholds": thresholds.tolist(),
        "state_counts": {str(i): int((main_labels == i).sum()) for i in range(n_states)},
        "classification": classification,
        "ablation": ablation,
        "parameter_sensitivity": sensitivity,
        "prediction": prediction,
        "cross_validation": cross_validation,
        "robustness": robustness,
        "pkdd_generalization": generalization,
        "deterioration": deterioration,
    }
    out_path = root / "outputs" / "reports" / "experiment_results.json"
    save_results(out_path, payload)
    print(f"[EXP] results saved to {out_path}")


if __name__ == "__main__":
    main()
