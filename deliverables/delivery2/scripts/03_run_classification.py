#!/usr/bin/env python
"""第三步：基于 XGBoost 训练当前交通状态分类器并输出指标 / 混淆矩阵 / 特征重要性。

输入：
  - outputs/features/all_windows.csv
  - outputs/labels/state_labels.csv

输出：
  - outputs/reports/classification_results.json
  - outputs/figures/confusion_matrix.png
  - outputs/figures/feature_importance.png
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

from classifier import NUMERIC_FEATURES_OBB, add_derived_features, train_eval_xgboost
from plotting import plot_confusion, plot_feature_importance
from state_labels import STATE_NAMES, STATE_NAMES_EN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "datasets.json"))
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    n_states = int(cfg["feature"].get("n_states", 4))
    main_ds = cfg["experiment"].get("main_dataset", "xamn6")
    test_ratio = float(cfg["experiment"].get("test_ratio", 0.30))
    seed = int(cfg["experiment"].get("random_seed", 42))
    xgb_params = cfg["experiment"].get("xgboost", {})

    # Read features and labels, keep only main dataset rows.
    feature_csv = ROOT / "outputs" / "features" / "all_windows.csv"
    label_csv = ROOT / "outputs" / "labels" / "state_labels.csv"
    if not feature_csv.exists() or not label_csv.exists():
        raise FileNotFoundError("Run 01_extract_features.py and 02_build_state_labels.py first.")

    with feature_csv.open("r", encoding="utf-8") as f:
        feat_rows = list(csv.DictReader(f))
    add_derived_features(feat_rows, mgti_weights=cfg.get("mgti_weights"))
    with label_csv.open("r", encoding="utf-8") as f:
        label_rows = list(csv.DictReader(f))

    # Align by (dataset, window_id)
    label_map = {(r["dataset"], r["window_id"]): int(r["label"]) for r in label_rows}
    keep_rows = []
    keep_labels = []
    for r in feat_rows:
        if r["dataset"] != main_ds:
            continue
        key = (r["dataset"], r["window_id"])
        if key in label_map:
            keep_rows.append(r)
            keep_labels.append(label_map[key])
    keep_labels = np.asarray(keep_labels, dtype=np.int64)
    print(f"[CLS] {main_ds}: {len(keep_rows)} windows, "
          f"label distribution = {np.bincount(keep_labels, minlength=n_states).tolist()}")

    # Train + evaluate XGBoost
    result = train_eval_xgboost(
        keep_rows,
        keep_labels,
        features=NUMERIC_FEATURES_OBB,
        test_ratio=test_ratio,
        random_seed=seed,
        n_classes=n_states,
        xgb_params=xgb_params,
    )

    out_json = ROOT / "outputs" / "reports" / "classification_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[CLS] metrics -> {out_json}")
    m = result["metrics"]
    print(f"[CLS] Macro-F1={m['f1_macro']:.4f}, Accuracy={m['accuracy']:.4f}")

    # Figures
    cm_path = ROOT / "outputs" / "figures" / "confusion_matrix.png"
    fi_path = ROOT / "outputs" / "figures" / "feature_importance.png"
    plot_confusion(result["confusion_matrix"], STATE_NAMES_EN[:n_states], cm_path,
                   title=f"XGBoost (Macro-F1={m['f1_macro']:.4f})")
    plot_feature_importance(result["feature_importance"], fi_path, top_k=12)
    print(f"[CLS] figures -> {cm_path}, {fi_path}")


if __name__ == "__main__":
    main()
