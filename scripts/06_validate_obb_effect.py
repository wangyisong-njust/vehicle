#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ute_pipeline.config import load_config, project_root
from ute_pipeline.experiments import (
    ABLATION_FEATURE_SETS,
    NUMERIC_FEATURES_HBB,
    NUMERIC_FEATURES_OBB,
    STATE_NAMES,
    make_state_labels,
    matrix_from_rows,
    metrics_dict,
    read_feature_table,
    standardize,
    xgb_model,
)


class _SubTable:
    def __init__(self, rows: list[dict[str, str]], dataset: str):
        self.rows = rows
        self.dataset = np.asarray([dataset] * len(rows))
        self.y = None
        self.x_obb = np.asarray([[float(r[k]) for k in NUMERIC_FEATURES_OBB] for r in rows], dtype=np.float32) if rows else np.empty((0, len(NUMERIC_FEATURES_OBB)), dtype=np.float32)
        self.x_hbb = np.asarray([[float(r[k]) for k in NUMERIC_FEATURES_HBB] for r in rows], dtype=np.float32) if rows else np.empty((0, len(NUMERIC_FEATURES_HBB)), dtype=np.float32)
        self.start_s = np.zeros(len(rows), dtype=np.float32)
        self.mgti_composite = None


def _mean(rows: list[dict[str, str]], key: str) -> float:
    return float(np.mean([float(r[key]) for r in rows])) if rows else 0.0


def _eval_pair(rows: list[dict[str, str]], labels: np.ndarray, features: list[str], n_states: int) -> dict[str, float]:
    rng = np.random.default_rng(42)
    train_parts = []
    test_parts = []
    all_idx = np.arange(len(rows))
    for cls in range(n_states):
        cls_idx = all_idx[labels == cls]
        if cls_idx.size == 0:
            continue
        shuffled = cls_idx.copy()
        rng.shuffle(shuffled)
        n_test = max(1, int(round(cls_idx.size * 0.25))) if cls_idx.size > 1 else 1
        test_parts.append(shuffled[:n_test])
        train_parts.append(shuffled[n_test:])
    train_idx = np.sort(np.concatenate(train_parts)) if train_parts else np.asarray([], dtype=np.int64)
    test_idx = np.sort(np.concatenate(test_parts)) if test_parts else np.asarray([], dtype=np.int64)
    if test_idx.size < 8:
        return {"accuracy": 0.0, "f1_macro": 0.0}
    x = matrix_from_rows(rows, features)
    x_train, x_test, _, _ = standardize(x[train_idx], x[test_idx])
    model = xgb_model(
        42,
        {
            "n_estimators": 180,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
        num_classes=n_states,
    )
    model.fit(x_train, labels[train_idx])
    pred = model.predict(x_test)
    return metrics_dict(labels[test_idx], pred, num_classes=n_states)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "datasets.json")
    n_states = int(cfg["feature"].get("n_states", 3))
    table = read_feature_table(root / "outputs" / "features" / "all_windows.csv")
    datasets = ["xamn6", "xamn5", "pkdd8"]
    payload: dict[str, object] = {}

    for dataset in datasets:
        idx = np.where(table.dataset == dataset)[0]
        rows = [table.rows[i] for i in idx]
        if not rows:
            continue
        sub = _SubTable(rows, dataset)
        labels, _ = make_state_labels(sub, cfg, main_dataset=dataset)

        baseline = _eval_pair(rows, labels, ABLATION_FEATURE_SETS["M1: V+D"], n_states)
        reference = _eval_pair(rows, labels, ABLATION_FEATURE_SETS["M3: V+D+R+F"], n_states)
        ours = _eval_pair(rows, labels, ABLATION_FEATURE_SETS["M4: Ours+headway+acc+MGTI"], n_states)

        payload[dataset] = {
            "windows": len(rows),
            "label_distribution": {STATE_NAMES[i] if i < len(STATE_NAMES) else str(i): int(np.sum(labels == i)) for i in range(n_states)},
            "spatial_effect": {
                "occupancy_reduction_mean": _mean(rows, "occupancy_reduction"),
                "hfgo_occupancy_reduction_mean": _mean(rows, "hfgo_occupancy_reduction"),
            },
            "method1_vd": baseline,
            "method3_vdrf": reference,
            "method4_ours": ours,
            "reference_minus_baseline": {
                "accuracy_point": float(reference["accuracy"] - baseline["accuracy"]),
                "f1_macro_point": float(reference["f1_macro"] - baseline["f1_macro"]),
            },
            "ours_minus_reference": {
                "accuracy_point": float(ours["accuracy"] - reference["accuracy"]),
                "f1_macro_point": float(ours["f1_macro"] - reference["f1_macro"]),
            },
        }

    out_json = root / "outputs" / "reports" / "obb_effect_validation.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OBB-VALID] wrote {out_json}")


if __name__ == "__main__":
    main()
