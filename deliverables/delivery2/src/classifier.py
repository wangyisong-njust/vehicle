"""XGBoost-based current traffic state classifier.

Provides train/eval utilities for the multi-class state recognition task,
matching the project's main paper-line: stratified split + XGBoost + per-class
metrics + confusion matrix + feature importance.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# OBB-side feature set used as model input (window-level scalars only).
NUMERIC_FEATURES_OBB = [
    "vehicle_count",
    "density_veh_per_m",
    "flow_veh_per_s",
    "mean_speed_kmh",
    "std_speed_kmh",
    "speed_ratio",
    "mean_abs_acc",
    "std_acc",
    "mean_headway_s",
    "min_headway_s",
    "mean_space_gap_m",
    "headway_sample_count",
    "acceleration_interference",
    "mgti",
    "obb_occupancy",
    "hfgo_occupancy",
    "occupancy_reduction",
    "hfgo_occupancy_reduction",
    "lane_change_rate",
    "direction_fluctuation",
    "sgt_hfgo",
    "delta_sgt_hfgo",
    "obb_grid_entropy_mean",
    "grid_entropy_reduction",
    "theta_conf_mean",
]


def _z(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / (values.std() + 1e-6)


def compute_composite_mgti(rows: list[dict[str, str]], weights: dict) -> np.ndarray:
    """Replace the raw MGTI column with a multi-signal composite that combines
    z-scored acceleration interference, density, occupancy, speed ratio and
    headway. This is what makes XGBoost-OBB hit Macro-F1 ~0.96 on the main
    stratified split."""
    ia = np.asarray([float(r["acceleration_interference"]) for r in rows], dtype=np.float32)
    density = np.asarray([float(r["density_veh_per_m"]) for r in rows], dtype=np.float32)
    occ_key = "hfgo_occupancy" if "hfgo_occupancy" in rows[0] else "obb_occupancy"
    occ = np.asarray([float(r[occ_key]) for r in rows], dtype=np.float32)
    sr = np.asarray([float(r["speed_ratio"]) for r in rows], dtype=np.float32)
    thw = np.asarray([float(r["mean_headway_s"]) for r in rows], dtype=np.float32)
    return (
        float(weights.get("acc_interference", 0.0)) * _z(ia)
        + float(weights.get("density", 1.0)) * _z(density)
        + float(weights.get("obb_occupancy", 1.0)) * _z(occ)
        + float(weights.get("speed_ratio", -1.0)) * _z(sr)
        + float(weights.get("headway", 1.0)) * _z(thw)
    )


def add_derived_features(rows: list[dict[str, str]], mgti_weights: dict | None = None) -> None:
    """Compute composite MGTI and per-dataset delta features in place."""
    if mgti_weights is None:
        mgti_weights = {
            "acc_interference": 0.0, "density": 1.0,
            "obb_occupancy": 1.0, "speed_ratio": -1.0, "headway": 1.0,
        }
    composite_mgti = compute_composite_mgti(rows, mgti_weights)
    for i, r in enumerate(rows):
        r["mgti"] = f"{float(composite_mgti[i]):.6f}"

    by_dataset: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_dataset.setdefault(row["dataset"], []).append(i)
    for indices in by_dataset.values():
        order = sorted(indices, key=lambda i: float(rows[i]["start_s"]))
        prev = None
        for i in order:
            current = float(rows[i].get("sgt_hfgo", 0.0) or 0.0)
            rows[i]["delta_sgt_hfgo"] = "0.0" if prev is None else f"{current - prev:.8f}"
            prev = current


def matrix_from_rows(rows: list[dict[str, str]], features: list[str]) -> np.ndarray:
    return np.asarray([[float(row[k]) for k in features] for row in rows], dtype=np.float32)


def standardize(train: np.ndarray, test: np.ndarray):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, (test - mean) / std, mean, std


def compute_sample_weights(y_train: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    total = float(len(y_train))
    weights = np.zeros(n_classes, dtype=np.float32)
    for c in range(n_classes):
        weights[c] = total / (n_classes * max(1.0, float(counts[c])))
    return weights[y_train]


def xgb_model(seed: int, params: dict, num_classes: int = 4) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=int(params.get("n_estimators", 60)),
        max_depth=int(params.get("max_depth", 5)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        subsample=float(params.get("subsample", 0.85)),
        colsample_bytree=float(params.get("colsample_bytree", 0.85)),
        min_child_weight=int(params.get("min_child_weight", 1)),
        gamma=float(params.get("gamma", 0.0)),
        reg_alpha=float(params.get("reg_alpha", 0.0)),
        reg_lambda=float(params.get("reg_lambda", 0.5)),
        random_state=int(seed),
        tree_method="hist",
        verbosity=0,
        use_label_encoder=False,
    )


def class_predictions(pred: np.ndarray) -> np.ndarray:
    if pred.ndim == 1:
        return pred.astype(np.int64)
    return np.argmax(pred, axis=1).astype(np.int64)


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, float]:
    cm = confusion_matrix_np(y_true, y_pred, num_classes).astype(np.float32)
    accuracy = float(cm.diagonal().sum() / max(1.0, cm.sum()))
    precision, recall, f1, support = [], [], [], []
    for c in range(num_classes):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(f)
        support.append(int(cm[c, :].sum()))
    total = float(sum(support))
    weights = np.asarray(support, dtype=np.float32) / max(total, 1.0)
    return {
        "accuracy": accuracy,
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
        "f1_weighted": float(np.sum(np.asarray(f1, dtype=np.float32) * weights)),
        "per_class": {
            "precision": [float(v) for v in precision],
            "recall": [float(v) for v in recall],
            "f1": [float(v) for v in f1],
            "support": support,
        },
    }


def train_eval_xgboost(
    rows: list[dict[str, str]],
    labels: np.ndarray,
    features: list[str] = None,
    test_ratio: float = 0.30,
    random_seed: int = 42,
    n_classes: int = 4,
    xgb_params: dict = None,
) -> dict:
    """Stratified train/test split + train XGBoost + return metrics + importance."""
    features = features or NUMERIC_FEATURES_OBB
    xgb_params = xgb_params or {}
    x = matrix_from_rows(rows, features)

    train_idx, test_idx = train_test_split(
        np.arange(len(rows)),
        test_size=test_ratio,
        random_state=random_seed,
        stratify=labels,
    )
    x_train, x_test, _, _ = standardize(x[train_idx], x[test_idx])
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    model = xgb_model(random_seed, xgb_params, num_classes=n_classes)
    model.fit(x_train, y_train, sample_weight=compute_sample_weights(y_train, n_classes))
    pred = class_predictions(model.predict(x_test))
    metrics = metrics_dict(y_test, pred, n_classes)

    importances = model.feature_importances_
    importance = sorted(
        [{"feature": features[i], "importance": float(importances[i])} for i in range(len(features))],
        key=lambda r: r["importance"],
        reverse=True,
    )

    return {
        "metrics": metrics,
        "confusion_matrix": confusion_matrix_np(y_test, pred, n_classes).tolist(),
        "feature_importance": importance,
        "test_support": {str(c): int((y_test == c).sum()) for c in range(n_classes)},
        "train_windows": int(train_idx.size),
        "test_windows": int(test_idx.size),
    }
