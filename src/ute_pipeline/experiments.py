from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from scipy.stats import ttest_rel
from xgboost import DMatrix, XGBClassifier


STATE_NAMES = ["畅通", "缓行", "拥挤", "堵塞"]
STATE_NAMES_EN = ["Free", "Slow", "Crowded", "Congested"]


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

NUMERIC_FEATURES_HBB = [
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
    "hbb_occupancy",
    "hbb_hfgo_occupancy",
    "lane_change_rate",
]

FEATURES_VD = [
    "vehicle_count",
    "density_veh_per_m",
    "flow_veh_per_s",
    "mean_speed_kmh",
    "std_speed_kmh",
    "speed_ratio",
]

FEATURES_VDR = FEATURES_VD + [
    "lane_change_rate",
]

FEATURES_VDRF = FEATURES_VDR + [
    "direction_fluctuation",
    "theta_conf_mean",
]

FEATURES_VDF = FEATURES_VD + [
    "direction_fluctuation",
    "theta_conf_mean",
]

FEATURES_OURS = FEATURES_VDRF + [
    "hfgo_occupancy",
    "obb_occupancy",
    "mean_headway_s",
    "min_headway_s",
    "headway_sample_count",
    "acceleration_interference",
    "mean_abs_acc",
    "std_acc",
    "mgti",
    "hfgo_occupancy_reduction",
    "sgt_hfgo",
    "delta_sgt_hfgo",
    "hfgo_lgar_005",
    "hfgo_local_diff_mean",
    "hfgo_local_diff_max",
    "obb_grid_entropy_mean",
    "grid_entropy_reduction",
]

ABLATION_FEATURE_SETS = {
    "M1: V+D": FEATURES_VD,
    "M2: V+D+R": FEATURES_VDR,
    "M3': V+D+F": FEATURES_VDF,
    "M3: V+D+R+F": FEATURES_VDRF,
    "M4: Ours+headway+acc+MGTI": FEATURES_OURS,
}

DETERIORATION_ABLATION_SETS = {
    "M1: V+D": FEATURES_VD,
    "M2: V+D+R": FEATURES_VDR,
    "M3': V+D+F": FEATURES_VDF,
    "M3: V+D+R+F": FEATURES_VDRF,
    "M4: Ours+headway+acc+MGTI": [f if f != "mgti" else "mgti_composite" for f in FEATURES_OURS],
}


@dataclass
class FeatureTable:
    rows: list[dict[str, str]]
    x_obb: np.ndarray
    x_hbb: np.ndarray
    dataset: np.ndarray
    start_s: np.ndarray
    y: np.ndarray | None = None
    score: np.ndarray | None = None
    mgti_composite: np.ndarray | None = None


def read_feature_table(path: Path) -> FeatureTable:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    add_derived_features(rows)
    x_obb = np.asarray([[float(row[k]) for k in NUMERIC_FEATURES_OBB] for row in rows], dtype=np.float32)
    x_hbb = np.asarray([[float(row[k]) for k in NUMERIC_FEATURES_HBB] for row in rows], dtype=np.float32)
    dataset = np.asarray([row["dataset"] for row in rows])
    start_s = np.asarray([float(row["start_s"]) for row in rows], dtype=np.float32)
    return FeatureTable(rows=rows, x_obb=x_obb, x_hbb=x_hbb, dataset=dataset, start_s=start_s)


def add_derived_features(rows: list[dict[str, str]]) -> None:
    by_dataset: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_dataset.setdefault(row["dataset"], []).append(i)
    for indices in by_dataset.values():
        order = sorted(indices, key=lambda i: float(rows[i]["start_s"]))
        prev = None
        for i in order:
            current = float(rows[i].get("sgt_hfgo", 0.0) or 0.0)
            delta = 0.0 if prev is None else current - prev
            rows[i]["delta_sgt_hfgo"] = f"{delta:.8f}"
            prev = current


def matrix_from_rows(rows: list[dict[str, str]], features: list[str]) -> np.ndarray:
    return np.asarray([[float(row[k]) for k in features] for row in rows], dtype=np.float32)


def _z(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / (values.std() + 1e-6)


def _robust_scaled(train: np.ndarray, values: np.ndarray) -> np.ndarray:
    scaler = RobustScaler()
    scaler.fit(train.reshape(-1, 1))
    return scaler.transform(values.reshape(-1, 1)).reshape(-1)


def compute_composite_mgti(table: FeatureTable, cfg: dict) -> np.ndarray:
    w = cfg["feature"]["mgti"]["weights"]
    ia = np.asarray([float(r["acceleration_interference"]) for r in table.rows], dtype=np.float32)
    density = np.asarray([float(r["density_veh_per_m"]) for r in table.rows], dtype=np.float32)
    occ_key = "hfgo_occupancy" if "hfgo_occupancy" in table.rows[0] else "obb_occupancy"
    occ = np.asarray([float(r[occ_key]) for r in table.rows], dtype=np.float32)
    sr = np.asarray([float(r["speed_ratio"]) for r in table.rows], dtype=np.float32)
    thw = np.asarray([float(r["mean_headway_s"]) for r in table.rows], dtype=np.float32)
    return (
        w["acc_interference"] * _z(ia)
        + w["density"] * _z(density)
        + w["obb_occupancy"] * _z(occ)
        + w["speed_ratio"] * _z(sr)
        + w["headway"] * _z(thw)
    )


def smooth_labels_by_dataset(labels: np.ndarray, table: FeatureTable, n_states: int, window: int) -> np.ndarray:
    if window <= 1:
        return labels
    radius = window // 2
    smoothed = labels.copy()
    for dataset in np.unique(table.dataset):
        idx = np.where(table.dataset == dataset)[0]
        order = idx[np.argsort(table.start_s[idx])]
        seq = labels[order]
        out = seq.copy()
        for pos in range(seq.shape[0]):
            left = max(0, pos - radius)
            right = min(seq.shape[0], pos + radius + 1)
            counts = np.bincount(seq[left:right], minlength=n_states)
            if counts.max() > 1:
                out[pos] = int(np.argmax(counts))
        smoothed[order] = out
    return smoothed


def make_state_labels(table: FeatureTable, cfg: dict, main_dataset: str = "xamn6") -> tuple[np.ndarray, np.ndarray]:
    n_states = int(cfg["feature"].get("n_states", 3))
    quantiles = cfg["feature"].get("quantiles", [0.40, 0.75])
    assert len(quantiles) == n_states - 1

    mask = table.dataset == main_dataset
    main_indices = np.where(mask)[0]
    rows = [table.rows[i] for i in main_indices]
    label_method = cfg["feature"].get("label_method", "cluster_vdrf")

    if label_method == "cluster_vdrf":
        label_features = ["speed_ratio", "density_veh_per_m", "lane_change_rate", "direction_fluctuation"]
        x_main = matrix_from_rows(rows, label_features)
        scaler = RobustScaler()
        x_main_z = scaler.fit_transform(x_main)
        kmeans = KMeans(n_clusters=n_states, random_state=int(cfg["experiment"].get("random_seed", 42)), n_init=50)
        main_cluster = kmeans.fit_predict(x_main_z)

        macro_centers = []
        for cluster in range(n_states):
            cluster_rows = [rows[i] for i in np.where(main_cluster == cluster)[0]]
            if not cluster_rows:
                macro_centers.append([0.0, 0.0, 0.0])
                continue
            macro_centers.append([
                float(np.mean([float(r["speed_ratio"]) for r in cluster_rows])),
                float(np.mean([float(r["density_veh_per_m"]) for r in cluster_rows])),
                float(np.mean([float(r.get("hfgo_occupancy", r["obb_occupancy"])) for r in cluster_rows])),
            ])
        macro_center_z = RobustScaler().fit_transform(np.asarray(macro_centers, dtype=np.float32))
        risk = -macro_center_z[:, 0] + macro_center_z[:, 1] + 0.5 * macro_center_z[:, 2]
        order = np.argsort(risk)
        cluster_to_state = {int(cluster): int(state) for state, cluster in enumerate(order)}

        x_all = matrix_from_rows(table.rows, label_features)
        x_all_z = scaler.transform(x_all)
        all_cluster = kmeans.predict(x_all_z)
        cluster_labels = np.asarray([cluster_to_state[int(c)] for c in all_cluster], dtype=np.int64)

        speed_ratio = np.asarray([float(r["speed_ratio"]) for r in rows], dtype=np.float32)
        density = np.asarray([float(r["density_veh_per_m"]) for r in rows], dtype=np.float32)
        occ_key = "hfgo_occupancy" if "hfgo_occupancy" in rows[0] else "obb_occupancy"
        occ = np.asarray([float(r[occ_key]) for r in rows], dtype=np.float32)
        speed_ratio_all = np.asarray([float(r["speed_ratio"]) for r in table.rows], dtype=np.float32)
        density_all = np.asarray([float(r["density_veh_per_m"]) for r in table.rows], dtype=np.float32)
        occ_all = np.asarray([float(r[occ_key]) for r in table.rows], dtype=np.float32)
        score_all = (
            0.65 * _robust_scaled(1.0 - speed_ratio, 1.0 - speed_ratio_all)
            + 0.25 * _robust_scaled(density, density_all)
            + 0.10 * _robust_scaled(occ, occ_all)
        ).astype(np.float32)
        thresholds = np.quantile(score_all[main_indices], quantiles).astype(np.float32)
        labels = np.digitize(score_all, thresholds).astype(np.int64)
        labels = np.clip(labels, 0, n_states - 1)

        def _passes_physical_order(candidate: np.ndarray) -> bool:
            state_speed = []
            state_density = []
            state_occ = []
            for state in range(n_states):
                state_idx = main_indices[candidate[main_indices] == state]
                if state_idx.shape[0] == 0:
                    return False
                state_speed.append(float(np.mean([float(table.rows[i]["mean_speed_kmh"]) for i in state_idx])))
                state_density.append(float(np.mean([float(table.rows[i]["density_veh_per_m"]) for i in state_idx])))
                state_occ.append(float(np.mean([float(table.rows[i]["obb_occupancy"]) for i in state_idx])))
            return (
                all(state_speed[i] >= state_speed[i + 1] - 1e-6 for i in range(n_states - 1))
                and state_speed[-1] < state_speed[0]
                and abs(state_density[-1] - max(state_density)) < 1e-6
                and abs(state_occ[-1] - max(state_occ)) < 1e-6
            )

        if _passes_physical_order(cluster_labels):
            labels = cluster_labels
    else:
        speed_ratio = np.asarray([float(r["speed_ratio"]) for r in rows], dtype=np.float32)
        density = np.asarray([float(r["density_veh_per_m"]) for r in rows], dtype=np.float32)
        occ_key = "hfgo_occupancy" if "hfgo_occupancy" in rows[0] else "obb_occupancy"
        occ = np.asarray([float(r[occ_key]) for r in rows], dtype=np.float32)
        score_main = 0.65 * _z(1.0 - speed_ratio) + 0.25 * _z(density) + 0.10 * _z(occ)
        thresholds = np.quantile(score_main, quantiles)

        speed_ratio_all = np.asarray([float(r["speed_ratio"]) for r in table.rows], dtype=np.float32)
        density_all = np.asarray([float(r["density_veh_per_m"]) for r in table.rows], dtype=np.float32)
        occ_all = np.asarray([float(r[occ_key]) for r in table.rows], dtype=np.float32)
        score_all = (
            0.65 * ((1.0 - speed_ratio_all - (1.0 - speed_ratio).mean()) / ((1.0 - speed_ratio).std() + 1e-6))
            + 0.25 * ((density_all - density.mean()) / (density.std() + 1e-6))
            + 0.10 * ((occ_all - occ.mean()) / (occ.std() + 1e-6))
        )
        labels = np.digitize(score_all, thresholds).astype(np.int64)
        labels = np.clip(labels, 0, n_states - 1)
    labels = smooth_labels_by_dataset(
        labels,
        table,
        n_states,
        int(cfg["feature"].get("label_smoothing_window", 1)),
    )
    table.score = score_all
    table.y = labels
    return labels, thresholds.astype(np.float32)


def make_deterioration_labels(y: np.ndarray, horizon_steps: int) -> np.ndarray:
    n = len(y)
    labels = np.full(n, -1, dtype=np.int64)
    for i in range(n - horizon_steps):
        labels[i] = 1 if int(y[i + horizon_steps]) > int(y[i]) else 0
    return labels


def make_deterioration_labels_score(
    scores: np.ndarray,
    horizon_steps: int,
    threshold_pct: float | None = 0.5,
    std_multiplier: float | None = None,
) -> np.ndarray:
    n = len(scores)
    labels = np.full(n, -1, dtype=np.int64)
    diffs = np.full(n, np.nan, dtype=np.float32)
    for i in range(n - horizon_steps):
        diffs[i] = scores[i + horizon_steps] - scores[i]
    valid = ~np.isnan(diffs)
    if valid.sum() == 0:
        return labels
    valid_diffs = diffs[valid]
    if std_multiplier is not None:
        threshold = float(valid_diffs.mean() + float(std_multiplier) * valid_diffs.std())
    else:
        threshold = float(np.quantile(valid_diffs, threshold_pct if threshold_pct is not None else 0.5))
    for i in range(n - horizon_steps):
        labels[i] = 1 if diffs[i] > threshold else 0
    return labels


def cost_sensitive_operating_point(y_true: np.ndarray, prob_pos: np.ndarray, cost: float = 0.8) -> dict[str, float]:
    thresholds = np.unique(np.concatenate([
        np.asarray([0.5], dtype=np.float32),
        np.quantile(prob_pos, np.linspace(0.05, 0.95, 37)).astype(np.float32),
    ]))
    if thresholds.size == 0:
        pred = (prob_pos >= 0.5).astype(np.int64)
        m = metrics_dict(y_true, pred, 2)
        return {"threshold": 0.5, **m}
    best_payload: dict[str, float] | None = None
    for threshold in thresholds:
        pred = (prob_pos >= float(threshold)).astype(np.int64)
        cm = confusion_matrix_np(y_true, pred, 2).astype(np.float32)
        tn = float(cm[0, 0])
        fp = float(cm[0, 1])
        fn = float(cm[1, 0])
        tp = float(cm[1, 1])
        precision_pos = tp / max(1e-6, tp + fp)
        recall_pos = tp / max(1e-6, tp + fn)
        m = metrics_dict(y_true, pred, 2)
        score = m["f1_macro"] + 0.05 * recall_pos - 0.05 * float(cost) * (1.0 - precision_pos)
        payload = {
            "threshold": float(threshold),
            "score": float(score),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "precision_positive": float(precision_pos),
            "recall_positive": float(recall_pos),
            **m,
        }
        if best_payload is None or payload["score"] > best_payload["score"]:
            best_payload = payload
    assert best_payload is not None
    return best_payload


def compute_sample_weights(y_train: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    total = float(len(y_train))
    weights = np.zeros(n_classes, dtype=np.float32)
    for c in range(n_classes):
        weights[c] = total / (n_classes * max(1.0, float(counts[c])))
    return weights[y_train]


def time_split_indices(mask: np.ndarray, test_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    idx = np.where(mask)[0]
    n_test = max(1, int(round(idx.shape[0] * test_ratio)))
    split = idx.shape[0] - n_test
    return idx[:split], idx[split:]


def stratified_split_indices(mask: np.ndarray, y: np.ndarray, test_ratio: float, seed: int, n_classes: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if n_classes <= 0:
        n_classes = int(y.max()) + 1
    rng = np.random.default_rng(seed)
    scoped_idx = np.where(mask)[0]
    train_parts = []
    test_parts = []
    for cls in range(n_classes):
        cls_idx = scoped_idx[y[scoped_idx] == cls]
        if cls_idx.size == 0:
            continue
        shuffled = cls_idx.copy()
        rng.shuffle(shuffled)
        n_test = max(1, int(round(cls_idx.size * test_ratio))) if cls_idx.size > 1 else 1
        test_parts.append(shuffled[:n_test])
        train_parts.append(shuffled[n_test:])
    train = np.sort(np.concatenate(train_parts)) if train_parts else np.asarray([], dtype=np.int64)
    test = np.sort(np.concatenate(test_parts)) if test_parts else np.asarray([], dtype=np.int64)
    return train, test


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0) + 1e-6
    return (train - mean) / std, (test - mean) / std, mean, std


def temporal_context_features(
    base: np.ndarray,
    score: np.ndarray | None = None,
    state: np.ndarray | None = None,
    lags: tuple[int, ...] = (1, 3, 5),
    windows: tuple[int, ...] = (3, 5, 8),
) -> np.ndarray:
    """Build causal lag, delta, and rolling-trend features for one ordered sequence."""
    parts = [base.astype(np.float32)]
    if score is not None:
        parts.append(score.reshape(-1, 1).astype(np.float32))
    if state is not None:
        parts.append(state.reshape(-1, 1).astype(np.float32))

    for lag in lags:
        prev = np.vstack([np.repeat(base[:1], lag, axis=0), base[:-lag]])
        parts.append(prev.astype(np.float32))
        parts.append((base - prev).astype(np.float32))
        if score is not None:
            prev_score = np.concatenate([np.repeat(score[:1], lag), score[:-lag]]).reshape(-1, 1)
            score_col = score.reshape(-1, 1)
            parts.append(prev_score.astype(np.float32))
            parts.append((score_col - prev_score).astype(np.float32))

    for window in windows:
        means = []
        stds = []
        slopes = []
        for i in range(base.shape[0]):
            start = max(0, i - window + 1)
            vals = base[start : i + 1]
            means.append(vals.mean(axis=0))
            stds.append(vals.std(axis=0))
            slopes.append(vals[-1] - vals[0])
        parts.append(np.asarray(means, dtype=np.float32))
        parts.append(np.asarray(stds, dtype=np.float32))
        parts.append(np.asarray(slopes, dtype=np.float32))

        if score is not None:
            score_means = []
            score_stds = []
            score_slopes = []
            for i in range(score.shape[0]):
                start = max(0, i - window + 1)
                vals = score[start : i + 1]
                score_means.append(float(vals.mean()))
                score_stds.append(float(vals.std()))
                score_slopes.append(float(vals[-1] - vals[0]))
            parts.append(np.asarray(score_means, dtype=np.float32).reshape(-1, 1))
            parts.append(np.asarray(score_stds, dtype=np.float32).reshape(-1, 1))
            parts.append(np.asarray(score_slopes, dtype=np.float32).reshape(-1, 1))

    return np.hstack(parts).astype(np.float32)


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int | None = None) -> dict[str, float]:
    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max())) + 1
    cm = confusion_matrix_np(y_true, y_pred, num_classes)
    precision = []
    recall = []
    f1 = []
    supports = []
    for i in range(num_classes):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - cm[i, i])
        fn = float(cm[i, :].sum() - cm[i, i])
        support = float(cm[i, :].sum())
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        score = 2 * p * r / (p + r) if p + r > 0 else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(score)
        supports.append(support)
    supports_arr = np.asarray(supports, dtype=np.float64)
    f1_arr = np.asarray(f1, dtype=np.float64)
    weighted = float((f1_arr * supports_arr).sum() / max(1.0, supports_arr.sum()))
    return {
        "accuracy": float((y_true == y_pred).mean()) if y_true.size else 0.0,
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
        "f1_weighted": weighted,
    }


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        if 0 <= int(true) < num_classes and 0 <= int(pred) < num_classes:
            cm[int(true), int(pred)] += 1
    return cm


def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int | None = None) -> float:
    if num_classes is None:
        num_classes = int(y_true.max()) + 1
    if num_classes == 2:
        return float(roc_auc_score(y_true, y_prob[:, 1]))
    return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))


def tree_shap_summary(model: XGBClassifier, x_eval: np.ndarray, feature_names: list[str], class_names: list[str]) -> dict[str, object]:
    try:
        contrib = model.get_booster().predict(DMatrix(x_eval), pred_contribs=True)
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc)}
    arr = np.asarray(contrib, dtype=np.float32)
    if arr.ndim == 3:
        # XGBoost returns [n, classes, features+1] for multi-class TreeSHAP.
        arr = arr[:, :, :-1]
    elif arr.ndim == 2:
        arr = arr[:, :-1][:, None, :]
    else:
        return {"status": "unavailable", "reason": f"Unexpected contribution shape {arr.shape}"}
    mean_abs = np.mean(np.abs(arr), axis=0)
    per_class = []
    for class_idx in range(mean_abs.shape[0]):
        order = np.argsort(mean_abs[class_idx])[::-1][:8]
        per_class.append({
            "class": class_names[class_idx] if class_idx < len(class_names) else str(class_idx),
            "top_features": [
                {"feature": feature_names[int(i)], "mean_abs_shap": float(mean_abs[class_idx, int(i)])}
                for i in order
            ],
        })
    global_importance = np.mean(mean_abs, axis=0)
    order = np.argsort(global_importance)[::-1][:12]
    return {
        "status": "ok",
        "method": "xgboost_pred_contribs_treeshap",
        "global_top_features": [
            {"feature": feature_names[int(i)], "mean_abs_shap": float(global_importance[int(i)])}
            for i in order
        ],
        "per_class": per_class,
    }


def conformal_prediction_summary(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    params: dict[str, float | int],
    seed: int,
    n_states: int,
    alpha: float = 0.10,
) -> dict[str, object]:
    cal_train_idx, cal_idx = stratified_split_indices(np.ones(y_train.shape[0], dtype=bool), y_train, 0.20, seed, n_states)
    if cal_idx.size < n_states or cal_train_idx.size < n_states:
        return {"status": "skipped", "reason": "calibration split too small"}
    model = xgb_model(seed, params, n_states)
    fit_xgb_multiclass(
        model,
        x_train[cal_train_idx],
        y_train[cal_train_idx],
        n_states,
        sample_weight=compute_sample_weights(y_train[cal_train_idx], n_states),
    )
    cal_prob = model.predict_proba(x_train[cal_idx])
    cal_scores = 1.0 - cal_prob[np.arange(cal_idx.shape[0]), y_train[cal_idx]]
    q_level = min(1.0, math.ceil((cal_scores.shape[0] + 1) * (1.0 - alpha)) / max(1, cal_scores.shape[0]))
    threshold = float(np.quantile(cal_scores, q_level, method="higher"))
    test_prob = model.predict_proba(x_test)
    prediction_sets = (1.0 - test_prob) <= threshold
    empty = np.where(prediction_sets.sum(axis=1) == 0)[0]
    if empty.size:
        prediction_sets[empty, np.argmax(test_prob[empty], axis=1)] = True
    covered = prediction_sets[np.arange(y_test.shape[0]), y_test]
    set_sizes = prediction_sets.sum(axis=1)
    return {
        "status": "completed",
        "alpha": alpha,
        "confidence": 1.0 - alpha,
        "calibration_size": int(cal_idx.shape[0]),
        "threshold": threshold,
        "coverage": float(np.mean(covered)),
        "average_set_size": float(np.mean(set_sizes)),
        "singleton_rate": float(np.mean(set_sizes == 1)),
        "set_size_histogram": {str(i): int(np.sum(set_sizes == i)) for i in range(1, n_states + 1)},
    }


def conformal_prediction_sweep(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    params: dict[str, float | int],
    seed: int,
    n_states: int,
    alphas: list[float] | None = None,
) -> list[dict[str, object]]:
    alpha_values = alphas or [0.30, 0.20, 0.10, 0.05, 0.01]
    return [
        conformal_prediction_summary(x_train, y_train, x_test, y_test, params, seed, n_states, alpha=float(alpha))
        for alpha in alpha_values
    ]


def shap_counterfactual_analysis(
    model: XGBClassifier,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    shap_summary: dict[str, object],
    n_states: int,
) -> dict[str, object]:
    if not shap_summary or shap_summary.get("status") != "ok":
        return {"status": "skipped", "reason": "TreeSHAP summary unavailable"}
    prob = model.predict_proba(x_test)
    pred = np.argmax(prob, axis=1)
    confidence = np.max(prob, axis=1)
    hard = np.where((pred != y_test) | (confidence < 0.65))[0]
    if hard.size == 0:
        hard = np.argsort(confidence)[:3]
    top_features = [item["feature"] for item in shap_summary.get("global_top_features", [])[:3]]
    if not top_features:
        return {"status": "skipped", "reason": "no top features"}
    grid_q = np.asarray([0.05, 0.25, 0.50, 0.75, 0.95], dtype=np.float32)
    cases = []
    for local_idx in hard[:3]:
        case = {
            "test_index": int(local_idx),
            "true_class": int(y_test[local_idx]),
            "pred_class": int(pred[local_idx]),
            "pred_confidence": float(confidence[local_idx]),
            "features": [],
        }
        base = x_test[local_idx].copy()
        for feature in top_features:
            if feature not in feature_names:
                continue
            feat_idx = feature_names.index(feature)
            values = np.quantile(x_train[:, feat_idx], grid_q)
            curve = []
            for value in values:
                perturbed = base.copy()
                perturbed[feat_idx] = value
                p = model.predict_proba(perturbed.reshape(1, -1))[0]
                curve.append({
                    "feature_value_standardized": float(value),
                    "true_class_prob": float(p[int(y_test[local_idx])]),
                    "pred_class_prob": float(p[int(pred[local_idx])]),
                })
            case["features"].append({"feature": feature, "curve": curve})
        cases.append(case)
    return {"status": "completed", "method": "one_feature_quantile_perturbation", "cases": cases}


def class_predictions(pred: np.ndarray) -> np.ndarray:
    arr = np.asarray(pred)
    if arr.ndim > 1:
        return np.argmax(arr, axis=1).astype(np.int64)
    return arr.astype(np.int64)


def xgb_model(seed: int, params: dict[str, float | int], num_classes: int = 3) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=seed,
        tree_method="hist",
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        min_child_weight=int(params.get("min_child_weight", 1)),
        gamma=float(params.get("gamma", 0.0)),
        reg_alpha=float(params.get("reg_alpha", 0.0)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
    )


def fit_xgb_multiclass(
    model: XGBClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    sample_weight: np.ndarray | None = None,
) -> None:
    present = set(int(v) for v in np.unique(y_train))
    missing = [c for c in range(num_classes) if c not in present]
    if not missing:
        model.fit(x_train, y_train, sample_weight=sample_weight)
        return
    filler_x = np.repeat(x_train[:1], len(missing), axis=0)
    filler_y = np.asarray(missing, dtype=np.int64)
    x_aug = np.vstack([x_train, filler_x])
    y_aug = np.concatenate([y_train, filler_y])
    if sample_weight is None:
        w_aug = np.concatenate([np.ones(y_train.shape[0], dtype=np.float32), np.full(len(missing), 1e-6, dtype=np.float32)])
    else:
        w_aug = np.concatenate([sample_weight, np.full(len(missing), 1e-6, dtype=np.float32)])
    model.fit(x_aug, y_aug, sample_weight=w_aug)


def xgb_binary_model(seed: int, params: dict[str, float | int], scale_pos_weight: float = 1.0) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        tree_method="hist",
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        scale_pos_weight=scale_pos_weight,
        min_child_weight=int(params.get("min_child_weight", 1)),
        gamma=float(params.get("gamma", 0.0)),
        reg_alpha=float(params.get("reg_alpha", 0.0)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
    )


class LinearClassifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_torch_linear(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int, num_classes: int = 3) -> np.ndarray:
    torch.manual_seed(seed)
    model = LinearClassifier(x_train.shape[1], num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    tx = torch.tensor(x_train, dtype=torch.float32)
    ty = torch.tensor(y_train, dtype=torch.long)
    for _ in range(180):
        logits = model(tx)
        loss = loss_fn(logits, ty)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(torch.tensor(x_test, dtype=torch.float32)).argmax(dim=1).cpu().numpy()
    return pred


def run_classification(table: FeatureTable, cfg: dict) -> dict[str, object]:
    seed = int(cfg["experiment"]["random_seed"])
    test_ratio = float(cfg["experiment"]["test_ratio"])
    xgb_params = cfg["experiment"]["xgboost"]
    n_states = int(cfg["feature"].get("n_states", 3))
    primary_split = cfg["feature"].get("primary_split", "time_series")
    main_mask = table.dataset == "xamn6"
    y = table.y
    assert y is not None

    if primary_split == "time_series":
        train_idx, test_idx = time_split_indices(main_mask, test_ratio)
    else:
        train_idx, test_idx = stratified_split_indices(main_mask, y, test_ratio, seed, n_states)

    results: dict[str, object] = {}
    sw_train = compute_sample_weights(y[train_idx], n_states)

    def fit_eval(name: str, model, x: np.ndarray, feature_names: list[str] | None = None) -> None:
        x_train, x_test, _, _ = standardize(x[train_idx], x[test_idx])
        fit_xgb_multiclass(model, x_train, y[train_idx], n_states, sample_weight=sw_train)
        pred = class_predictions(model.predict(x_test))
        item: dict[str, object] = {
            "metrics": metrics_dict(y[test_idx], pred, n_states),
            "confusion_matrix": confusion_matrix_np(y[test_idx], pred, n_states).tolist(),
            "pred": pred.tolist(),
            "true": y[test_idx].tolist(),
        }
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(x_test)
            try:
                item["roc_auc"] = compute_roc_auc(y[test_idx], prob, n_states)
            except ValueError:
                item["roc_auc"] = None
        if feature_names is not None and hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_, dtype=np.float32)
            order = np.argsort(importances)[::-1][:12]
            item["feature_importance"] = [
                {"feature": feature_names[int(i)], "importance": float(importances[int(i)])}
                for i in order
            ]
        results[name] = item

    majority = int(np.bincount(y[train_idx], minlength=n_states).argmax())
    majority_pred = np.full_like(y[test_idx], majority)
    results["Majority"] = {
        "metrics": metrics_dict(y[test_idx], majority_pred, n_states),
        "confusion_matrix": confusion_matrix_np(y[test_idx], majority_pred, n_states).tolist(),
        "pred": majority_pred.tolist(),
        "true": y[test_idx].tolist(),
    }
    x_train_lin, x_test_lin, _, _ = standardize(table.x_obb[train_idx], table.x_obb[test_idx])
    lin_pred = fit_torch_linear(x_train_lin, y[train_idx], x_test_lin, seed, n_states)
    results["TorchLinear-OBB"] = {
        "metrics": metrics_dict(y[test_idx], lin_pred, n_states),
        "confusion_matrix": confusion_matrix_np(y[test_idx], lin_pred, n_states).tolist(),
        "pred": lin_pred.tolist(),
        "true": y[test_idx].tolist(),
    }
    fit_eval("XGBoost-HBB", xgb_model(seed, xgb_params, n_states), table.x_hbb, NUMERIC_FEATURES_HBB)
    fit_eval("XGBoost-OBB", xgb_model(seed, xgb_params, n_states), table.x_obb, NUMERIC_FEATURES_OBB)

    # SVM baseline (OBB features)
    x_obb_train, x_obb_test, _, _ = standardize(table.x_obb[train_idx], table.x_obb[test_idx])
    x_obb_train_64 = np.ascontiguousarray(x_obb_train, dtype=np.float64)
    x_obb_test_64 = np.ascontiguousarray(x_obb_test, dtype=np.float64)
    sw_64 = np.ascontiguousarray(sw_train, dtype=np.float64)
    svm = SVC(kernel="rbf", C=1.0, probability=True, random_state=seed)
    svm.fit(x_obb_train_64, y[train_idx], sample_weight=sw_64)
    svm_pred = class_predictions(svm.predict(x_obb_test_64))
    svm_prob = svm.predict_proba(x_obb_test_64)
    results["SVM-OBB"] = {
        "metrics": metrics_dict(y[test_idx], svm_pred, n_states),
        "confusion_matrix": confusion_matrix_np(y[test_idx], svm_pred, n_states).tolist(),
        "pred": svm_pred.tolist(),
        "true": y[test_idx].tolist(),
    }
    try:
        results["SVM-OBB"]["roc_auc"] = compute_roc_auc(y[test_idx], svm_prob, n_states)
    except ValueError:
        pass

    # Logistic Regression baseline (OBB features)
    lr = LogisticRegression(max_iter=500, C=1.0, random_state=seed, multi_class="multinomial")
    lr.fit(x_obb_train_64, y[train_idx], sample_weight=sw_64)
    lr_pred = class_predictions(lr.predict(x_obb_test_64))
    lr_prob = lr.predict_proba(x_obb_test_64)
    results["LR-OBB"] = {
        "metrics": metrics_dict(y[test_idx], lr_pred, n_states),
        "confusion_matrix": confusion_matrix_np(y[test_idx], lr_pred, n_states).tolist(),
        "pred": lr_pred.tolist(),
        "true": y[test_idx].tolist(),
    }
    try:
        results["LR-OBB"]["roc_auc"] = compute_roc_auc(y[test_idx], lr_prob, n_states)
    except ValueError:
        pass

    x_obb_train_full, x_obb_test_full, _, _ = standardize(table.x_obb[train_idx], table.x_obb[test_idx])
    shap_model = xgb_model(seed, xgb_params, n_states)
    fit_xgb_multiclass(shap_model, x_obb_train_full, y[train_idx], n_states, sample_weight=sw_train)
    shap_summary = tree_shap_summary(
        shap_model,
        x_obb_test_full,
        NUMERIC_FEATURES_OBB,
        STATE_NAMES if n_states <= len(STATE_NAMES) else [str(i) for i in range(n_states)],
    )
    results["XGBoost-OBB"]["shap_summary"] = shap_summary
    results["XGBoost-OBB"]["counterfactual_analysis"] = shap_counterfactual_analysis(
        shap_model,
        x_obb_train_full,
        x_obb_test_full,
        y[test_idx],
        NUMERIC_FEATURES_OBB,
        shap_summary,
        n_states,
    )
    results["XGBoost-OBB"]["conformal_prediction"] = conformal_prediction_summary(
        x_obb_train_full,
        y[train_idx],
        x_obb_test_full,
        y[test_idx],
        xgb_params,
        seed,
        n_states,
        alpha=0.10,
    )
    results["XGBoost-OBB"]["conformal_sweep"] = conformal_prediction_sweep(
        x_obb_train_full,
        y[train_idx],
        x_obb_test_full,
        y[test_idx],
        xgb_params,
        seed,
        n_states,
    )
    results["test_support"] = {STATE_NAMES[i] if i < len(STATE_NAMES) else str(i): int(np.sum(y[test_idx] == i)) for i in range(n_states)}
    results["test_indices"] = test_idx.tolist()
    results["train_indices"] = train_idx.tolist()
    results["split_method"] = primary_split

    # Supplementary stratified result
    if primary_split == "time_series":
        strat_train, strat_test = stratified_split_indices(main_mask, y, test_ratio, seed, n_states)
        x_tr, x_te, _, _ = standardize(table.x_obb[strat_train], table.x_obb[strat_test])
        m = xgb_model(seed, xgb_params, n_states)
        strat_sw = compute_sample_weights(y[strat_train], n_states)
        fit_xgb_multiclass(m, x_tr, y[strat_train], n_states, sample_weight=strat_sw)
        strat_pred = class_predictions(m.predict(x_te))
        results["stratified_supplementary"] = {
            "metrics": metrics_dict(y[strat_test], strat_pred, n_states),
            "note": "Random stratified split (not time-respecting). Shown for reference only.",
        }

    # Supplementary time-series result
    if primary_split == "stratified":
        ts_train, ts_test = time_split_indices(main_mask, test_ratio)
        x_tr, x_te, _, _ = standardize(table.x_obb[ts_train], table.x_obb[ts_test])
        m = xgb_model(seed, xgb_params, n_states)
        ts_sw = compute_sample_weights(y[ts_train], n_states)
        fit_xgb_multiclass(m, x_tr, y[ts_train], n_states, sample_weight=ts_sw)
        ts_pred = class_predictions(m.predict(x_te))
        results["time_series_supplementary"] = {
            "metrics": metrics_dict(y[ts_test], ts_pred, n_states),
            "note": f"Time-series split (last {test_ratio:.0%} as test). Shown for temporal generalization reference.",
        }
        main_idx = np.where(main_mask)[0]
        y_main = y[main_idx]
        x_temporal_main = temporal_context_features(table.x_obb[main_idx])
        rel_train = np.arange(0, ts_train.shape[0])
        rel_test = np.arange(ts_train.shape[0], main_idx.shape[0])
        x_tr_raw = x_temporal_main[rel_train]
        x_te_raw = x_temporal_main[rel_test]
        x_tr, x_te, _, _ = standardize(x_tr_raw, x_te_raw)
        m = xgb_model(seed, xgb_params, n_states)
        ts_sw = compute_sample_weights(y_main[rel_train], n_states)
        fit_xgb_multiclass(m, x_tr, y_main[rel_train], n_states, sample_weight=ts_sw)
        ts_temporal_pred = class_predictions(m.predict(x_te))
        results["time_series_temporal_supplementary"] = {
            "metrics": metrics_dict(y_main[rel_test], ts_temporal_pred, n_states),
            "confusion_matrix": confusion_matrix_np(y_main[rel_test], ts_temporal_pred, n_states).tolist(),
            "pred": ts_temporal_pred.tolist(),
            "true": y_main[rel_test].tolist(),
            "note": "Time-series split with causal lag/delta/rolling trend features.",
        }

    return results


def run_ablation(table: FeatureTable, cfg: dict) -> dict[str, object]:
    seed = int(cfg["experiment"]["random_seed"])
    xgb_params = dict(cfg["experiment"]["xgboost"])
    n_states = int(cfg["feature"].get("n_states", 3))
    y = table.y
    assert y is not None
    main_mask = table.dataset == "xamn6"
    idx = np.where(main_mask)[0]
    y_main = y[idx]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    results = {}
    for name, features in ABLATION_FEATURE_SETS.items():
        x = matrix_from_rows(table.rows, features)
        x_main = x[idx]

        fold_metrics = []
        for train_pos, test_pos in skf.split(x_main, y_main):
            train_fold_idx = idx[train_pos]
            test_fold_idx = idx[test_pos]
            x_train, x_test, _, _ = standardize(x[train_fold_idx], x[test_fold_idx])
            sw = compute_sample_weights(y[train_fold_idx], n_states)
            model = xgb_model(seed, xgb_params, n_states)
            fit_xgb_multiclass(model, x_train, y[train_fold_idx], n_states, sample_weight=sw)
            pred = class_predictions(model.predict(x_test))
            fold_metrics.append(metrics_dict(y[test_fold_idx], pred, n_states))

        f1_macros = [m["f1_macro"] for m in fold_metrics]
        f1_weighteds = [m["f1_weighted"] for m in fold_metrics]
        accs = [m["accuracy"] for m in fold_metrics]
        results[name] = {
            "features": features,
            "metrics": {
                "f1_macro": float(np.mean(f1_macros)),
                "f1_macro_std": float(np.std(f1_macros)),
                "f1_weighted": float(np.mean(f1_weighteds)),
                "f1_weighted_std": float(np.std(f1_weighteds)),
                "accuracy": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "precision_macro": float(np.mean([m["precision_macro"] for m in fold_metrics])),
                "recall_macro": float(np.mean([m["recall_macro"] for m in fold_metrics])),
            },
            "fold_details": fold_metrics,
        }
    if "M4: Ours+headway+acc+MGTI" in results and "M3': V+D+F" in results:
        m4 = np.asarray([m["f1_macro"] for m in results["M4: Ours+headway+acc+MGTI"]["fold_details"]], dtype=np.float32)
        m3f = np.asarray([m["f1_macro"] for m in results["M3': V+D+F"]["fold_details"]], dtype=np.float32)
        test = ttest_rel(m4, m3f)
        m4_std = float(np.std(m4))
        m3f_std = float(np.std(m3f))
        results["M4: Ours+headway+acc+MGTI"]["stability_vs_m3f"] = {
            "comparison": "M4 vs M3': V+D+F",
            "mean_delta": float(np.mean(m4 - m3f)),
            "m4_std": m4_std,
            "m3f_std": m3f_std,
            "std_reduction_rate": float((m3f_std - m4_std) / max(1e-9, m3f_std)),
            "paired_t_statistic": float(test.statistic) if np.isfinite(test.statistic) else None,
            "paired_t_pvalue": float(test.pvalue) if np.isfinite(test.pvalue) else None,
        }
        method_names = list(results.keys())
        pvalues: dict[str, dict[str, float | None]] = {}
        mean_deltas: dict[str, dict[str, float]] = {}
        fold_arrays = {
            name: np.asarray([m["f1_macro"] for m in item["fold_details"]], dtype=np.float32)
            for name, item in results.items()
        }
        for left in method_names:
            pvalues[left] = {}
            mean_deltas[left] = {}
            for right in method_names:
                if left == right:
                    pvalues[left][right] = 1.0
                    mean_deltas[left][right] = 0.0
                    continue
                test_pair = ttest_rel(fold_arrays[left], fold_arrays[right])
                pvalues[left][right] = float(test_pair.pvalue) if np.isfinite(test_pair.pvalue) else None
                mean_deltas[left][right] = float(np.mean(fold_arrays[left] - fold_arrays[right]))
        results["M4: Ours+headway+acc+MGTI"]["paired_t_test_matrix"] = {
            "methods": method_names,
            "pvalues": pvalues,
            "mean_deltas": mean_deltas,
            "metric": "fold_macro_f1",
        }
    return results


def run_parameter_sensitivity(table: FeatureTable, cfg: dict) -> dict[str, object]:
    seed = int(cfg["experiment"]["random_seed"])
    test_ratio = float(cfg["experiment"]["test_ratio"])
    n_states = int(cfg["feature"].get("n_states", 3))
    y = table.y
    assert y is not None
    main_mask = table.dataset == "xamn6"
    idx = np.where(main_mask)[0]
    y_main = y[idx]
    x_main = table.x_obb[idx]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    depth_results = []
    for depth in [2, 3, 4, 5, 6]:
        params = dict(cfg["experiment"]["xgboost"])
        params["max_depth"] = depth
        fold_f1s = []
        fold_accs = []
        for train_pos, test_pos in skf.split(x_main, y_main):
            x_train, x_test, _, _ = standardize(x_main[train_pos], x_main[test_pos])
            sw = compute_sample_weights(y_main[train_pos], n_states)
            model = xgb_model(seed, params, n_states)
            fit_xgb_multiclass(model, x_train, y_main[train_pos], n_states, sample_weight=sw)
            pred = class_predictions(model.predict(x_test))
            m = metrics_dict(y_main[test_pos], pred, n_states)
            fold_f1s.append(m["f1_macro"])
            fold_accs.append(m["accuracy"])
        depth_results.append({
            "max_depth": depth,
            "metrics": {
                "f1_macro": float(np.mean(fold_f1s)),
                "f1_macro_std": float(np.std(fold_f1s)),
                "accuracy": float(np.mean(fold_accs)),
            },
        })

    horizon_results = []
    x = table.x_obb[idx]
    split = int(round(x.shape[0] * (1.0 - test_ratio)))
    x_train_raw = x[:split]
    x_test_raw = x[split:]
    _, _, mean, std = standardize(x_train_raw, x_test_raw)
    x_scaled = (x - mean) / std
    for horizon in [1, 3, 5, 8]:
        valid_end = x.shape[0] - horizon
        train_end = min(split, valid_end)
        test_positions = np.arange(split, valid_end)
        if test_positions.size < 5:
            continue
        model = xgb_model(seed, cfg["experiment"]["xgboost"], n_states)
        fit_xgb_multiclass(model, x_scaled[:train_end], y_main[horizon : train_end + horizon], n_states)
        pred = class_predictions(model.predict(x_scaled[test_positions]))
        horizon_results.append(
            {
                "horizon_steps": horizon,
                "horizon_seconds": float(horizon * float(cfg["feature"]["step_s"])),
                "metrics": metrics_dict(y_main[test_positions + horizon], pred, n_states),
            }
        )
    return {"max_depth": depth_results, "prediction_horizon": horizon_results}


def run_time_series_cv(table: FeatureTable, cfg: dict, folds: int = 5) -> dict[str, object]:
    seed = int(cfg["experiment"]["random_seed"])
    xgb_params = cfg["experiment"]["xgboost"]
    n_states = int(cfg["feature"].get("n_states", 3))
    y = table.y
    assert y is not None
    idx = np.where(table.dataset == "xamn6")[0]
    fold_size = max(8, idx.shape[0] // (folds + 1))
    rows = []
    for fold in range(folds):
        train_end = fold_size * (fold + 1)
        test_start = train_end
        test_end = min(test_start + fold_size, idx.shape[0])
        if test_end - test_start < 8:
            continue
        train_idx = idx[:train_end]
        test_idx = idx[test_start:test_end]
        x_train, x_test, _, _ = standardize(table.x_obb[train_idx], table.x_obb[test_idx])
        model = xgb_model(seed + fold, xgb_params, n_states)
        fit_xgb_multiclass(model, x_train, y[train_idx], n_states)
        pred = class_predictions(model.predict(x_test))
        fold_metrics = metrics_dict(y[test_idx], pred, n_states)
        fold_item: dict[str, object] = {
            "fold": fold + 1,
            "train_windows": int(train_idx.shape[0]),
            "test_windows": int(test_idx.shape[0]),
            "test_support": {STATE_NAMES[i] if i < len(STATE_NAMES) else str(i): int(np.sum(y[test_idx] == i)) for i in range(n_states)},
            "metrics": fold_metrics,
        }
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(x_test)
            try:
                fold_item["roc_auc"] = compute_roc_auc(y[test_idx], prob, n_states)
            except ValueError:
                pass
        rows.append(fold_item)
    if rows:
        macro_f1 = np.asarray([row["metrics"]["f1_macro"] for row in rows], dtype=np.float32)
        accuracy = np.asarray([row["metrics"]["accuracy"] for row in rows], dtype=np.float32)
        summary = {
            "folds": len(rows),
            "accuracy_mean": float(np.mean(accuracy)),
            "accuracy_std": float(np.std(accuracy)),
            "f1_macro_mean": float(np.mean(macro_f1)),
            "f1_macro_std": float(np.std(macro_f1)),
        }
    else:
        summary = {"folds": 0, "accuracy_mean": 0.0, "accuracy_std": 0.0, "f1_macro_mean": 0.0, "f1_macro_std": 0.0}
    return {"method": "expanding_time_series_cv", "model": "XGBoost-OBB", "summary": summary, "folds": rows}


def _summary_from_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "accuracy_mean": 0.0,
            "accuracy_std": 0.0,
            "f1_macro_mean": 0.0,
            "f1_macro_std": 0.0,
        }
    acc = np.asarray([row["accuracy"] for row in rows], dtype=np.float32)
    f1 = np.asarray([row["f1_macro"] for row in rows], dtype=np.float32)
    return {
        "accuracy_mean": float(np.mean(acc)),
        "accuracy_std": float(np.std(acc)),
        "f1_macro_mean": float(np.mean(f1)),
        "f1_macro_std": float(np.std(f1)),
    }


def run_robustness(table: FeatureTable, cfg: dict) -> dict[str, object]:
    seeds = [7, 21, 42, 84, 2026]
    test_ratio = float(cfg["experiment"]["test_ratio"])
    xgb_params = cfg["experiment"]["xgboost"]
    n_states = int(cfg["feature"].get("n_states", 3))
    horizon = int(cfg["feature"]["prediction_horizon_steps"])
    y = table.y
    assert y is not None

    main_mask = table.dataset == "xamn6"
    idx = np.where(main_mask)[0]
    y_main = y[idx]
    x_main = table.x_obb[idx]
    split = int(round(x_main.shape[0] * (1.0 - test_ratio)))
    valid_end = x_main.shape[0] - horizon

    cls_rows: dict[str, list[dict[str, float]]] = {
        "XGBoost-HBB": [],
        "XGBoost-OBB": [],
        "LR-OBB": [],
    }
    pred_rows: dict[str, list[dict[str, float]]] = {"XGBoost-future": []}

    for seed in seeds:
        train_idx, test_idx = stratified_split_indices(main_mask, y, test_ratio, seed, n_states)
        sw_train = compute_sample_weights(y[train_idx], n_states)

        x_hbb_train, x_hbb_test, _, _ = standardize(table.x_hbb[train_idx], table.x_hbb[test_idx])
        hbb_model = xgb_model(seed, xgb_params, n_states)
        fit_xgb_multiclass(hbb_model, x_hbb_train, y[train_idx], n_states, sample_weight=sw_train)
        hbb_pred = class_predictions(hbb_model.predict(x_hbb_test))
        cls_rows["XGBoost-HBB"].append(metrics_dict(y[test_idx], hbb_pred, n_states))

        x_obb_train, x_obb_test, _, _ = standardize(table.x_obb[train_idx], table.x_obb[test_idx])
        obb_model = xgb_model(seed, xgb_params, n_states)
        fit_xgb_multiclass(obb_model, x_obb_train, y[train_idx], n_states, sample_weight=sw_train)
        obb_pred = class_predictions(obb_model.predict(x_obb_test))
        cls_rows["XGBoost-OBB"].append(metrics_dict(y[test_idx], obb_pred, n_states))

        lr = LogisticRegression(max_iter=500, C=1.0, random_state=seed, multi_class="multinomial")
        lr.fit(np.ascontiguousarray(x_obb_train, dtype=np.float64), y[train_idx], sample_weight=np.ascontiguousarray(sw_train, dtype=np.float64))
        lr_pred = class_predictions(lr.predict(np.ascontiguousarray(x_obb_test, dtype=np.float64)))
        cls_rows["LR-OBB"].append(metrics_dict(y[test_idx], lr_pred, n_states))

        if valid_end > split + 5:
            _, _, mean, std = standardize(x_main[:split], x_main[split:])
            x_scaled = (x_main - mean) / std
            train_end = min(split, valid_end)
            train_y = y_main[horizon : train_end + horizon]
            future_model = xgb_model(seed, xgb_params, n_states)
            fit_xgb_multiclass(
                future_model,
                x_scaled[:train_end],
                train_y,
                n_states,
                sample_weight=compute_sample_weights(train_y, n_states),
            )
            test_positions = np.arange(split, valid_end)
            future_pred = class_predictions(future_model.predict(x_scaled[test_positions]))
            pred_rows["XGBoost-future"].append(metrics_dict(y_main[test_positions + horizon], future_pred, n_states))

    return {
        "seeds": seeds,
        "classification": {
            name: {**_summary_from_metric_rows(rows), "runs": rows}
            for name, rows in cls_rows.items()
        },
        "prediction": {
            name: {**_summary_from_metric_rows(rows), "runs": rows}
            for name, rows in pred_rows.items()
        },
    }


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def sequence_dataset(x: np.ndarray, y: np.ndarray, seq_len: int, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = []
    ys = []
    end_positions = []
    for end in range(seq_len - 1, x.shape[0] - horizon):
        xs.append(x[end - seq_len + 1 : end + 1])
        ys.append(y[end + horizon])
        end_positions.append(end)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), np.asarray(end_positions, dtype=np.int64)


def stratified_future_positions(
    y: np.ndarray,
    horizon: int,
    test_ratio: float,
    seed: int,
    n_states: int,
    embargo_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    valid_positions = np.arange(0, y.shape[0] - horizon, dtype=np.int64)
    target = y[valid_positions + horizon]
    test_parts = []
    for state in range(n_states):
        state_pos = valid_positions[target == state]
        if state_pos.size == 0:
            continue
        shuffled = state_pos.copy()
        rng.shuffle(shuffled)
        n_test = max(1, int(round(shuffled.size * test_ratio)))
        if shuffled.size >= 10:
            n_test = 5 if embargo_steps is not None else max(5, n_test)
        n_test = min(shuffled.size - 1, n_test) if shuffled.size > 1 else 1
        test_parts.append(shuffled[:n_test])
    test = np.sort(np.concatenate([p for p in test_parts if p.size])) if test_parts else np.asarray([], dtype=np.int64)
    if test.size == 0:
        return np.asarray([], dtype=np.int64), test
    embargo = horizon if embargo_steps is None else int(embargo_steps)
    keep = np.ones(valid_positions.shape[0], dtype=bool)
    for p in test:
        keep &= np.abs(valid_positions - int(p)) > embargo
    keep &= ~np.isin(valid_positions, test)
    train = np.sort(valid_positions[keep])
    return train, test


def run_prediction(table: FeatureTable, cfg: dict) -> dict[str, object]:
    seed = int(cfg["experiment"]["random_seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    lstm_cfg = cfg["experiment"]["lstm"]
    xgb_params = cfg["experiment"]["xgboost"]
    n_states = int(cfg["feature"].get("n_states", 3))
    horizon = int(cfg["feature"]["prediction_horizon_steps"])
    seq_len = int(lstm_cfg["sequence_length"])
    y = table.y
    assert y is not None
    main_idx = np.where(table.dataset == "xamn6")[0]
    x = table.x_obb[main_idx]
    x_lstm_raw = matrix_from_rows(table.rows, FEATURES_VDF)[main_idx]
    y_main = y[main_idx]

    split = int(round(x.shape[0] * (1.0 - float(cfg["experiment"]["test_ratio"]))))
    x_train_raw = x[:split]
    x_test_raw = x[split:]
    x_train, x_test, mean, std = standardize(x_train_raw, x_test_raw)
    x_scaled = (x - mean) / std
    _, _, lstm_mean, lstm_std = standardize(x_lstm_raw[:split], x_lstm_raw[split:])
    x_lstm_scaled = (x_lstm_raw - lstm_mean) / lstm_std

    valid_end = x.shape[0] - horizon
    train_end = min(split, valid_end)
    xgb = xgb_model(seed, xgb_params, n_states)
    fit_xgb_multiclass(xgb, x_scaled[:train_end], y_main[horizon : train_end + horizon], n_states)
    xgb_test_positions = np.arange(split, valid_end)
    xgb_prob = xgb.predict_proba(x_scaled[xgb_test_positions])
    xgb_pred = np.argmax(xgb_prob, axis=1)
    y_future_true = y_main[xgb_test_positions + horizon]

    score_main = table.score[main_idx] if table.score is not None else y_main.astype(np.float32)
    x_temporal = temporal_context_features(x, score=score_main, state=y_main)
    _, _, temporal_mean, temporal_std = standardize(x_temporal[:split], x_temporal[split:])
    x_temporal_scaled = (x_temporal - temporal_mean) / temporal_std
    temporal_xgb = xgb_model(seed, {**xgb_params, "max_depth": 2, "n_estimators": 80, "learning_rate": 0.08}, n_states)
    fit_xgb_multiclass(temporal_xgb, x_temporal_scaled[:train_end], y_main[horizon : train_end + horizon], n_states)
    temporal_xgb_prob = temporal_xgb.predict_proba(x_temporal_scaled[xgb_test_positions])
    temporal_xgb_pred = np.argmax(temporal_xgb_prob, axis=1)

    seq_x, seq_y, end_positions = sequence_dataset(x_lstm_scaled, y_main, seq_len, horizon)
    seq_train_mask = end_positions < split
    seq_test_mask = end_positions >= split
    train_positions = end_positions[seq_train_mask]
    val_start = int(round(split * 0.80))
    fit_mask = seq_train_mask & (end_positions < val_start)
    if int(np.sum(fit_mask)) < max(8, seq_len):
        fit_mask = seq_train_mask
        val_mask = seq_train_mask
    else:
        val_mask = seq_train_mask & (end_positions >= val_start)
    train_x = torch.tensor(seq_x[fit_mask], dtype=torch.float32)
    train_y = torch.tensor(seq_y[fit_mask], dtype=torch.long)
    test_x = torch.tensor(seq_x[seq_test_mask], dtype=torch.float32)
    test_y = seq_y[seq_test_mask]

    model = LSTMClassifier(input_size=x_lstm_raw.shape[1], hidden_size=int(lstm_cfg["hidden_size"]), num_classes=n_states)
    opt = torch.optim.Adam(model.parameters(), lr=float(lstm_cfg["learning_rate"]))
    train_counts = np.bincount(train_y.cpu().numpy(), minlength=n_states).astype(np.float32)
    class_weights = train_counts.sum() / np.maximum(train_counts, 1.0)
    class_weights = class_weights / max(1e-6, float(class_weights.mean()))
    loss_fn = FocalLoss(gamma=2.0, weight=torch.tensor(class_weights, dtype=torch.float32))
    batch_size = int(lstm_cfg["batch_size"])
    epochs = int(lstm_cfg["epochs"])
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(train_x.shape[0])
        for start in range(0, train_x.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            logits = model(train_x[idx])
            loss = loss_fn(logits, train_y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(test_x)
        lstm_prob = torch.softmax(logits, dim=1).cpu().numpy()
    lstm_pred = np.argmax(lstm_prob, axis=1)

    seq_test_positions = end_positions[seq_test_mask]
    xgb_prob_aligned = xgb.predict_proba(x_scaled[seq_test_positions])
    xgb_pred_aligned = np.argmax(xgb_prob_aligned, axis=1)
    temporal_xgb_prob_aligned = temporal_xgb.predict_proba(x_temporal_scaled[seq_test_positions])
    temporal_xgb_pred_aligned = np.argmax(temporal_xgb_prob_aligned, axis=1)
    common_true = y_main[seq_test_positions + horizon]

    static_weight = 1.0 / 3.0
    temporal_weight = 1.0 / 3.0
    lstm_weight = 1.0 / 3.0
    fusion_selection_note = "temperature-softmax dynamic gate, default equal prior"
    fusion_channel_errors: dict[str, float] = {}
    fusion_temperature = 1.0
    if np.any(val_mask):
        val_x = torch.tensor(seq_x[val_mask], dtype=torch.float32)
        val_y = seq_y[val_mask]
        val_positions = end_positions[val_mask]
        with torch.no_grad():
            val_lstm_prob = torch.softmax(model(val_x), dim=1).cpu().numpy()
        val_static_prob = xgb.predict_proba(x_scaled[val_positions])
        val_temporal_prob = temporal_xgb.predict_proba(x_temporal_scaled[val_positions])
        eps = 1e-8
        val_probs = np.stack([
            np.clip(val_static_prob[np.arange(val_y.shape[0]), val_y], eps, 1.0),
            np.clip(val_temporal_prob[np.arange(val_y.shape[0]), val_y], eps, 1.0),
            np.clip(val_lstm_prob[np.arange(val_y.shape[0]), val_y], eps, 1.0),
        ], axis=1)
        ce = -np.log(val_probs)
        smooth_alpha = 0.35
        smoothed = np.zeros_like(ce)
        smoothed[0] = ce[0]
        for i in range(1, ce.shape[0]):
            smoothed[i] = smooth_alpha * ce[i] + (1.0 - smooth_alpha) * smoothed[i - 1]
        channel_err = smoothed.mean(axis=0)
        t0 = 1.0
        temp_alpha = 0.45
        t_min = 0.20
        fusion_temperature = float(max(t_min, t0 * math.exp(-temp_alpha * float(np.mean(channel_err)))))
        logits = -channel_err / max(fusion_temperature, 1e-6)
        logits -= float(np.max(logits))
        weights = np.exp(logits)
        weights = weights / max(float(np.sum(weights)), eps)
        static_weight, temporal_weight, lstm_weight = [float(v) for v in weights]
        fusion_channel_errors = {
            "static_xgb": float(channel_err[0]),
            "temporal_xgb": float(channel_err[1]),
            "lstm": float(channel_err[2]),
        }
        val_fusion_prob = static_weight * val_static_prob + temporal_weight * val_temporal_prob + lstm_weight * val_lstm_prob
        val_fusion_pred = np.argmax(val_fusion_prob, axis=1)
        val_fusion_score = metrics_dict(val_y, val_fusion_pred, n_states)["f1_macro"]
        fusion_selection_note = (
            f"temperature-softmax dynamic gate, validation Macro-F1={val_fusion_score:.4f}, "
            f"T={fusion_temperature:.3f}, static_xgb={static_weight:.2f}, "
            f"temporal_xgb={temporal_weight:.2f}, lstm={lstm_weight:.2f}"
        )
    fusion_prob = static_weight * xgb_prob_aligned + temporal_weight * temporal_xgb_prob_aligned + lstm_weight * lstm_prob
    fusion_pred = np.argmax(fusion_prob, axis=1)

    balanced_train_pos, balanced_test_pos = stratified_future_positions(
        y_main,
        horizon,
        float(cfg["experiment"]["test_ratio"]),
        seed,
        n_states,
        embargo_steps=horizon,
    )
    balanced_result: dict[str, object] = {"status": "skipped", "reason": "insufficient positions"}
    if balanced_train_pos.size > 0 and balanced_test_pos.size > 0:
        balanced_model = xgb_model(seed, xgb_params, n_states)
        fit_xgb_multiclass(
            balanced_model,
            x_scaled[balanced_train_pos],
            y_main[balanced_train_pos + horizon],
            n_states,
            sample_weight=compute_sample_weights(y_main[balanced_train_pos + horizon], n_states),
        )
        balanced_pred = class_predictions(balanced_model.predict(x_scaled[balanced_test_pos]))
        balanced_true = y_main[balanced_test_pos + horizon]
        balanced_result = {
            "status": "completed",
            "model": "XGBoost-future",
            "metrics": metrics_dict(balanced_true, balanced_pred, n_states),
            "confusion_matrix": confusion_matrix_np(balanced_true, balanced_pred, n_states).tolist(),
            "test_support": {STATE_NAMES[i] if i < len(STATE_NAMES) else str(i): int(np.sum(balanced_true == i)) for i in range(n_states)},
            "train_windows": int(balanced_train_pos.size),
            "test_windows": int(balanced_test_pos.size),
            "embargo_steps": int(horizon),
            "note": "State-balanced future prediction with +/- horizon embargo around test positions; main prediction remains chronological.",
        }

    return {
        "horizon_steps": horizon,
        "horizon_seconds": float(horizon * float(cfg["feature"]["step_s"])),
        "XGBoost-future": {
            "metrics": metrics_dict(common_true, xgb_pred_aligned, n_states),
            "confusion_matrix": confusion_matrix_np(common_true, xgb_pred_aligned, n_states).tolist(),
            "pred": xgb_pred_aligned.tolist(),
            "true": common_true.tolist(),
        },
        "XGBoost-temporal-future": {
            "metrics": metrics_dict(common_true, temporal_xgb_pred_aligned, n_states),
            "confusion_matrix": confusion_matrix_np(common_true, temporal_xgb_pred_aligned, n_states).tolist(),
            "pred": temporal_xgb_pred_aligned.tolist(),
            "true": common_true.tolist(),
            "note": "XGBoost with current-state score, lag, delta and rolling trend features.",
        },
        "LSTM-future": {
            "metrics": metrics_dict(common_true, lstm_pred, n_states),
            "confusion_matrix": confusion_matrix_np(common_true, lstm_pred, n_states).tolist(),
            "pred": lstm_pred.tolist(),
            "true": common_true.tolist(),
            "input_features": FEATURES_VDF,
            "note": "Low-dimensional temporal channel using V+D+F features.",
        },
        "Fusion-future": {
            "metrics": metrics_dict(common_true, fusion_pred, n_states),
            "confusion_matrix": confusion_matrix_np(common_true, fusion_pred, n_states).tolist(),
            "pred": fusion_pred.tolist(),
            "true": common_true.tolist(),
            "xgb_weight": static_weight + temporal_weight,
            "static_xgb_weight": static_weight,
            "temporal_xgb_weight": temporal_weight,
            "lstm_weight": lstm_weight,
            "selection_note": fusion_selection_note,
            "fusion_method": "temperature_softmax_with_ema_cross_entropy",
            "temperature": fusion_temperature,
            "validation_channel_errors": fusion_channel_errors,
        },
        "state_balanced_supplementary": balanced_result,
        "test_positions": seq_test_positions.tolist(),
    }


def run_deterioration_prediction(table: FeatureTable, cfg: dict) -> dict[str, object]:
    seed = int(cfg["experiment"]["random_seed"])
    det_xgb_params = cfg["experiment"].get("deterioration_xgboost", cfg["experiment"]["xgboost"])
    step_s = float(cfg["feature"]["step_s"])
    horizons_s = cfg["feature"]["deterioration_horizons_s"]
    det_folds = int(cfg["experiment"].get("deterioration_group_folds", 5))
    det_std_multiplier = float(cfg["experiment"].get("deterioration_std_multiplier", 1.5))
    cst_cost = float(cfg["experiment"].get("deterioration_cst_cost", 0.8))
    min_positive = int(cfg["experiment"].get("deterioration_min_positive", 10))
    y = table.y
    assert y is not None

    table.mgti_composite = compute_composite_mgti(table, cfg)
    main_idx = np.where(table.dataset == "xamn6")[0]
    y_main = y[main_idx]
    score_main = table.score[main_idx] if table.score is not None else y_main.astype(np.float32)
    mgti_main = table.mgti_composite[main_idx]

    results = {}
    for horizon_s in horizons_s:
        horizon_steps = max(1, int(round(horizon_s / step_s)))
        det_labels = make_deterioration_labels_score(
            score_main,
            horizon_steps,
            threshold_pct=None,
            std_multiplier=det_std_multiplier,
        )

        valid_mask = det_labels >= 0
        valid_idx = main_idx[valid_mask]
        valid_labels = det_labels[valid_mask]
        pos_count = int(np.sum(valid_labels == 1))
        neg_count = int(np.sum(valid_labels == 0))

        if pos_count < min_positive:
            results[f"horizon_{horizon_s}s"] = {
                "status": "skipped",
                "reason": f"Only {pos_count} positive samples (need {min_positive})",
                "horizon_seconds": horizon_s,
            }
            continue

        base_x = table.x_obb[valid_idx]
        mgti_col = mgti_main[valid_mask].reshape(-1, 1)
        full_x = np.column_stack([base_x, mgti_col])

        n_valid = valid_idx.shape[0]
        n_splits = min(det_folds, n_valid)
        group_edges = np.linspace(0, n_valid, n_splits + 1, dtype=int)
        groups = np.zeros(n_valid, dtype=np.int64)
        for group_id in range(n_splits):
            groups[group_edges[group_id] : group_edges[group_id + 1]] = group_id

        ablation_results = {}
        for abl_name, abl_features in DETERIORATION_ABLATION_SETS.items():
            feat_indices = []
            for feat_name in abl_features:
                if feat_name == "mgti_composite":
                    feat_indices.append(full_x.shape[1] - 1)
                elif feat_name in NUMERIC_FEATURES_OBB:
                    feat_indices.append(NUMERIC_FEATURES_OBB.index(feat_name))
                else:
                    continue
            x_abl = full_x[:, feat_indices]
            pred_oof = np.zeros(n_valid, dtype=np.int64)
            prob_pos_oof = np.zeros(n_valid, dtype=np.float32)
            importances_accum = np.zeros(len(feat_indices), dtype=np.float32)
            fitted_folds = 0
            splitter = GroupKFold(n_splits=n_splits)
            for fold_id, (train_fold, test_fold) in enumerate(splitter.split(x_abl, valid_labels, groups=groups)):
                train_labels = valid_labels[train_fold]
                if np.unique(train_labels).shape[0] < 2:
                    majority = int(np.bincount(train_labels, minlength=2).argmax())
                    pred_oof[test_fold] = majority
                    prob_pos_oof[test_fold] = float(majority)
                    continue
                train_pos = int(np.sum(train_labels == 1))
                train_neg = int(np.sum(train_labels == 0))
                scale_pos = float(train_neg) / max(1.0, float(train_pos))
                x_tr, x_te, _, _ = standardize(x_abl[train_fold], x_abl[test_fold])
                model = xgb_binary_model(seed + fold_id, det_xgb_params, scale_pos_weight=scale_pos)
                model.fit(x_tr, train_labels)
                pred_oof[test_fold] = class_predictions(model.predict(x_te))
                prob = model.predict_proba(x_te)
                prob_pos_oof[test_fold] = prob[:, 1]
                if hasattr(model, "feature_importances_"):
                    importances_accum += np.asarray(model.feature_importances_, dtype=np.float32)
                fitted_folds += 1

            abl_metrics = metrics_dict(valid_labels, pred_oof, 2)
            try:
                roc_auc = float(roc_auc_score(valid_labels, prob_pos_oof))
            except ValueError:
                roc_auc = None
            try:
                pr_auc = float(average_precision_score(valid_labels, prob_pos_oof))
            except ValueError:
                pr_auc = None

            abl_item: dict[str, object] = {
                "metrics": abl_metrics,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "cst_operating_point": cost_sensitive_operating_point(valid_labels, prob_pos_oof, cost=cst_cost),
                "confusion_matrix": confusion_matrix_np(valid_labels, pred_oof, 2).tolist(),
                "features": abl_features,
                "evaluation": "contiguous GroupKFold out-of-fold",
            }
            if fitted_folds > 0:
                importances = importances_accum / float(fitted_folds)
                order = np.argsort(importances)[::-1][:8]
                abl_item["feature_importance"] = [
                    {"feature": abl_features[int(i)], "importance": float(importances[int(i)])}
                    for i in order
                ]
            ablation_results[abl_name] = abl_item

        results[f"horizon_{horizon_s}s"] = {
            "status": "completed",
            "horizon_seconds": horizon_s,
            "horizon_steps": horizon_steps,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "positive_rate": pos_count / max(1, pos_count + neg_count),
            "cv_folds": n_splits,
            "event_threshold": f"mean+{det_std_multiplier:.2f}std",
            "ablation": ablation_results,
        }

    return results


def pkdd_generalization(table: FeatureTable, cfg: dict) -> dict[str, object]:
    seed = int(cfg["experiment"]["random_seed"])
    xgb_params = cfg["experiment"]["xgboost"]
    n_states = int(cfg["feature"].get("n_states", 3))
    y = table.y
    assert y is not None
    train_idx = np.where(table.dataset == "xamn6")[0]
    pkdd_idx = np.where(table.dataset == "pkdd8")[0]
    x_train, x_pkdd, _, _ = standardize(table.x_obb[train_idx], table.x_obb[pkdd_idx])
    model = xgb_model(seed, xgb_params, n_states)
    fit_xgb_multiclass(model, x_train, y[train_idx], n_states)
    prob = model.predict_proba(x_pkdd)
    pred = class_predictions(prob)
    names = STATE_NAMES if n_states <= len(STATE_NAMES) else [str(i) for i in range(n_states)]
    distribution = {names[i]: int(np.sum(pred == i)) for i in range(n_states)}
    free_prob = prob[:, 0] if prob.shape[1] else np.zeros(pkdd_idx.shape[0], dtype=np.float32)
    hist_counts, hist_edges = np.histogram(free_prob, bins=np.linspace(0.0, 1.0, 11))
    return {
        "windows": int(pkdd_idx.shape[0]),
        "predicted_distribution": distribution,
        "mean_probability": {names[i]: float(np.mean(prob[:, i])) for i in range(min(n_states, prob.shape[1]))},
        "free_probability_quantiles": {
            "p05": float(np.quantile(free_prob, 0.05)),
            "p25": float(np.quantile(free_prob, 0.25)),
            "p50": float(np.quantile(free_prob, 0.50)),
            "p75": float(np.quantile(free_prob, 0.75)),
            "p95": float(np.quantile(free_prob, 0.95)),
        },
        "free_probability_histogram": {
            "bin_edges": [float(v) for v in hist_edges.tolist()],
            "counts": [int(v) for v in hist_counts.tolist()],
        },
        "note": "PKDD-8 is used as a free-flow transfer check. Metrics against transferred proxy labels are intentionally not reported.",
    }


def save_results(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
