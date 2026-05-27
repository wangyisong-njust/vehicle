"""Traffic state label generation for window-level features.

Implements an automatic 4-class state labeler (free / slow / crowded / congested)
using K-Means clustering on V+D+R+F features with a physical-monotonicity
fallback guard.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler


STATE_NAMES = ["畅通", "缓行", "拥挤", "堵塞"]
STATE_NAMES_EN = ["Free", "Slow", "Crowded", "Congested"]


def _matrix_from_rows(rows: list[dict[str, str]], features: list[str]) -> np.ndarray:
    return np.asarray([[float(row[k]) for k in features] for row in rows], dtype=np.float32)


def _z(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / (values.std() + 1e-6)


def _robust_scaled(train: np.ndarray, values: np.ndarray) -> np.ndarray:
    scaler = RobustScaler()
    scaler.fit(train.reshape(-1, 1))
    return scaler.transform(values.reshape(-1, 1)).reshape(-1)


def smooth_labels(labels: np.ndarray, start_s: np.ndarray, dataset: np.ndarray,
                  n_states: int, window: int) -> np.ndarray:
    """Median-vote smoothing within each dataset along the time axis."""
    if window <= 1:
        return labels
    radius = window // 2
    smoothed = labels.copy()
    for ds in np.unique(dataset):
        idx = np.where(dataset == ds)[0]
        order = idx[np.argsort(start_s[idx])]
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


def make_state_labels(
    rows_all: list[dict[str, str]],
    dataset_all: np.ndarray,
    start_s_all: np.ndarray,
    main_dataset: str = "xamn6",
    n_states: int = 4,
    quantiles: tuple[float, ...] = (0.25, 0.50, 0.75),
    random_seed: int = 42,
    smoothing_window: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate 4-class traffic state labels for each window.

    Uses K-Means on [speed_ratio, density, lane_change_rate, direction_fluctuation]
    on the main dataset to obtain candidate clusters; orders them by a macro
    risk score (-speed + density + 0.5*occupancy); if the resulting label
    sequence does not satisfy the speed-decreasing / density-increasing /
    occupancy-increasing monotonicity check, falls back to a quantile-based
    risk score discretization.

    Returns:
        labels: int array shape (N,) -- state label per window
        score:  float array shape (N,) -- continuous risk score
        thresholds: float array shape (n_states - 1,) -- score quantile thresholds
    """
    assert len(quantiles) == n_states - 1, "quantiles length must be n_states - 1"

    main_indices = np.where(dataset_all == main_dataset)[0]
    rows_main = [rows_all[i] for i in main_indices]

    label_features = ["speed_ratio", "density_veh_per_m", "lane_change_rate", "direction_fluctuation"]
    x_main = _matrix_from_rows(rows_main, label_features)
    scaler = RobustScaler()
    x_main_z = scaler.fit_transform(x_main)
    kmeans = KMeans(n_clusters=n_states, random_state=random_seed, n_init=50)
    main_cluster = kmeans.fit_predict(x_main_z)

    macro_centers = []
    for cluster in range(n_states):
        cluster_rows = [rows_main[i] for i in np.where(main_cluster == cluster)[0]]
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

    x_all = _matrix_from_rows(rows_all, label_features)
    x_all_z = scaler.transform(x_all)
    all_cluster = kmeans.predict(x_all_z)
    cluster_labels = np.asarray([cluster_to_state[int(c)] for c in all_cluster], dtype=np.int64)

    occ_key = "hfgo_occupancy" if "hfgo_occupancy" in rows_main[0] else "obb_occupancy"
    sr_main = np.asarray([float(r["speed_ratio"]) for r in rows_main], dtype=np.float32)
    d_main = np.asarray([float(r["density_veh_per_m"]) for r in rows_main], dtype=np.float32)
    occ_main = np.asarray([float(r[occ_key]) for r in rows_main], dtype=np.float32)
    sr_all = np.asarray([float(r["speed_ratio"]) for r in rows_all], dtype=np.float32)
    d_all = np.asarray([float(r["density_veh_per_m"]) for r in rows_all], dtype=np.float32)
    occ_all = np.asarray([float(r[occ_key]) for r in rows_all], dtype=np.float32)

    score_all = (
        0.65 * _robust_scaled(1.0 - sr_main, 1.0 - sr_all)
        + 0.25 * _robust_scaled(d_main, d_all)
        + 0.10 * _robust_scaled(occ_main, occ_all)
    ).astype(np.float32)
    thresholds = np.quantile(score_all[main_indices], quantiles).astype(np.float32)
    fallback_labels = np.digitize(score_all, thresholds).astype(np.int64)
    fallback_labels = np.clip(fallback_labels, 0, n_states - 1)

    def _passes_physical_order(candidate: np.ndarray) -> bool:
        state_speed, state_density, state_occ = [], [], []
        for state in range(n_states):
            state_idx = main_indices[candidate[main_indices] == state]
            if state_idx.shape[0] == 0:
                return False
            state_speed.append(float(np.mean([float(rows_all[i]["mean_speed_kmh"]) for i in state_idx])))
            state_density.append(float(np.mean([float(rows_all[i]["density_veh_per_m"]) for i in state_idx])))
            state_occ.append(float(np.mean([float(rows_all[i]["obb_occupancy"]) for i in state_idx])))
        return (
            all(state_speed[i] >= state_speed[i + 1] - 1e-6 for i in range(n_states - 1))
            and state_speed[-1] < state_speed[0]
            and abs(state_density[-1] - max(state_density)) < 1e-6
            and abs(state_occ[-1] - max(state_occ)) < 1e-6
        )

    labels = cluster_labels if _passes_physical_order(cluster_labels) else fallback_labels
    labels = smooth_labels(labels, start_s_all, dataset_all, n_states, smoothing_window)
    return labels, score_all, thresholds
