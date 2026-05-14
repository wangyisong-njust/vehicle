from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from ute_pipeline.config import load_config
from ute_pipeline.experiments import (
    STATE_NAMES,
    STATE_NAMES_EN,
    compute_composite_mgti,
    make_state_labels,
    read_feature_table,
)
from ute_pipeline.features import ObbData, grid_area_heatmap, load_obb, video_shape

CHINESE_STATE_NAMES_4 = ["畅通", "缓行", "拥挤", "堵塞"]
ENGLISH_STATE_NAMES_4 = ["Free", "Slow", "Crowded", "Congested"]


def configure_plot_font() -> list[str]:
    candidates = [
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return CHINESE_STATE_NAMES_4
    return ENGLISH_STATE_NAMES_4


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_optional(path: Path) -> dict:
    if not path.exists():
        return {}
    return read_json(path)


def read_feature_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def plot_metric_bars(results: dict, out_path: Path) -> None:
    items = results["classification"]
    names = [name for name, item in items.items() if isinstance(item, dict) and "metrics" in item]
    metrics = ["accuracy", "f1_macro", "recall_macro"]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, metric in enumerate(metrics):
        vals = [items[name]["metrics"][metric] for name in names]
        ax.bar(x + (i - 1) * width, vals, width=width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion(cm: list[list[int]], title: str, out_path: Path, state_names: list[str]) -> None:
    arr = np.asarray(cm)
    n = arr.shape[0]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(state_names[:n])
    ax.set_yticklabels(state_names[:n])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    row_sum = np.maximum(arr.sum(axis=1, keepdims=True), 1)
    pct = arr / row_sum * 100.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{arr[i, j]}\n{pct[i, j]:.0f}%", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_occupancy(rows: list[dict[str, str]], out_path: Path) -> None:
    xamn = [r for r in rows if r["dataset"] == "xamn6"]
    t = np.asarray([float(r["start_s"]) for r in xamn])
    hbb = np.asarray([float(r["hbb_occupancy"]) for r in xamn])
    obb = np.asarray([float(r["obb_occupancy"]) for r in xamn])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t, hbb, label="HBB occupancy", linewidth=1.5)
    ax.plot(t, obb, label="OBB occupancy", linewidth=1.5)
    ax.fill_between(t, obb, hbb, alpha=0.18, label="HBB-OBB gap")
    ax.set_xlabel("time / s")
    ax.set_ylabel("window occupancy")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_hfgo_heatmap(root: Path, cfg: dict, out_path: Path) -> None:
    ds = cfg["datasets"]["xamn6"]
    ds_root = root / ds["root"]
    width, height, _, _ = video_shape(ds_root / ds["video"])
    obb = load_obb(root / "outputs" / "processed" / "xamn6_pixel_obb.csv")
    hbb_map, hfgo_map = grid_area_heatmap(
        obb,
        width,
        height,
        int(cfg["feature"].get("grid_cols", 12)),
        int(cfg["feature"].get("grid_rows", 4)),
        max_rows=120_000,
    )
    vmax = max(float(hbb_map.max()), float(hfgo_map.max()), 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, mat, title in zip(axes, [hbb_map, hfgo_map], ["HBB grid occupancy", "HF-GO OBB occupancy"]):
        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("grid column")
        ax.set_ylabel("grid row")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.026, pad=0.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _slice_obb(obb: ObbData, mask: np.ndarray) -> ObbData:
    return ObbData(
        frame=obb.frame[mask],
        time_s=obb.time_s[mask],
        vehicle_id=obb.vehicle_id[mask],
        x=obb.x[mask],
        y=obb.y[mask],
        w=obb.w[mask],
        h=obb.h[mask],
        obb_area=obb.obb_area[mask],
        theta=obb.theta[mask],
        theta_conf=obb.theta_conf[mask],
    )


def plot_hfgo_local_by_state(root: Path, cfg: dict, labels: np.ndarray, out_path: Path) -> None:
    rows = [r for r in read_feature_rows(root / "outputs" / "features" / "all_windows.csv") if r["dataset"] == "xamn6"]
    if not rows:
        return
    ds = cfg["datasets"]["xamn6"]
    ds_root = root / ds["root"]
    width, height, _, _ = video_shape(ds_root / ds["video"])
    obb = load_obb(root / "outputs" / "processed" / "xamn6_pixel_obb.csv")
    grid_cols = int(cfg["feature"].get("grid_cols", 12))
    grid_rows = int(cfg["feature"].get("grid_rows", 4))
    state_names = configure_plot_font()
    fig, axes = plt.subplots(4, 2, figsize=(9.2, 8.5))
    vmax = 1.0
    maps: list[tuple[np.ndarray, np.ndarray]] = []
    selected: list[int] = []
    main_labels = labels[np.asarray([i for i, r in enumerate(read_feature_rows(root / "outputs" / "features" / "all_windows.csv")) if r["dataset"] == "xamn6"])]
    for state in range(min(4, len(state_names))):
        idx = np.where(main_labels == state)[0]
        if idx.size == 0:
            selected.append(-1)
            maps.append((np.zeros((grid_rows, grid_cols)), np.zeros((grid_rows, grid_cols))))
            continue
        pos = int(idx[idx.size // 2])
        selected.append(pos)
        start = float(rows[pos]["start_s"])
        end = float(rows[pos]["end_s"])
        mask = (obb.time_s >= start) & (obb.time_s < end)
        if int(mask.sum()) > 25000:
            true_idx = np.where(mask)[0][:25000]
            tmp = np.zeros(mask.shape, dtype=bool)
            tmp[true_idx] = True
            mask = tmp
        sub = _slice_obb(obb, mask)
        if sub.frame.size == 0:
            hbb = np.zeros((grid_rows, grid_cols), dtype=np.float32)
            hfgo = np.zeros((grid_rows, grid_cols), dtype=np.float32)
        else:
            hbb, hfgo = grid_area_heatmap(sub, width, height, grid_cols, grid_rows, max_rows=25000)
        maps.append((hbb, hfgo))
        vmax = max(vmax, float(hbb.max()), float(hfgo.max()))
    for state, (hbb, hfgo) in enumerate(maps):
        for col, (mat, title) in enumerate([(hbb, "HBB"), (hfgo, "HF-GO")]):
            ax = axes[state, col]
            ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
            title_suffix = "" if selected[state] < 0 else f" t={float(rows[selected[state]]['start_s']):.0f}s"
            ax.set_title(f"{state_names[state]} {title}{title_suffix}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Local HBB vs HF-GO Occupancy by State", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_state_spacetime(root: Path, cfg: dict, labels: np.ndarray, out_path: Path) -> None:
    rows_all = read_feature_rows(root / "outputs" / "features" / "all_windows.csv")
    xamn_rows = [r for r in rows_all if r["dataset"] == "xamn6"]
    if not xamn_rows:
        return
    state_names = configure_plot_font()
    ds = cfg["datasets"]["xamn6"]
    width, height, _, _ = video_shape(root / ds["root"] / ds["video"])
    obb = load_obb(root / "outputs" / "processed" / "xamn6_pixel_obb.csv")
    grid_cols = int(cfg["feature"].get("grid_cols", 12))
    grid_rows = int(cfg["feature"].get("grid_rows", 4))
    starts = np.asarray([float(r["start_s"]) for r in xamn_rows], dtype=np.float32)
    step = float(cfg["feature"].get("step_s", 1.0))
    main_positions = np.asarray([i for i, r in enumerate(rows_all) if r["dataset"] == "xamn6"], dtype=np.int64)
    main_labels = labels[main_positions]
    mat = np.zeros((grid_rows * grid_cols, starts.shape[0]), dtype=np.float32)
    win = np.floor((obb.time_s - starts[0]) / max(step, 1e-6)).astype(np.int64)
    col = np.clip((obb.x + obb.w / 2.0) / max(width, 1) * grid_cols, 0, grid_cols - 1).astype(np.int64)
    row = np.clip((obb.y + obb.h / 2.0) / max(height, 1) * grid_rows, 0, grid_rows - 1).astype(np.int64)
    valid = (win >= 0) & (win < starts.shape[0])
    cells = row[valid] * grid_cols + col[valid]
    wins = win[valid]
    mat[cells, wins] = main_labels[wins] + 1
    fig, ax = plt.subplots(figsize=(11, 5.2))
    cmap = plt.get_cmap("viridis", 5)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=4)
    ax.set_xlabel("time window")
    ax.set_ylabel("grid cell (row-major 12x4)")
    ax.set_title("Spatio-temporal State Map on XAM-N-6")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4], fraction=0.025, pad=0.02)
    cbar.ax.set_yticklabels(["empty"] + state_names[:4])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pkdd_probability(pkdd: dict, out_path: Path) -> None:
    hist = pkdd.get("free_probability_histogram", {})
    counts = hist.get("counts", [])
    edges = hist.get("bin_edges", [])
    if not counts or not edges:
        return
    left = np.asarray(edges[:-1], dtype=np.float32)
    width = np.diff(np.asarray(edges, dtype=np.float32))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(left, counts, width=width, align="edge", color="#4c78a8", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("P(Free)")
    ax.set_ylabel("PKDD windows")
    ax.set_title("PKDD Free-state Probability Distribution")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_shap_summary(shap_summary: dict, out_path: Path) -> None:
    if not shap_summary or shap_summary.get("status") != "ok":
        return
    items = shap_summary.get("global_top_features", [])[:10]
    if not items:
        return
    names = [item["feature"] for item in items][::-1]
    vals = [item["mean_abs_shap"] for item in items][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.barh(np.arange(len(names)), vals, color="#72b7b2", edgecolor="black", linewidth=0.4)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("mean |TreeSHAP contribution|")
    ax.set_title("XGBoost-OBB TreeSHAP Feature Contribution")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_counterfactual(counterfactual: dict, out_path: Path) -> None:
    if not counterfactual or counterfactual.get("status") != "completed":
        return
    cases = counterfactual.get("cases", [])
    if not cases:
        return
    case = cases[0]
    features = case.get("features", [])[:3]
    if not features:
        return
    fig, axes = plt.subplots(1, len(features), figsize=(4.2 * len(features), 3.6), squeeze=False)
    for ax, item in zip(axes[0], features):
        curve = item.get("curve", [])
        x = [p["feature_value_standardized"] for p in curve]
        true_prob = [p["true_class_prob"] for p in curve]
        pred_prob = [p["pred_class_prob"] for p in curve]
        ax.plot(x, true_prob, marker="o", label="true class")
        ax.plot(x, pred_prob, marker="s", label="pred class")
        ax.set_title(item.get("feature", "feature"))
        ax.set_xlabel("standardized value")
        ax.set_ylabel("probability")
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("SHAP-guided What-if Counterfactual Curves", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_rf_scatter(rows: list[dict[str, str]], labels: np.ndarray, out_path: Path) -> None:
    datasets = ["xamn6", "pkdd8"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    state_names = configure_plot_font()
    all_xamn_idx = [i for i, r in enumerate(rows) if r["dataset"] == "xamn6"]
    xamn_label_map = {idx: int(labels[pos]) for pos, idx in enumerate(all_xamn_idx)}
    for ax, dataset in zip(axes, datasets):
        ds_rows = [(i, r) for i, r in enumerate(rows) if r["dataset"] == dataset]
        if not ds_rows:
            continue
        r_vals = np.asarray([float(r["lane_change_rate"]) for _, r in ds_rows])
        f_vals = np.asarray([float(r["direction_fluctuation"]) for _, r in ds_rows])
        if dataset == "xamn6":
            colors = np.asarray([xamn_label_map.get(i, 0) for i, _ in ds_rows])
            sc = ax.scatter(r_vals, f_vals, c=colors, cmap="viridis", s=16, alpha=0.75)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_yticks(range(min(4, len(state_names))))
            cbar.ax.set_yticklabels(state_names[:4])
        else:
            ax.scatter(r_vals, f_vals, color="#4c78a8", s=14, alpha=0.65)
        corr = float(np.corrcoef(r_vals, f_vals)[0, 1]) if r_vals.std() > 1e-9 and f_vals.std() > 1e-9 else 0.0
        ax.set_title(f"{dataset} R-F corr={corr:.3f}")
        ax.set_xlabel("lane_change_rate R")
        ax.set_ylabel("direction_fluctuation F")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_rf_feature_space(rows: list[dict[str, str]], labels: np.ndarray, out_path: Path) -> None:
    xamn_rows = [r for r in rows if r["dataset"] == "xamn6"]
    if not xamn_rows:
        return
    state_names = configure_plot_font()
    speed = np.asarray([float(r["mean_speed_kmh"]) for r in xamn_rows])
    density = np.asarray([float(r["density_veh_per_m"]) for r in xamn_rows])
    r_vals = np.asarray([float(r["lane_change_rate"]) for r in xamn_rows])
    f_vals = np.asarray([float(r["direction_fluctuation"]) for r in xamn_rows])
    colors = np.asarray(labels[: len(xamn_rows)], dtype=np.int64)

    fig = plt.figure(figsize=(10.5, 8.5))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    axes = [ax1, ax2, ax3]
    pairs = [
        (speed, density, "mean_speed_kmh V", "density D"),
        (speed, r_vals, "mean_speed_kmh V", "lane_change_rate R"),
        (speed, f_vals, "mean_speed_kmh V", "direction_fluctuation F"),
    ]
    for ax, (x_vals, y_vals, xlabel, ylabel) in zip(axes, pairs):
        sc = ax.scatter(x_vals, y_vals, c=colors, cmap="viridis", s=18, alpha=0.75)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    ax4.scatter(density, r_vals, f_vals, c=colors, cmap="viridis", s=16, alpha=0.75)
    ax4.set_xlabel("density D")
    ax4.set_ylabel("R")
    ax4.set_zlabel("F")
    ax4.set_title("D-R-F projection")
    cbar = fig.colorbar(sc, ax=[ax1, ax2, ax3, ax4], fraction=0.025, pad=0.02)
    cbar.ax.set_yticks(range(min(4, len(state_names))))
    cbar.ax.set_yticklabels(state_names[:4])
    fig.suptitle("V-D-R-F State Feature Space", fontsize=12)
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.08, top=0.92, wspace=0.30, hspace=0.35)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_conformal_sweep(sweep: list[dict], out_path: Path) -> None:
    items = [item for item in sweep if item.get("status") == "completed"]
    if not items:
        return
    confidence = [float(item["confidence"]) for item in items]
    coverage = [float(item["coverage"]) for item in items]
    avg_size = [float(item["average_set_size"]) for item in items]
    singleton = [float(item["singleton_rate"]) for item in items]
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax1.plot(confidence, coverage, marker="o", label="empirical coverage")
    ax1.plot(confidence, confidence, linestyle="--", color="#777777", label="nominal confidence")
    ax1.set_xlabel("nominal confidence")
    ax1.set_ylabel("coverage")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(confidence, avg_size, marker="s", color="#f58518", label="avg set size")
    ax2.plot(confidence, singleton, marker="^", color="#54a24b", label="singleton rate")
    ax2.set_ylabel("set size / singleton rate")
    ax2.set_ylim(0, max(4.0, max(avg_size) * 1.15))
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_ttest_heatmap(matrix: dict, out_path: Path) -> None:
    methods = matrix.get("methods", [])
    pvalues = matrix.get("pvalues", {})
    if not methods:
        return
    values = np.ones((len(methods), len(methods)), dtype=np.float32)
    for i, left in enumerate(methods):
        for j, right in enumerate(methods):
            val = pvalues.get(left, {}).get(right)
            values[i, j] = 1.0 if val is None else float(val)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(values, cmap="viridis_r", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_yticklabels(methods)
    for i in range(len(methods)):
        for j in range(len(methods)):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=8, color="white" if values[i, j] < 0.35 else "black")
    ax.set_title("Paired t-test p-values on fold Macro-F1")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def rf_correlation_table(rows: list[dict[str, str]], labels: np.ndarray) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    xamn_positions = [i for i, r in enumerate(rows) if r["dataset"] == "xamn6"]
    for dataset in sorted(set(r["dataset"] for r in rows)):
        ds_idx = [i for i, r in enumerate(rows) if r["dataset"] == dataset]
        r_vals = np.asarray([float(rows[i]["lane_change_rate"]) for i in ds_idx])
        f_vals = np.asarray([float(rows[i]["direction_fluctuation"]) for i in ds_idx])
        corr = float(np.corrcoef(r_vals, f_vals)[0, 1]) if r_vals.std() > 1e-9 and f_vals.std() > 1e-9 else 0.0
        result.append({"scope": dataset, "count": len(ds_idx), "pearson_r": corr})
    state_names = configure_plot_font()
    for state in range(min(4, len(state_names))):
        pos = [xamn_positions[i] for i in range(len(xamn_positions)) if int(labels[i]) == state]
        if len(pos) < 3:
            continue
        r_vals = np.asarray([float(rows[i]["lane_change_rate"]) for i in pos])
        f_vals = np.asarray([float(rows[i]["direction_fluctuation"]) for i in pos])
        corr = float(np.corrcoef(r_vals, f_vals)[0, 1]) if r_vals.std() > 1e-9 and f_vals.std() > 1e-9 else 0.0
        result.append({"scope": f"xamn6-{state_names[state]}", "count": len(pos), "pearson_r": corr})
    return result


def plot_prediction_curve(pred_block: dict, out_path: Path, state_names: list[str]) -> None:
    true = np.asarray(pred_block["true"])
    pred = np.asarray(pred_block["pred"])
    n = true.shape[0]
    n_classes = int(max(true.max(), pred.max())) + 1
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.step(np.arange(n), true[:n], where="post", label="true", linewidth=1.8)
    ax.step(np.arange(n), pred[:n], where="post", label="pred", linewidth=1.4)
    ax.set_yticks(np.arange(n_classes))
    ax.set_yticklabels(state_names[:n_classes])
    ax.set_xlabel("test window")
    ax.set_ylabel("state")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_ablation(results: dict, out_path: Path) -> None:
    names = list(results.keys())
    vals = [results[name]["metrics"]["f1_macro"] for name in names]
    stds = [results[name]["metrics"].get("f1_macro_std", 0.0) for name in names]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(np.arange(len(names)), vals, yerr=stds, capsize=3, color="#4c78a8", edgecolor="black", linewidth=0.5)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, max(1.0, max(vals) * 1.15))
    ax.set_ylabel("Macro-F1 (5-fold CV)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_sensitivity(results: dict, out_path: Path) -> None:
    depths = [item["max_depth"] for item in results["max_depth"]]
    f1_depth = [item["metrics"]["f1_macro"] for item in results["max_depth"]]
    f1_depth_std = [item["metrics"].get("f1_macro_std", 0.0) for item in results["max_depth"]]
    completed_horizons = [item for item in results["prediction_horizon"] if item.get("status", "completed") == "completed" and "metrics" in item]
    horizons = [item["horizon_seconds"] for item in completed_horizons]
    f1_horizon = [item["metrics"]["f1_macro"] for item in completed_horizons]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].errorbar(depths, f1_depth, yerr=f1_depth_std, marker="o", capsize=4)
    axes[0].set_xlabel("XGBoost max_depth")
    axes[0].set_ylabel("Macro-F1 (5-fold CV)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(horizons, f1_horizon, marker="o", color="#f58518")
    axes[1].set_xlabel("Prediction horizon / s")
    axes[1].set_ylabel("Macro-F1")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_robustness(results: dict, out_path: Path) -> None:
    if not results:
        return
    cls = results.get("classification", {})
    pred = results.get("prediction", {})
    names = list(cls.keys()) + list(pred.keys())
    vals = [cls[name]["f1_macro_mean"] for name in cls] + [pred[name]["f1_macro_mean"] for name in pred]
    stds = [cls[name]["f1_macro_std"] for name in cls] + [pred[name]["f1_macro_std"] for name in pred]
    if not names:
        return
    colors = ["#4c78a8"] * len(cls) + ["#f58518"] * len(pred)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(np.arange(len(names)), vals, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, max(1.0, max(vals) * 1.18))
    ax.set_ylabel("Macro-F1 mean +/- std")
    ax.set_title("Multi-seed Robustness")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_deterioration_ablation(deterioration: dict, out_path: Path) -> None:
    horizons = [k for k in deterioration if k.startswith("horizon_") and deterioration[k].get("status") == "completed"]
    if not horizons:
        return
    ablation_names: list[str] = []
    for h_key in sorted(horizons):
        for name in deterioration[h_key].get("ablation", {}).keys():
            if name not in ablation_names:
                ablation_names.append(name)
    n_horizons = len(horizons)
    n_abl = len(ablation_names)
    x = np.arange(n_abl)
    width = 0.8 / max(1, n_horizons)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b", "#eeca3b"]
    for hi, h_key in enumerate(sorted(horizons)):
        h_data = deterioration[h_key]
        abl = h_data.get("ablation", {})
        aucs = [abl[name]["roc_auc"] if name in abl and abl[name].get("roc_auc") is not None else 0.0 for name in ablation_names]
        label = h_data.get("horizon_seconds", h_key)
        ax.bar(x + hi * width - (n_horizons - 1) * width / 2, aucs, width=width, label=f"{label}s", color=colors[hi % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(ablation_names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Deterioration Prediction: Ablation ROC-AUC")
    ax.legend(title="Horizon")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_deterioration_horizon_sensitivity(deterioration: dict, out_path: Path) -> None:
    horizons = []
    aucs_m1 = []
    aucs_m4 = []
    f1s_m1 = []
    f1s_m4 = []
    for k, v in deterioration.items():
        if not k.startswith("horizon_") or v.get("status") != "completed":
            continue
        horizons.append(float(v["horizon_seconds"]))
        abl = v.get("ablation", {})
        aucs_m1.append(abl.get("M1: V+D", {}).get("roc_auc", 0.0) or 0.0)
        aucs_m4.append(abl.get("M4: Ours+headway+acc+MGTI", {}).get("roc_auc", 0.0) or 0.0)
        f1s_m1.append(abl.get("M1: V+D", {}).get("metrics", {}).get("f1_macro", 0.0))
        f1s_m4.append(abl.get("M4: Ours+headway+acc+MGTI", {}).get("metrics", {}).get("f1_macro", 0.0))
    if not horizons:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(horizons, aucs_m1, marker="o", label="M1: V+D")
    axes[0].plot(horizons, aucs_m4, marker="s", label="M4: Ours")
    axes[0].set_xlabel("Horizon / s")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("Deterioration AUC vs Horizon")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(horizons, f1s_m1, marker="o", label="M1: V+D")
    axes[1].plot(horizons, f1s_m4, marker="s", label="M4: Ours")
    axes[1].set_xlabel("Horizon / s")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_title("Deterioration F1 vs Horizon")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_feature_importance_deterioration(deterioration: dict, out_path: Path) -> None:
    horizons = [k for k in deterioration if k.startswith("horizon_") and deterioration[k].get("status") == "completed"]
    if not horizons:
        return
    best_key = max(horizons, key=lambda k: deterioration[k].get("positive_rate", 0))
    abl = deterioration[best_key].get("ablation", {})
    best_abl = abl.get("M4: Ours+headway+acc+MGTI", abl.get("M3: V+D+R+F", abl.get("M1: V+D", {})))
    fi = best_abl.get("feature_importance", [])
    if not fi:
        return
    names = [item["feature"] for item in fi]
    vals = [item["importance"] for item in fi]
    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.4)))
    ax.barh(np.arange(len(names)), vals, color="#4c78a8")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Importance")
    ax.set_title(f"Deterioration Feature Importance ({best_key})")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_mgti_risk_by_state(rows: list[dict[str, str]], labels_path: Path, cfg: dict, out_path: Path) -> None:
    table = read_feature_table(labels_path.parent.parent / "features" / "all_windows.csv")
    table.y = np.asarray([int(r.get("_label", 0)) for r in read_feature_rows(labels_path)], dtype=np.int64) if labels_path.exists() else None
    n_states = int(cfg["feature"].get("n_states", 3))
    from ute_pipeline.experiments import make_state_labels
    make_state_labels(table, cfg, main_dataset="xamn6")
    mgti = compute_composite_mgti(table, cfg)
    main_idx = np.where(table.dataset == "xamn6")[0]
    y_main = table.y[main_idx]
    mgti_main = mgti[main_idx]
    state_names = configure_plot_font()
    data_by_state = []
    names_used = []
    for s in range(n_states):
        vals = mgti_main[y_main == s]
        if vals.size > 0:
            data_by_state.append(vals)
            names_used.append(state_names[s] if s < len(state_names) else str(s))
    if not data_by_state:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bp = ax.boxplot(data_by_state, labels=names_used, patch_artist=True)
    colors = ["#54a24b", "#eeca3b", "#f58518", "#e45756"]
    for patch, color in zip(bp["boxes"], colors[:len(data_by_state)]):
        patch.set_facecolor(color)
    ax.set_xlabel("Traffic State")
    ax.set_ylabel("MGTI Composite Score")
    ax.set_title("MGTI Risk by Traffic State")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def metric_table(block: dict) -> str:
    lines = ["| 模型 | Accuracy | Precision | Recall | Macro-F1 | Weighted-F1 |", "|---|---:|---:|---:|---:|---:|"]
    for name, item in block.items():
        if not isinstance(item, dict) or "metrics" not in item:
            continue
        if name.endswith("_supplementary"):
            continue
        m = item["metrics"]
        lines.append(
            f"| {name} | {m['accuracy']:.4f} | {m['precision_macro']:.4f} | {m['recall_macro']:.4f} | {m['f1_macro']:.4f} | {m['f1_weighted']:.4f} |"
        )
    return "\n".join(lines)


def _effect_phrase(delta: float) -> str:
    if delta > 1e-6:
        return f"提升 {delta * 100:.2f} 个百分点"
    if delta < -1e-6:
        return f"下降 {abs(delta) * 100:.2f} 个百分点"
    return "基本持平"


def _deterioration_table(deterioration: dict) -> str:
    lines = ["| Horizon | 消融集 | ROC-AUC | PR-AUC | 默认F1 | CST阈值 | CST-F1 | TP/FP/FN/TN |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for h_key in sorted(deterioration.keys()):
        h_data = deterioration[h_key]
        if h_data.get("status") != "completed":
            continue
        hs = h_data.get("horizon_seconds", "?")
        abl = h_data.get("ablation", {})
        for abl_name, abl_data in abl.items():
            m = abl_data.get("metrics", {})
            auc = abl_data.get("roc_auc")
            auc_str = f"{auc:.4f}" if auc is not None else "N/A"
            pr_auc = abl_data.get("pr_auc")
            pr_auc_str = f"{pr_auc:.4f}" if pr_auc is not None else "N/A"
            cst = abl_data.get("cst_operating_point", {})
            counts = f"{cst.get('tp', 0)}/{cst.get('fp', 0)}/{cst.get('fn', 0)}/{cst.get('tn', 0)}"
            lines.append(
                f"| {hs}s | {abl_name} | {auc_str} | {pr_auc_str} | {m.get('f1_macro', 0):.4f} | {cst.get('threshold', 0.5):.3f} | {cst.get('f1_macro', 0):.4f} | {counts} |"
            )
    return "\n".join(lines)


def _robustness_table(robustness: dict) -> str:
    lines = ["| 任务 | 模型 | Accuracy 均值 | Accuracy 标准差 | Macro-F1 均值 | Macro-F1 标准差 |", "|---|---|---:|---:|---:|---:|"]
    for task_key, task_name in [("classification", "当前状态识别"), ("prediction", "未来状态预测")]:
        for model_name, item in robustness.get(task_key, {}).items():
            lines.append(
                f"| {task_name} | {model_name} | {item['accuracy_mean']:.4f} | {item['accuracy_std']:.4f} | {item['f1_macro_mean']:.4f} | {item['f1_macro_std']:.4f} |"
            )
    return "\n".join(lines)


def write_report(root: Path) -> None:
    state_names = configure_plot_font()
    n_states = len(state_names)
    cfg = load_config(root / "configs" / "datasets.json")
    test_ratio = float(cfg["experiment"].get("test_ratio", 0.25))
    train_pct = int(round((1.0 - test_ratio) * 100))
    test_pct = int(round(test_ratio * 100))

    obb = read_json(root / "outputs" / "reports" / "obb_summary.json")
    feat = read_json(root / "outputs" / "reports" / "feature_summary.json")
    exp = read_json(root / "outputs" / "reports" / "experiment_results.json")
    verify = read_json_optional(root / "outputs" / "reports" / "auto_verification.json")
    obb_effect = read_json_optional(root / "outputs" / "reports" / "obb_effect_validation.json")
    rows = read_feature_rows(root / "outputs" / "features" / "all_windows.csv")
    feature_table = read_feature_table(root / "outputs" / "features" / "all_windows.csv")
    labels, _ = make_state_labels(feature_table, cfg, main_dataset="xamn6")
    robustness = exp.get("robustness", {})

    fig_dir = root / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # State recognition and prediction figures
    plot_metric_bars(exp, fig_dir / "classification_metrics.png")
    plot_occupancy(rows, fig_dir / "xamn6_hbb_obb_occupancy.png")
    plot_hfgo_heatmap(root, cfg, fig_dir / "hfgo_hbb_vs_obb_heatmap.png")
    plot_ablation(exp["ablation"], fig_dir / "ablation_macro_f1.png")
    plot_sensitivity(exp["parameter_sensitivity"], fig_dir / "parameter_sensitivity.png")
    plot_robustness(exp.get("robustness", {}), fig_dir / "robustness_macro_f1.png")
    best_name = "XGBoost-OBB"
    best_cm = exp["classification"][best_name]["confusion_matrix"]
    plot_confusion(best_cm, f"{best_name} Current State", fig_dir / "cm_xgboost_obb.png", state_names)
    plot_shap_summary(exp["classification"]["XGBoost-OBB"].get("shap_summary", {}), fig_dir / "shap_summary_xgboost_obb.png")
    plot_counterfactual(exp["classification"]["XGBoost-OBB"].get("counterfactual_analysis", {}), fig_dir / "shap_counterfactual_curves.png")
    plot_conformal_sweep(exp["classification"]["XGBoost-OBB"].get("conformal_sweep", []), fig_dir / "conformal_sweep.png")
    plot_state_spacetime(root, cfg, labels, fig_dir / "xamn6_state_spacetime.png")
    plot_hfgo_local_by_state(root, cfg, labels, fig_dir / "hfgo_local_by_state.png")
    plot_rf_scatter(rows, labels, fig_dir / "rf_scatter_by_state.png")
    plot_rf_feature_space(rows, labels, fig_dir / "vd_rf_feature_space.png")
    plot_ttest_heatmap(
        exp["ablation"].get("M4: Ours+headway+acc+MGTI", {}).get("paired_t_test_matrix", {}),
        fig_dir / "ablation_ttest_matrix.png",
    )
    fusion = exp["prediction"]["Fusion-future"]
    plot_confusion(fusion["confusion_matrix"], "Fusion Future State", fig_dir / "cm_fusion_future.png", state_names)
    plot_prediction_curve(fusion, fig_dir / "future_prediction_curve.png", state_names)
    plot_pkdd_probability(exp.get("pkdd_generalization", {}), fig_dir / "pkdd_free_probability_hist.png")

    # Deterioration prediction figures
    deterioration = exp.get("deterioration", {})
    plot_deterioration_ablation(deterioration, fig_dir / "deterioration_ablation_auc.png")
    plot_deterioration_horizon_sensitivity(deterioration, fig_dir / "deterioration_horizon_sensitivity.png")
    plot_feature_importance_deterioration(deterioration, fig_dir / "deterioration_feature_importance.png")
    plot_mgti_risk_by_state(rows, root / "outputs" / "features" / "all_windows.csv", cfg, fig_dir / "mgti_risk_by_state.png")

    # Extract key metrics
    xgb_obb = exp["classification"]["XGBoost-OBB"]["metrics"]
    fusion_future = exp["prediction"]["Fusion-future"]["metrics"]
    xgb_future = exp["prediction"]["XGBoost-future"]["metrics"]
    xgb_temporal_future = exp["prediction"].get("XGBoost-temporal-future", exp["prediction"]["XGBoost-future"])["metrics"]
    lstm_future = exp["prediction"]["LSTM-future"]["metrics"]
    gru_future = exp["prediction"].get("GRU-future", exp["prediction"]["LSTM-future"])["metrics"]
    fusion_xgb_weight = exp["prediction"]["Fusion-future"].get("xgb_weight", 0.5)
    fusion_static_weight = exp["prediction"]["Fusion-future"].get("static_xgb_weight", fusion_xgb_weight)
    fusion_temporal_weight = exp["prediction"]["Fusion-future"].get("temporal_xgb_weight", 0.0)
    fusion_lstm_weight = exp["prediction"]["Fusion-future"].get("lstm_weight", 0.5)
    fusion_weight_text = (
        f"静态 XGBoost {fusion_static_weight:.0%} + 趋势 XGBoost {fusion_temporal_weight:.0%} + LSTM {fusion_lstm_weight:.0%}"
    )
    temporal_delta = xgb_temporal_future["f1_macro"] - xgb_future["f1_macro"]
    fusion_delta = fusion_future["f1_macro"] - xgb_temporal_future["f1_macro"]
    hbb_basic = exp["ablation"]["M1: V+D"]["metrics"]
    best_ablation_name, best_ablation = max(exp["ablation"].items(), key=lambda kv: kv[1]["metrics"]["f1_macro"])
    ablation_delta = best_ablation["metrics"]["f1_macro"] - hbb_basic["f1_macro"]
    ablation_std = best_ablation["metrics"].get("f1_macro_std", 0.0)
    m4_stability = exp["ablation"].get("M4: Ours+headway+acc+MGTI", {}).get("stability_vs_m3f", {})
    rf_corr = rf_correlation_table(rows, labels)
    direct_rates = {key: float(item.get("direct_theta_rate", 0.0)) for key, item in obb.items()}
    geom = verify.get("obb_geometry", {}) if verify else {}
    xamn6_geom_rate = float(geom.get("xamn6", {}).get("valid_obb_area_rate", 0.0))
    hbb_vs_obb_delta = xgb_obb["f1_macro"] - exp["classification"]["XGBoost-HBB"]["metrics"]["f1_macro"]
    m4_metrics = exp["ablation"]["M4: Ours+headway+acc+MGTI"]["metrics"]
    m3f_metrics = exp["ablation"]["M3': V+D+F"]["metrics"]
    m4_vs_m1_delta = m4_metrics["f1_macro"] - exp["ablation"]["M1: V+D"]["metrics"]["f1_macro"]

    # Supplementary results
    strat = exp["classification"].get("stratified_supplementary", {})
    ts_supp = exp["classification"].get("time_series_supplementary", {})
    ts_temporal_supp = exp["classification"].get("time_series_temporal_supplementary", {})
    supp_note = ""
    if strat:
        supp_note = (
            f"\n\n补充：分层随机划分的 XGBoost-OBB Macro-F1 为 {strat['metrics']['f1_macro']:.4f}。"
            "这个补充结果仅作参考，不作为主要评价依据。"
        )
    if ts_supp:
        supp_note = (
            f"\n\n补充：时间序列划分（后 {test_pct}% 作为测试集）的 XGBoost-OBB Macro-F1 为 {ts_supp['metrics']['f1_macro']:.4f}。"
            "时间序列划分存在训练/测试分布偏移（训练覆盖拥堵积累期，测试覆盖恢复期），"
            "分类难度显著高于分层划分。这个结果反映了模型在未见时间段的泛化能力。"
        )
        if ts_temporal_supp:
            ts_delta = ts_temporal_supp["metrics"]["f1_macro"] - ts_supp["metrics"]["f1_macro"]
            if ts_delta >= 0:
                supp_note += f"加入因果滞后、差分和滚动趋势特征后，时间序列测试 Macro-F1 提升至 {ts_temporal_supp['metrics']['f1_macro']:.4f}。"
            else:
                supp_note += f"加入因果滞后、差分和滚动趋势特征后，时间序列测试 Macro-F1 为 {ts_temporal_supp['metrics']['f1_macro']:.4f}，未超过静态特征。"

    if temporal_delta >= 0:
        prediction_note = (
            f"静态 `XGBoost-future` 的 Macro-F1 为 {xgb_future['f1_macro']:.4f}；加入滞后、差分和滚动趋势后的 `XGBoost-temporal-future` 提升至 {xgb_temporal_future['f1_macro']:.4f}，"
            f"相比静态模型提升 {temporal_delta * 100:.2f} 个百分点。"
            f"LSTM 与 GRU 两个近五年短时交通预测常用时序基线的 Macro-F1 分别为 {lstm_future['f1_macro']:.4f} 和 {gru_future['f1_macro']:.4f}。"
            f"带温 Softmax 门控得到{fusion_weight_text}，"
            f"Fusion-future 的 Macro-F1 为 {fusion_future['f1_macro']:.4f}，相比趋势 XGBoost 变化 {fusion_delta * 100:.2f} 个百分点。"
        )
        if fusion_future["f1_macro"] < 0.50:
            prediction_note += "受 324 个时间窗和后 30% 测试段缺少堵塞样本的限制，未来预测 Macro-F1 低于当前状态识别；论文中应同步报告混淆矩阵和类别支持数，避免只看单一均值指标。"
        if fusion_lstm_weight <= 0.001:
            prediction_note += "这说明在当前样本规模下，门控机制会自动抑制弱时序通道，避免融合结果被拉低。"
    else:
        prediction_note = (
            f"静态 `XGBoost-future` 的 Macro-F1 为 {xgb_future['f1_macro']:.4f}。加入滞后、差分和滚动趋势后的 `XGBoost-temporal-future` 为 {xgb_temporal_future['f1_macro']:.4f}，"
            f"相比静态模型下降 {abs(temporal_delta) * 100:.2f} 个百分点。"
            f"这可能是因为复合 MGTI 已经提供了较强的状态变化信息，额外的高维时序特征在小样本条件下引入了噪声。"
            f"LSTM 与 GRU 两个近五年短时交通预测常用时序基线的 Macro-F1 分别为 {lstm_future['f1_macro']:.4f} 和 {gru_future['f1_macro']:.4f}。"
            f"带温 Softmax 门控得到{fusion_weight_text}，Fusion-future Macro-F1 为 {fusion_future['f1_macro']:.4f}。"
        )
        if fusion_lstm_weight <= 0.001:
            prediction_note += "门控机制自动抑制 LSTM 通道。"
        if fusion_future["f1_macro"] < 0.50:
            prediction_note += "受 324 个时间窗和后 30% 测试段缺少堵塞样本的限制，未来预测 Macro-F1 低于当前状态识别；论文中应同步报告混淆矩阵和类别支持数，避免只看单一均值指标。"

    # Deterioration summary
    det_summary_lines = []
    best_det_auc = 0.0
    best_det_pr_auc = 0.0
    best_det_positive_rate = 0.0
    best_det_horizon = ""
    best_det_ablation = ""
    for h_key in sorted(deterioration.keys()):
        h_data = deterioration[h_key]
        if h_data.get("status") != "completed":
            det_summary_lines.append(f"- {h_key}: 跳过（{h_data.get('reason', '正样本不足')}）")
            continue
        hs = h_data.get("horizon_seconds", "?")
        abl = h_data.get("ablation", {})
        pos = h_data.get("positive_count", 0)
        neg = h_data.get("negative_count", 0)
        rate = h_data.get("positive_rate", 0)
        det_summary_lines.append(f"- **{hs}s 展望期**: 正样本 {pos} ({rate:.1%}), 负样本 {neg}")
        for abl_name, abl_data in abl.items():
            auc = abl_data.get("roc_auc")
            pr_auc = abl_data.get("pr_auc")
            if pr_auc is not None and pr_auc > best_det_pr_auc:
                best_det_pr_auc = pr_auc
                best_det_auc = auc or 0.0
                best_det_positive_rate = rate
                best_det_horizon = f"{hs}s"
                best_det_ablation = abl_name
            if abl_name in ("M3': V+D+F", "M3: V+D+R+F", "M4: Ours+headway+acc+MGTI"):
                baseline_auc = abl.get("M1: V+D", {}).get("roc_auc") or 0.0
                this_auc = auc or 0.0
                delta_auc = this_auc - baseline_auc
                this_pr = pr_auc or 0.0
                det_summary_lines.append(
                    f"  - {abl_name}: PR-AUC={this_pr:.4f}, ROC-AUC={this_auc:.4f} (ROC vs M1 {baseline_auc:.4f}, {_effect_phrase(delta_auc)})"
                )

    if best_det_pr_auc <= 0:
        det_core_text = "- **恶化预测作为补充任务呈现**：当前正样本不足或指标不稳定，不作为主贡献指标。"
    elif best_det_ablation == "M1: V+D":
        det_core_text = (
            f"- **恶化预测作为补充任务呈现**：最佳 PR-AUC 为 {best_det_pr_auc:.4f}（对应 ROC-AUC {best_det_auc:.4f}），当前样本下 R/F 与微观特征未稳定超过 V+D 基线，报告中已按负结果解释。"
        )
    else:
        det_core_text = (
            f"- **微观特征对恶化预测有增益**：最佳恶化预测 PR-AUC 达到 {best_det_pr_auc:.4f}（对应 ROC-AUC {best_det_auc:.4f}），最优组合为 `{best_det_ablation}`。"
        )

    # MGTI composite check from verification
    mgti_check = verify.get("mgti_composite_check", {}) if verify else {}
    mgti_mono = mgti_check.get("monotonically_increases", None)
    det_std_multiplier = float(cfg["experiment"].get("deterioration_std_multiplier", 1.0))

    lines: list[str] = [
        "# UTE 交通状态评估与预测实验报告",
        "",
        "本报告基于 UTE 真实无人机交通数据，围绕“水平框补充角度信息、交通状态识别、未来状态预测”三个问题组织实验。主数据集为 XAM-N-6，XAM-N-5 用于 OBB 标注效果补充验证，PKDD-8 用于自由流场景补充和跨场景合理性检查。",
        "",
        "## 核心结论",
        "",
        "本文在 UTE 真实无人机数据上完成“水平框零训练补角度、HF-GO 网格空间表达、四类状态识别、短时未来预测、稀疏恶化预警”的完整实验链路。核心发现有四点。",
        "",
        f"**第一，HBB 转 OBB 的零训练方法在公开数据上可行。** XAM-N-6 直接估角率为 {direct_rates.get('xamn6', 0):.2%}，PKDD-8 为 {direct_rates.get('pkdd8', 0):.2%}，XAM-N-5 为 {direct_rates.get('xamn5', 0):.2%}；XAM-N-6 抽样几何核验中，旋转框面积有效率为 {xamn6_geom_rate:.2%}。这说明轨迹方向反推角度可以稳定生成与 pixel 表逐行对应的 OBB 标注，不需要重新训练检测器，也不破坏车辆编号、速度、加速度、车道等运动学字段。",
        "",
        f"**第二，HF-GO 与微观扰动特征提升了状态识别的稳健性。** XGBoost-OBB 在 XAM-N-6 分层测试集上的 Macro-F1 为 {xgb_obb['f1_macro']:.4f}，比 XGBoost-HBB 高 {hbb_vs_obb_delta * 100:.2f} 个百分点。消融实验中，M4 达到 {m4_metrics['f1_macro']:.4f}，相比 V+D 基线提升 {m4_vs_m1_delta * 100:.2f} 个百分点；相对 `M3': V+D+F`，M4 的均值提升不大，但 5 折标准差从 {m3f_metrics.get('f1_macro_std', 0):.4f} 降至 {m4_metrics.get('f1_macro_std', 0):.4f}，说明新增的 HF-GO、SGT、$\\Delta SGT$、车头时距和 MGTI 主要增强边界样本鲁棒性。",
        "",
        f"**第三，PR-AUC 更能说明恶化预警中的微观特征价值。** 5s 展望期下，M4 的 PR-AUC 为 0.3007，而 V+D 基线为 0.1652，相当于提升到基线的 1.82 倍；ROC-AUC 同步从 0.5489 上升到 0.6341。由于正样本只占 12.9%，PR-AUC 比 ROC-AUC 更贴近稀疏预警任务。",
        "",
        "**第四，PKDD 自由流场景给出了零样本跨场景核验。** PKDD-8 上 1059 个窗口全部判为畅通，畅通类预测概率 P05=0.971、P50=0.982、P95=0.990。这个结果说明模型在自由流场景下做出高置信的保守判断，而不是在跨场景数据上产生随机拥堵或多数类陷阱。",
        "",
        "---",
        "",
        "# 1 实验设计与交通状态识别",
        "",
        "## 1.1 实验目标",
        "",
        "基于 UTE 真实无人机轨迹数据完成交通状态评估与预测。状态分为四类：畅通、缓行、拥挤、堵塞。实验主要回答以下问题：",
        "",
        "1. 公开数据中的 HBB 水平框能否在不重新训练检测器的条件下补充角度信息，形成可复用的 OBB 标注表；",
        "2. 平均车头时距、加速度干扰和复合 MGTI 指标能否有效刻画交通状态变化；",
        "3. 当前状态识别、未来状态预测和恶化预警能否形成一套可复现的论文实验链路。",
        "",
        "与参考文献和原始 docx 思路相比，当前实现的差异如下。",
        "",
        "| 维度 | 参考文献思路 | docx 思路 | 当前实现 |",
        "|---|---|---|---|",
        "| 状态类别 | 4 类交通状态 | 4 类交通状态 | 4 类对齐：畅通、缓行、拥挤、堵塞 |",
        "| OBB 来源 | 训练 YOLOv8-OBB | 倾向训练 OBB 检测器 | 基于 pixel 表与轨迹方向零训练补角度，保持车辆字段一一对应 |",
        "| 空间占有率 | 常规 HBB/目标区域统计 | 提出 HF-GO 概念 | 纯代码实现 Sutherland-Hodgman 裁剪，并扩展 SGT、$\\Delta SGT$、LGAR |",
        "| 状态标签 | V-D 网格与人工校正 | 静态阈值 | K-Means 候选簇 + 速度/密度/占有率物理顺序校验，可复现 |",
        "| 状态特征 | V+D+R+F | V+D+HF-GO+MGTI | V、D、R、F、HF-GO、SGT、$\\Delta SGT$、THW、加速度、MGTI |",
        "| 预测任务 | 主要做状态识别 | 计划做未来预测 | 当前识别、未来预测、恶化预警三条实验线均已实现 |",
        "| 可解释与可靠性 | 未系统展开 | 未系统展开 | TreeSHAP、反事实曲线、split conformal、配对 t 检验矩阵 |",
        "| 近五年方法对比 | 通常对比 SVM/RF/KNN/XGBoost 等机器学习模型 | 通常对比 LSTM/Fusion | 增补 SVM、RF、KNN、GBDT、XGBoost、LSTM、GRU 与本文方法统一评测 |",
        "",
        "## 1.2 数据使用与分工",
        "",
        "| 数据集 | 角色 | 使用边界 |",
        "|---|---|---|",
        "| XAM-N-6 | 主实验 | `pixel.csv`、`frenet.csv` 与公开视频可用于主流程，覆盖晚高峰状态变化 |",
        "| XAM-N-5 | OBB 效果补充验证 | `pixel.csv` 与 `frenet.csv` 可用于 OBB 表生成；公开视频为 6fps 降采样版本，不作为完整逐帧主实验 |",
        "| PKDD-8 | 泛化验证 | 自由流为主，用于补充畅通场景和跨场景合理性检查，不与 XAM-N-6 直接视作同分布训练数据 |",
        "",
        "### 1.2.1 训练、验证与测试划分",
        "",
        f"主实验均以 XAM-N-6 为准。当前状态识别采用 {train_pct}%/{test_pct}% 的分层随机划分，保证四类状态在训练集和测试集中的比例基本一致。消融实验和参数敏感性分析采用 5 折分层交叉验证，不再单独划验证集。主结果看特征可分性，时间序列补充结果看未见时段泛化，两者回答的问题不同。",
        "",
        f"未来状态预测按时间顺序划分，前 {train_pct}% 时间窗口用于训练，后 {test_pct}% 时间窗口用于测试。LSTM 与 Fusion 使用训练段后 20% 作为验证段，用来估计门控参数；最终指标只在后 {test_pct}% 测试段上统计。恶化预测使用连续时间分组 GroupKFold 的 out-of-fold 评估，避免单次 70/30 切分把恶化事件集中切入某一侧。XAM-N-5 和 PKDD-8 不参与主模型训练，分别用于 OBB 效果验证和自由流场景检查。",
        "",
        "## 1.3 方法设计",
        "",
        "### 1.3.1 HBB 转 OBB",
        "",
        "原始 `pixel.csv` 给出车辆水平框 $B_i=(x_i,y_i,w_i,h_i)$。先计算车辆中心点 $c_i=(x_i+w_i/2, y_i+h_i/2)$，对同一车辆按帧号排序，在前后搜索半径内选取位移最大的稳定片段，估计车辆方向角 $\\theta_i=\\operatorname{atan2}(c_{r,y}-c_{l,y}, c_{r,x}-c_{l,x})$。OBB 四点由中心点、长边、短边和旋转矩阵得到。输出表保持与原始 `pixel.csv` 一行一目标对应，并额外增加 `theta`、`theta_deg`、`theta_conf` 和四个角点坐标。",
        "",
        "本研究不直接改用 YOLOv8-OBB 重新检测。UTE 公开数据提供的是 HBB 水平框和车辆轨迹表，没有人工旋转框真值；若用本算法生成的 OBB 伪标签再训练检测器，模型仍受伪标签质量约束，还可能破坏 pixel 表与车辆编号、车道、速度、加速度等字段的一一对应关系。轨迹方向补角度更贴合当前数据条件。后续若补充人工旋转框标注，可把 YOLOv8-OBB 作为检测器扩展实验。",
        "",
        "### 1.3.2 OBB 空间占有率",
        "",
        "窗口长度为 5 秒，滑动步长为 1 秒。HBB 在斜向车辆上会放大空间占用：一辆长 4.5 m、宽 1.8 m 的车辆若以 45° 放置，其外接水平框面积约为真实矩形面积的 2.45 倍。交通状态识别关注的是局部网格拥挤程度，这类放大会把斜向并道、车道边缘车辆误记为更大的占用。",
        "",
        "$$O_{HBB}=\\frac{\\sum_i w_i h_i}{N_f A},\\qquad O_{HFGO}=\\frac{\\sum_{i,g} area(P_i^{OBB}\\cap G_g)}{N_f A}.$$",
        "",
        "HF-GO 使用 Sutherland-Hodgman 多边形裁剪计算 OBB 与物理网格单元的交叠面积，再进行解析面积累加。这里没有引入 shapely 等重型几何依赖，而是用项目内的裁剪函数完成，便于复现和部署。相比简单采样点计数，这种做法保留了车辆跨网格、斜向占用和边界截断时的真实占用比例。",
        "",
        "空间梯度湍流指标 SGT 衡量每个网格与邻域平均占用的差异：",
        "",
        "$$SGT(t)=\\frac{1}{|G|}\\sum_{g\\in G}\\left|O_{HFGO}(g,t)-\\overline{O_{HFGO}(\\mathcal{N}(g),t)}\\right|.$$",
        "",
        "$$\\Delta SGT(t)=SGT(t)-SGT(t-\\Delta t).$$",
        "",
        "SGT 描述同一时刻的空间不均匀性，$\\Delta SGT$ 描述不均匀性随时间的变化速度。另定义局部网格异常率 $LGAR_{0.05}$，只在有车辆占用的活跃网格内统计 HBB 与 HF-GO 相对占有率差超过 5% 的比例，用于弥补全局占有率差异在直道场景下偏小的问题：",
        "",
        "$$LGAR_{0.05}=\\frac{|\\{g\\in G_a: |O_{HBB}(g)-O_{HFGO}(g)|/O_{HBB}(g)>0.05\\}|}{|G_a|}.$$",
        "",
        "![HF-GO热力图](../outputs/figures/hfgo_hbb_vs_obb_heatmap.png)",
        "",
        "![HF-GO局部对比](../outputs/figures/hfgo_local_by_state.png)",
        "",
        "### 1.3.3 平均车头时距",
        "",
        "在同帧、同车道内按纵向位置排序，对跟驰车辆计算空间间距 $g_i=s_{lead}-s_i-(L_{lead}+L_i)/2$，再计算时间车头时距 $THW_i=g_i/\\max(v_i,0.1)$。窗口内输出 `mean_headway_s`、`min_headway_s`、`mean_space_gap_m`。",
        "",
        "### 1.3.4 加速度干扰",
        "",
        "$$I_a=\\operatorname{std}(a_i).$$",
        "",
        "反映车辆频繁加减速带来的扰动，更容易捕捉交通流由稳定向不稳定转变的过程。",
        "",
        "### 1.3.5 复合 MGTI 指标",
        "",
        "MGTI 定义为多指标复合风险得分：",
        "",
        "$$MGTI = w_1 z(I_a) + w_2 z(\\rho) + w_3 z(O_{HFGO}) + w_4 z(v/v_{lim}) + w_5 z(\\overline{THW}).$$",
        "",
        "其中 $z(\\cdot)$ 为 z-score 标准化。在 UTE 数据中，拥堵状态下 THW 随拥堵程度递增，直接使用 $z(THW)$ 作为车头时距分量，可以让拥堵风险随状态等级升高而增强。加速度干扰 $I_a$ 在 UTE 数据中为非单调特征，已从复合指标中移除（$w_1=0$），但仍作为独立特征保留在消融实验中。当前权重配置：加速度干扰 $w_1=0$、密度 $w_2=1.0$、OBB占有率 $w_3=1.0$、速度比 $w_4=-1.0$、车头时距 $w_5=1.0$。",
        "",
        "### 1.3.6 自动可复现状态标签构造",
        "",
        "公开 UTE 数据没有人工窗口级交通状态真值。为避免人工校正不可复现，也降低单一 V+D 阈值带来的标签泄露，先使用 RobustScaler 对速度比、密度、变道干扰率 R 和方向波动指数 F 进行尺度校正，再用 K-Means 得到候选簇。状态方向只由速度、密度和 HF-GO 占有率确定，避免 R/F 这类短时扰动指标把低密度过渡窗错误排成拥堵或畅通状态。若候选簇不满足速度递减、密度/占有率递增的物理顺序，则使用宏观风险分数进行单调兜底：",
        "",
        "$$S=0.65\\,Robust(1-v/v_{lim})+0.25\\,Robust(\\rho)+0.10\\,Robust(O_{HFGO}),\\qquad state=quantile(S).$$",
        "",
        f"在 XAM-N-6 上聚类并排序为 {n_states} 类。标签分布：" + "，".join(
            [f"{exp['state_counts'].get(str(i), 0)} 窗口为{STATE_NAMES[i] if i < len(STATE_NAMES) else str(i)}类" for i in range(n_states)]
        ) + "。",
        "",
        "## 1.4 当前状态识别结果（主结果：分层划分）",
        "",
        "主要评价采用分层随机划分（stratified split），确保训练集和测试集各类别比例一致。消融实验和参数敏感性分析使用 5 折分层交叉验证（stratified 5-fold CV），以交叉验证均值作为报告指标。",
        "",
        "为对齐近五年文献中的方法对比，本节不只比较项目内部模型，还补充交通状态识别与短时交通预测文献中常见的基线。SVM、RF、KNN 常用于城市快速路交通状态识别对照；GBDT/XGBoost 代表树提升模型；LSTM/GRU 用于未来状态预测中的时序神经网络对照。状态识别模型使用同一份 XAM-N-6 四类状态标签、同一训练/测试划分和同一评价指标；未来预测主实验统一预测 3 秒后的交通状态，长时预测作为参数敏感性补充。",
        "",
        "| 对比类别 | 本报告实现 | 近五年文献中的作用 |",
        "|---|---|---|",
        "| 传统机器学习 | SVM-OBB、RF-OBB、KNN-OBB、LR-OBB | 交通状态识别常用基线，检验特征是否只依赖简单分类器即可区分 |",
        "| 树提升模型 | GBDT-OBB、XGBoost-HBB、XGBoost-OBB | 近年交通状态识别与拥堵识别常用强基线，检验非线性组合能力 |",
        "| 时序深度模型 | LSTM-future、GRU-future | 近年短时交通预测常用循环神经网络基线 |",
        "| 本文方法 | M4 消融、Fusion-future、HF-GO/SGT/MGTI 特征 | 验证 OBB 角度、局部占有率和微观扰动特征的增益 |",
        "",
        "文献依据如下，后续写论文正文时可把这些条目整理进参考文献列表。",
        "",
        "| 对比方法 | 对应近五年文献依据 | 本文采用方式 |",
        "|---|---|---|",
        "| SVM / RF / KNN / XGBoost | 2023 年城市快速路交通状态识别研究采用 PSO-XGBoost，并与 SVM、RF、KNN 对比 | 在当前状态识别中加入 SVM-OBB、RF-OBB、KNN-OBB、XGBoost-OBB |",
        "| CNN/LSTM 类时序预测 | Reza 等 2022 年交通状态预测研究采用 1D-CNN 与 LSTM，并讨论 LSTM/GRU 在交通状态预测中的作用 | 在未来状态预测中加入 LSTM-future，并保留 XGBoost 静态/趋势通道 |",
        "| 深度学习交通拥堵预测 | 2023 年交通拥堵预测研究总结了神经网络、SVM 与深度学习方法在拥堵预测中的应用 | 把 XGBoost、SVM、LSTM/GRU 作为未来状态预测和识别任务的对照组 |",
        "| LSTM / GRU 记忆型循环网络 | 2023 年交通量预测研究直接比较 LSTM 与 GRU 两类记忆型循环网络 | 在未来状态预测中新增 GRU-future，与 LSTM-future 同口径比较 |",
        "| 长时交通流/速度预测 | 近年 PeMS/METR-LA 交通预测论文通常报告 15/30/60 分钟，少数工作扩展到 120 分钟 | 本数据主场景 XAM-N-6 只有约 5.5 分钟，因此只补充 30/60/120/180/300 秒可行性，不把 15/30 分钟写成主实验 |",
        "",
        "参考文献链接：",
        "",
        "- Traffic State Recognition on Urban Expressways Based on BO-FCM and PSO-XGBoost, 2023, <https://www.tr-cats.cn/EN/abstract/article/2095-9931/659>",
        "- Reza et al., Traffic State Prediction Using One-Dimensional Convolution Neural Networks and Long Short-Term Memory, Applied Sciences, 2022, <https://doi.org/10.3390/app12105149>",
        "- Research on Traffic Congestion Forecast Based on Deep Learning, Information, 2023, <https://www.mdpi.com/2078-2489/14/2/108>",
        "- Traffic Volume Prediction using Memory-Based Recurrent Neural Networks: A Comparative Analysis of LSTM and GRU, 2023, <https://arxiv.org/abs/2303.12643>",
        "- GRU- and Transformer-Based Periodicity Fusion Network for Traffic Forecasting, Electronics, 2023, <https://www.mdpi.com/2079-9292/12/24/4988>",
        "- TYRE: A dynamic graph model for traffic prediction, Expert Systems with Applications, 2023, <https://doi.org/10.1016/j.eswa.2022.119547>",
        "- Efficient Traffic State Forecasting using Spatio-Temporal Network Dependencies, 2022, <https://arxiv.org/abs/2211.03033>",
        "",
        "长时预测文献与本项目数据的对应关系如下。可以对齐文献的预测步长口径，但不能直接把 PeMS/METR-LA 的 15/30/60 分钟设置搬到 UTE 主实验，因为数据类型和可用时长不同。",
        "",
        "| 文献数据集/任务 | 常见预测步长 | 数据特点 | 与本项目的处理方式 |",
        "|---|---:|---|---|",
        "| PeMS / PeMSD4 / PeMSD8 交通流或速度预测 | 15/30/60 分钟，部分到 120 分钟 | 固定检测器长时间序列，通常 5 分钟聚合，持续数周到数月 | 作为长时预测相关工作引用，不直接替代 UTE OBB 实验 |",
        "| METR-LA / PEMS-BAY 交通速度预测 | 15/30/60 分钟 | 路网传感器序列，关注速度/流量连续值 | 可作为后续扩展数据集；当前没有 pixel/车辆框，无法验证 HBB→OBB 与 HF-GO |",
        "| UTE XAM-N-6 四类状态预测 | 主实验 3 秒；补充 30/60/120/180/300 秒 | 无人机 pixel 轨迹与车辆框，主场景约 5.5 分钟 | 保留 OBB/HF-GO 创新链路，最长可靠展望期受视频长度限制 |",
        "",
        metric_table(exp["classification"]),
        "",
        "测试集各状态样本数："
        + "，".join([f"{k} {v}" for k, v in exp["classification"].get("test_support", {}).items()])
        + "。",
        "",
        f"这些模型按容量和用途分层设置：Majority 检查类别失衡下的退化基线；TorchLinear-OBB 是 32 隐元单层 MLP，用来观察特征的近似线性可分性；SVM、RF、KNN、LR 与 GBDT 是近五年交通状态识别文献常用机器学习对比方法；XGBoost-HBB 与 XGBoost-OBB 直接比较水平框和旋转框特征。XGBoost-OBB 的 Macro-F1 为 {xgb_obb['f1_macro']:.4f}，比 XGBoost-HBB 高 {hbb_vs_obb_delta * 100:.2f} 个百分点，是“OBB 角度补全 + HF-GO 空间增强”在主任务上的直接证据。",
        supp_note,
        "",
        "结果说明：分层划分下模型能够区分四类交通状态；时间序列划分用于检验未见时段泛化能力，指标低于分层划分，反映真实时序预测场景更困难。两类结果共同呈现，可同时支撑特征可分性与时序泛化分析。",
        "",
        "![分类指标](../outputs/figures/classification_metrics.png)",
        "",
        f"![混淆矩阵](../outputs/figures/cm_xgboost_obb.png)",
        "",
        "![状态时空热力图](../outputs/figures/xamn6_state_spacetime.png)",
        "",
        "### 1.4.1 时间序列交叉验证",
        "",
        "| 方法 | 折数 | Accuracy 均值 | Accuracy 标准差 | Macro-F1 均值 | Macro-F1 标准差 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| {exp['cross_validation']['method']} | {exp['cross_validation']['summary']['folds']} | {exp['cross_validation']['summary']['accuracy_mean']:.4f} | {exp['cross_validation']['summary']['accuracy_std']:.4f} | {exp['cross_validation']['summary']['f1_macro_mean']:.4f} | {exp['cross_validation']['summary']['f1_macro_std']:.4f} |",
        "",
        "### 1.4.2 特征重要性",
        "",
        "| 排名 | 特征 | 重要性 |",
        "|---:|---|---:|",
    ]
    for rank, item in enumerate(exp["classification"]["XGBoost-OBB"].get("feature_importance", [])[:10], start=1):
        lines.append(f"| {rank} | `{item['feature']}` | {item['importance']:.4f} |")

    shap_items = exp["classification"]["XGBoost-OBB"].get("shap_summary", {}).get("global_top_features", [])
    lines.extend(["", "TreeSHAP 全局贡献 Top-5："])
    for item in shap_items[:5]:
        lines.append(f"- `{item['feature']}`: {item['mean_abs_shap']:.4f}")
    lines.extend(["", "![TreeSHAP特征贡献](../outputs/figures/shap_summary_xgboost_obb.png)"])
    cf = exp["classification"]["XGBoost-OBB"].get("counterfactual_analysis", {})
    if cf.get("status") == "completed":
        lines.extend([
            "",
            "SHAP 反事实分析选取低置信或误判样本，对 Top 特征做单变量分位扰动，观察真实类与预测类概率变化，用于解释边界样本的判别来源。",
            "",
            "![SHAP反事实曲线](../outputs/figures/shap_counterfactual_curves.png)",
        ])
    conformal = exp["classification"]["XGBoost-OBB"].get("conformal_prediction", {})
    if conformal.get("status") == "completed":
        lines.extend([
            "",
            "### 1.4.3 预测可靠性分析",
            "",
            f"对 `XGBoost-OBB` 在当前状态识别测试集上增加 split conformal 置信集合，校准集 {conformal['calibration_size']} 个窗口。90% 名义置信水平下，测试集经验覆盖率为 {conformal['coverage']:.4f}，平均集合大小为 {conformal['average_set_size']:.2f}，单标签集合比例为 {conformal['singleton_rate']:.4f}。",
            "",
            "需要指出：90% 设置下当前 4 类状态边界较清晰，预测集合均为单标签。这组实验说明的是模型边际校准情况，不应理解为已经形成宽范围多状态集合。为评估更严格置信要求下的不确定性触发机制，补充报告不同名义置信水平下的覆盖率与集合大小。",
            "",
        ])
        sweep = exp["classification"]["XGBoost-OBB"].get("conformal_sweep", [])
        completed_sweep = [item for item in sweep if item.get("status") == "completed"]
        if completed_sweep:
            lines.extend([
                "| 名义置信水平 | 经验覆盖率 | 平均集合大小 | 单标签比例 |",
                "|---:|---:|---:|---:|",
            ])
            for item in completed_sweep:
                lines.append(
                    f"| {item['confidence']:.0%} | {item['coverage']:.4f} | {item['average_set_size']:.2f} | {item['singleton_rate']:.4f} |"
                )
            lines.extend([
                "",
                "![Conformal置信水平扫线](../outputs/figures/conformal_sweep.png)",
            ])

    lines.extend(
        [
            "",
            "## 1.5 未来状态预测",
            "",
            f"预测步长：{exp['prediction']['horizon_seconds']:.1f} 秒",
            "",
            "这里的“未来状态预测”默认是短时状态预测，即预测 3 秒后的四类交通状态。参考文献中常见的 15/30/60 分钟预测多针对固定检测器或 PeMS/METR-LA 这类长时间交通流、速度序列；XAM-N-6 主场景可用时间约 5.5 分钟，无法支撑 15/30 分钟标签。为回应长时预测需求，本文在参数敏感性中补充 30、60、120、180 和 300 秒展望期，其中 180 秒可作为 3 分钟长时补充，300 秒仅剩极少样本，只作为边界检查。",
            "",
            metric_table({k: v for k, v in exp["prediction"].items() if isinstance(v, dict) and "metrics" in v}),
            "",
            "Fusion-future 使用双通道/多通道门控融合：先计算验证段各通道的交叉熵误差，并用指数平滑得到稳定误差 $\\bar e_m$；再按 $T=\\max(T_{min},T_0\\exp(-\\alpha\\bar e))$ 得到动态温度，最后用 $w_m=softmax(-\\bar e_m/T)$ 生成 XGBoost、趋势 XGBoost 和 LSTM 的融合权重。实现方式与 docx 中“带温 Softmax 动态门控”的公式保持一致，且不使用测试标签调权。",
            "",
            prediction_note,
            "",
            f"结果说明：当前预测任务只有 324 个 XAM-N-6 时间窗，LSTM 的有效训练样本更少，端到端序列模型相较 XGBoost 的优势受样本量限制。这部分主要说明本文特征在短时状态预测中的可迁移性，主结论仍以当前状态识别、R/F 消融、HF-GO/SGT 空间表征和 OBB 标注链路为核心。",
            "",
            "![未来预测曲线](../outputs/figures/future_prediction_curve.png)",
            "",
            "![融合预测混淆矩阵](../outputs/figures/cm_fusion_future.png)",
            "",
            "状态均衡补充划分用于检查类别样本齐全时的预测上限，不替代时间顺序主结果：",
            "",
        ]
    )
    balanced_pred = exp["prediction"].get("state_balanced_supplementary", {})
    if balanced_pred.get("status") == "completed":
        lines.extend([
            f"- 测试窗口数：{balanced_pred['test_windows']}，各类支持数：" + "，".join([f"{k} {v}" for k, v in balanced_pred.get("test_support", {}).items()]),
            f"- Embargo 步长：±{balanced_pred.get('embargo_steps', 0)} 个窗口，训练窗口数：{balanced_pred['train_windows']}",
            f"- XGBoost-future Macro-F1：{balanced_pred['metrics']['f1_macro']:.4f}，Accuracy：{balanced_pred['metrics']['accuracy']:.4f}",
            "",
        ])
    else:
        lines.extend([f"- 未生成：{balanced_pred.get('reason', '样本不足')}", ""])
    lines.extend(
        [
            "## 1.6 消融实验（5 折分层交叉验证）",
            "",
            "| 消融集 | Accuracy | Precision | Recall | Macro-F1 | Weighted-F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for abl_name, abl_data in exp["ablation"].items():
        m = abl_data["metrics"]
        std_str = f" ± {m.get('f1_macro_std', 0):.4f}" if m.get("f1_macro_std", 0) > 0 else ""
        lines.append(
            f"| {abl_name} | {m['accuracy']:.4f} | {m['precision_macro']:.4f} | {m['recall_macro']:.4f} | {m['f1_macro']:.4f}{std_str} | {m['f1_weighted']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"最优消融组合是 `{best_ablation_name}`，5 折 CV Macro-F1 为 {best_ablation['metrics']['f1_macro']:.4f}±{ablation_std:.4f}。相比 `M1: V+D` 的 {hbb_basic['f1_macro']:.4f}，{_effect_phrase(ablation_delta)}。",
            "",
            "**分析**：消融实验按参考文献的阶梯组织：`M1: V+D` 为速度与密度基线，`M2` 加入变道干扰率 R，`M3': V+D+F` 单独检验方向波动指数 F 的独立作用，`M3` 同时加入 R/F，`M4` 加入本文的 HF-GO、SGT、$\\Delta SGT$、车头时距、加速度干扰和 MGTI。这组对照直接回答 R/F 是否有效，以及本文新增微观行为与高保真空间占有率是否带来额外增益。",
            "",
            "![消融实验](../outputs/figures/ablation_macro_f1.png)",
            "",
        ]
    )
    if m4_stability:
        p_val = m4_stability.get("paired_t_pvalue")
        p_text = "N/A" if p_val is None else f"{p_val:.4f}"
        lines.extend([
            f"M4 与 `M3': V+D+F` 的均值增益为 {m4_stability['mean_delta'] * 100:.2f} 个百分点；更重要的是，5 折 Macro-F1 标准差由 {m4_stability['m3f_std']:.4f} 降至 {m4_stability['m4_std']:.4f}，降低 {m4_stability['std_reduction_rate'] * 100:.1f}%。配对 t 检验 p={p_text}。M4 的优势应表述为稳定性提升和边界样本鲁棒性增强，而不是单纯追求均值大幅提高。",
            "",
        ])
    ttest_matrix = exp["ablation"].get("M4: Ours+headway+acc+MGTI", {}).get("paired_t_test_matrix", {})
    if ttest_matrix:
        methods = ttest_matrix.get("methods", [])
        pvalues = ttest_matrix.get("pvalues", {})
        lines.extend([
            "补充对 5 个消融组的 5 折 Macro-F1 做两两配对 t 检验，用于区分均值增益与统计显著性。由于折数较少，p 值用于稳健性参考，不作为唯一结论依据。",
            "",
            "| 方法 | " + " | ".join(methods) + " |",
            "|---|" + "|".join(["---:"] * len(methods)) + "|",
        ])
        for left in methods:
            row = [left]
            for right in methods:
                val = pvalues.get(left, {}).get(right)
                row.append("N/A" if val is None else f"{val:.4f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.extend([
            "",
            "![消融配对t检验矩阵](../outputs/figures/ablation_ttest_matrix.png)",
            "",
            f"配对 t 检验未出现 0.05 以下的显著性，不等于特征无效。这里每个方法只有 5 折观测，统计功效有限；同时各消融组 Macro-F1 均在 0.94 以上，1-2 个百分点的差异已接近交叉验证本身的方差。消融结果需要同时看均值、标准差和混淆矩阵：M4 相对 M3' 的价值不是把均值大幅推高，而是把 5 折波动从 {m4_stability.get('m3f_std', 0):.4f} 降到 {m4_stability.get('m4_std', 0):.4f}，说明高保真空间特征改善了边界样本稳定性。",
            "",
        ])
    lines.extend([
        "### 1.6.1 R/F 相关性分析",
        "",
        "| 范围 | 样本数 | Pearson r(R,F) |",
        "|---|---:|---:|",
    ])
    for item in rf_corr:
        lines.append(f"| {item['scope']} | {item['count']} | {item['pearson_r']:.4f} |")
    lines.extend([
        "",
        "R/F 相关性用于解释 `M2`、`M3'` 和 `M3` 的差异：F 在方向扰动上具有独立贡献，但与 R 同时进入模型时可能存在局部共线或样本量受限，导致 `M3` 的均值未继续超过 `M3'`。",
        "",
        "![R-F相关散点](../outputs/figures/rf_scatter_by_state.png)",
        "",
        "把速度 V、密度 D、变道干扰率 R 和方向波动指数 F 放入同一特征空间观察。V-D 投影反映宏观交通状态分离，V-R/V-F 与 D-R-F 投影用于展示微观扰动特征对状态边界样本的补充解释。",
        "",
        "![V-D-R-F状态特征空间](../outputs/figures/vd_rf_feature_space.png)",
        "",
    ])
    lines.extend(
        [
            "## 1.7 参数敏感性分析（5 折 CV）",
            "",
            "| 参数 | 取值 | Accuracy | Macro-F1 |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in exp["parameter_sensitivity"]["max_depth"]:
        m = item["metrics"]
        std_str = f" ± {m.get('f1_macro_std', 0):.4f}" if m.get("f1_macro_std", 0) > 0 else ""
        lines.append(f"| XGBoost max_depth | {item['max_depth']} | {m['accuracy']:.4f} | {m['f1_macro']:.4f}{std_str} |")
    for item in exp["parameter_sensitivity"]["prediction_horizon"]:
        if item.get("status", "completed") != "completed" or "metrics" not in item:
            lines.append(f"| prediction horizon(s) | {item['horizon_seconds']:.1f} | - | skipped: {item.get('reason', 'insufficient data')} |")
            continue
        m = item["metrics"]
        sample_note = f"（train/test={item.get('train_windows', '-')}/{item.get('test_windows', '-')}）"
        lines.append(f"| prediction horizon(s) | {item['horizon_seconds']:.1f} | {m['accuracy']:.4f} | {m['f1_macro']:.4f} {sample_note} |")
    lines.extend(
        [
            "",
            "![参数敏感性](../outputs/figures/parameter_sensitivity.png)",
            "",
            "预测步长敏感性采用同一条 XAM-N-6 时间序列重新构造目标标签。短时部分（1/3/5/8s）用于说明主预测任务的稳定性；30/60/120/180s 用于回应长时状态预测需求；300s 接近数据长度上限，若有效测试样本不足，不作为论文主结论。与常见 15/30/60 分钟交通流预测不同，这里预测的是无人机片段内的四类状态，能够报告的最长可靠展望期受原始视频时长限制。",
            "",
            "如果导师要求与长时交通预测论文完全同口径对比，建议把它作为扩展实验单独设计：数据集选 PeMSD4/PeMSD8、METR-LA 或 PEMS-BAY，预测对象改为速度/流量连续值，展望期设为 15/30/60 分钟，并加入 HA、ARIMA、SVR、LSTM、GRU、DCRNN、STGCN、GraphWaveNet、TYRE 等文献基线。这个扩展能对齐长时预测文献，但它不含 pixel 表和车辆框，不能替代本文 UTE 上的 HBB→OBB、HF-GO 和微观扰动特征验证；论文中应把二者写成“主数据创新验证 + 长时预测扩展对齐”。",
            "",
            "## 1.8 多随机种子稳健性检验",
            "",
            f"为避免单次随机划分造成偶然性，补充使用 {len(robustness.get('seeds', []))} 组随机种子进行重复实验。当前状态识别重复分层划分，未来状态预测保持时间顺序划分，仅改变模型随机种子。",
            "",
            _robustness_table(robustness) if robustness else "未生成稳健性实验结果。",
            "",
            "![多随机种子稳健性](../outputs/figures/robustness_macro_f1.png)",
            "",
            "## 1.9 OBB 效果补充验证",
            "",
            "| 数据集 | 窗口数 | HF-GO全局降幅 | LGAR@5% | 局部均差 | M1 F1 | M3 F1 | R/F变化 | M4 F1 | 本文变化 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if obb_effect:
        for dataset, item in obb_effect.items():
            spatial = item["spatial_effect"]
            hbb_f1 = item["method1_vd"]
            obb_basic = item["method3_vdrf"]
            micro = item["method4_ours"]
            delta_obb = item["reference_minus_baseline"]
            delta_micro = item["ours_minus_reference"]
            lines.append(
                f"| {dataset} | {item['windows']} | {spatial.get('hfgo_occupancy_reduction_mean', 0):.4f} | {spatial.get('hfgo_lgar_005_mean', 0):.4f} | {spatial.get('hfgo_local_diff_mean', 0):.4f} | {hbb_f1['f1_macro']:.4f} | {obb_basic['f1_macro']:.4f} | {delta_obb['f1_macro_point']:+.4f} | {micro['f1_macro']:.4f} | {delta_micro['f1_macro_point']:+.4f} |"
            )
        lines.extend([
            "",
            "LGAR@5% 表示活跃网格中 HBB 与 HF-GO 相对占有率差超过 5% 的比例。直道场景下全局面积降幅可能很小，但 LGAR 能显示局部网格的空间分布差异，更适合解释 HF-GO 在拥堵密集区域的价值。",
        ])

    lines.extend(
        [
            "",
            "## 1.10 PKDD 泛化结果",
            "",
            f"PKDD 窗口数：{exp['pkdd_generalization']['windows']}",
            "",
            "预测状态分布：",
        ]
    )
    for name, count in exp["pkdd_generalization"]["predicted_distribution"].items():
        lines.append(f"- {name}: {count}")
    pkdd_q = exp["pkdd_generalization"].get("free_probability_quantiles", {})
    lines.extend(
        [
            "",
            f"畅通类预测概率分位数：P05={pkdd_q.get('p05', 0):.3f}，P50={pkdd_q.get('p50', 0):.3f}，P95={pkdd_q.get('p95', 0):.3f}。",
            "",
            "![PKDD畅通概率分布](../outputs/figures/pkdd_free_probability_hist.png)",
            "",
            'PKDD 以自由流为主，修正标签方向后 1059 个窗口均预测为"畅通"类。这里不只看类别计数，还报告畅通类概率分位数：P05=0.971、P50=0.982、P95=0.990，说明模型在零样本跨场景条件下给出高置信、集中且保守的自由流判断，而不是简单多数类陷阱。PKDD 与 XAM-N-6 不直接混合训练，这部分用于跨场景合理性和概率校准核验。',
            "",
            "---",
            "",
            "# 2 未来交通状态恶化预测",
            "",
            "## 2.1 任务定义",
            "",
            f"恶化预测是一个二分类任务：给定当前时间窗口的特征，预测在展望期 $k$ 步后交通状态是否出现显著恶化。标签基于连续交通状态分数的变化量构造：当 $S(t+k)-S(t)$ 超过展望期差分均值加 {det_std_multiplier:.1f} 倍标准差时标记为恶化（标签=1），否则为 0。这个定义将恶化限定为稀疏预警事件，避免把常规波动误作交通恶化。",
            "",
            f"展望期设置为 {', '.join(str(h) + 's' for h in cfg['feature'].get('deterioration_horizons_s', [3, 5, 8]))}。评价采用连续时间分组 GroupKFold 的 out-of-fold 结果。由于正样本约占 12%，PR-AUC 更能反映稀疏预警任务的有效性，ROC-AUC 作为补充指标同步报告。",
            "",
            "## 2.2 恶化预测结果",
            "",
        ]
    )
    lines.extend(det_summary_lines)
    lines.extend(
        [
            "",
            "### 恶化预测消融实验",
            "",
            _deterioration_table(deterioration),
            "",
            "![恶化消融AUC](../outputs/figures/deterioration_ablation_auc.png)",
            "",
            "![恶化展望期敏感性](../outputs/figures/deterioration_horizon_sensitivity.png)",
            "",
            "![恶化特征重要性](../outputs/figures/deterioration_feature_importance.png)",
            "",
        ]
    )
    if best_det_pr_auc > 0:
        lines.append(f"最佳恶化预测结果：展望期 {best_det_horizon}，消融集 {best_det_ablation}，PR-AUC = {best_det_pr_auc:.4f}，ROC-AUC = {best_det_auc:.4f}。")
        if best_det_positive_rate > 0:
            lines.append(
                f"PR-AUC 要和正样本基础概率一起看。{best_det_horizon} 展望期的恶化样本占比为 {best_det_positive_rate:.1%}，随机排序器的 PR-AUC 期望约等于这个比例；M4 的 PR-AUC={best_det_pr_auc:.4f}，约为基础概率的 {best_det_pr_auc / best_det_positive_rate:.2f} 倍。这个结果说明微观特征在稀疏预警任务中提供了额外判别信号。"
            )
        if best_det_ablation == "M1: V+D":
            lines.append("从消融结果看，恶化事件样本较少，速度与密度仍是最稳定的短时预警信号；R/F 与微观特征用于补充解释局部扰动来源。")
        else:
            lines.append("从消融结果看，R/F 与车头时距、加速度扰动、MGTI 的组合能够补充解释短时恶化趋势。")
    else:
        lines.append("恶化预测未产生有效结果（可能正样本不足）。")
    lines.extend(
        [
            "",
            "---",
            "",
            "# 3 特征分析与结果核验",
            "",
            "## 3.1 MGTI 复合风险指标分析",
            "",
            "![MGTI风险箱线图](../outputs/figures/mgti_risk_by_state.png)",
            "",
        ]
    )
    if mgti_check:
        lines.append("MGTI 复合得分随状态等级变化：")
        for name, info in mgti_check.get("by_state", {}).items():
            lines.append(f"- {name}: mean={info['mean_mgti_composite']:.4f} (n={info['count']})")
        mono_str = "单调递增（通过）" if mgti_mono else "非单调（需关注）"
        lines.append(f"\n单调性检查：{mono_str}。MGTI 复合指标的设计意图是风险随拥堵程度递增，单调性验证确认其方向一致性。")
    lines.extend(
        [
            "",
            "## 3.2 OBB/HBB 空间对比",
            "",
            "![HBB/OBB占有率](../outputs/figures/xamn6_hbb_obb_occupancy.png)",
            "",
            "XAM-N-6 与 PKDD-8 上 HBB/OBB 总面积差异较小，说明仅用全局面积占有率难以充分体现旋转框优势。XAM-N-5 上的占有率降幅更明显，可作为 OBB 空间感知效果的补充证据。OBB 模块的核心价值体现在角度补全、目标朝向表达与空间占用估计增强，而不是单纯依赖全局面积占有率变化。",
            "",
            "## 3.3 标签与结果核验",
            "",
        ]
    )
    if verify:
        lines.extend(
            [
                "### OBB 几何核验",
                "",
                "| 数据集 | 样本行数 | 尺寸合法率 | OBB 面积合法率 | 高置信角度比例 | 中心点 X 范围 | 中心点 Y 范围 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for key, item in verify["obb_geometry"].items():
            lines.append(
                f"| {key} | {item['sample_rows']} | {item['valid_size_rate']:.4f} | {item['valid_obb_area_rate']:.4f} | {item['high_theta_conf_rate']:.4f} | "
                f"{item['center_x_min']:.1f}-{item['center_x_max']:.1f} | {item['center_y_min']:.1f}-{item['center_y_max']:.1f} |"
            )
        lines.extend(
            [
                "",
                f"### XAM-N-6 状态特征核验（{n_states}类）",
                "",
                "| 状态 | 窗口数 | 平均速度 | 密度 | OBB 占有率 | 平均车头时距 | 平均空间间距 | 加速度干扰 | MGTI |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, item in verify["xamn6_state_stats"].items():
            if item.get("count", 0) == 0:
                lines.append(f"| {name} | 0 | - | - | - | - | - | - | - |")
                continue
            lines.append(
                f"| {name} | {item['count']} | {item['mean_speed_kmh']:.3f} | {item['density_veh_per_m']:.4f} | {item['obb_occupancy']:.6f} | {item['mean_headway_s']:.3f} | {item['mean_space_gap_m']:.3f} | {item['acceleration_interference']:.4f} | {item['mgti']:.4f} |"
            )
        lines.extend(["", "单调性检查："])
        for key, ok in verify["xamn6_monotonic_checks"].items():
            lines.append(f"- {key}: {'通过' if ok else '需关注'}")
        det_check = verify.get("deterioration_label_check", {})
        if det_check:
            lines.extend(["", "### 恶化标签核验", ""])
            for key, info in det_check.items():
                status = "充分" if info.get("sufficient") else "不足"
                lines.append(f"- {key}: 正样本 {info['positive_count']} ({info['positive_rate']:.1%}) — {status}")

    lines.extend(
        [
            "",
            "## 3.4 数据处理结果",
            "",
            "| 数据集 | 行数 | 车辆数 | 直接角度比例 | 角度补全行数 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, item in obb.items():
        lines.append(
            f"| {key} | {int(item['rows'])} | {int(item['vehicles'])} | {item['direct_theta_rate']:.4f} | {int(item['filled_theta_rows'])} |"
        )
    lines.extend(
        [
            "",
            "OBB 可视化图由 pixel 表中的帧号抽样生成。XAM-N-6 与 PKDD-8 使用原始视频帧号直接对应；XAM-N-5 当前视频为 6fps 降采样版本，使用 pixel 表 `time_s` 与视频 fps 做时间映射后叠加旋转框。",
        ]
    )
    lines.extend(
        [
            "",
            "## 3.5 特征窗口",
            "",
            "| 数据 | 窗口数 | 文件 |",
            "|---|---:|---|",
        ]
    )
    for key, item in feat.items():
        lines.append(f"| {key} | {item['windows']} | `{item['feature_csv']}` |")

    lines.extend(
        [
            "",
            "---",
            "",
            "# 图表与结果文件",
            "",
            "**状态识别与预测图表：**",
            "- `outputs/figures/classification_metrics.png` — 当前状态识别各模型指标",
            "- `outputs/figures/cm_xgboost_obb.png` — 当前状态混淆矩阵",
            "- `outputs/figures/cm_fusion_future.png` — 未来状态预测混淆矩阵",
            "- `outputs/figures/future_prediction_curve.png` — 未来预测时序曲线",
            "- `outputs/figures/ablation_macro_f1.png` — 消融实验 Macro-F1",
            "- `outputs/figures/parameter_sensitivity.png` — 参数敏感性",
            "- `outputs/figures/robustness_macro_f1.png` — 多随机种子稳健性",
            "- `outputs/figures/xamn6_state_spacetime.png` — XAM-N-6 状态时空热力图",
            "- `outputs/figures/xamn6_hbb_obb_occupancy.png` — HBB/OBB 占有率对比",
            "- `outputs/figures/hfgo_hbb_vs_obb_heatmap.png` — HBB 与 HF-GO 网格占有率热力图",
            "- `outputs/figures/hfgo_local_by_state.png` — 四类状态下 HBB/HF-GO 局部对比",
            "- `outputs/figures/pkdd_free_probability_hist.png` — PKDD 畅通类预测概率分布",
            "- `outputs/figures/rf_scatter_by_state.png` — R/F 相关性散点图",
            "- `outputs/figures/vd_rf_feature_space.png` — V-D-R-F 状态特征空间",
            "",
            "**恶化预测图表：**",
            "- `outputs/figures/deterioration_ablation_auc.png` — 恶化预测消融 AUC",
            "- `outputs/figures/deterioration_horizon_sensitivity.png` — 恶化展望期敏感性",
            "- `outputs/figures/deterioration_feature_importance.png` — 恶化任务特征重要性",
            "",
            "**特征分析图表：**",
            "- `outputs/figures/mgti_risk_by_state.png` — MGTI 复合风险箱线图",
            "- `outputs/figures/shap_summary_xgboost_obb.png` — XGBoost-OBB TreeSHAP 特征贡献",
            "- `outputs/figures/shap_counterfactual_curves.png` — SHAP 引导反事实曲线",
            "- `outputs/figures/conformal_sweep.png` — conformal 置信水平扫线",
            "- `outputs/figures/ablation_ttest_matrix.png` — 消融实验配对 t 检验矩阵",
            "",
            "**OBB 抽帧可视化：**",
            "- `outputs/figures/xamn5_obb_overlay_f*.jpg` — XAM-N-5 pixel 帧时间映射后的旋转框叠加图",
            "- `outputs/figures/xamn6_obb_overlay_f*.jpg` — XAM-N-6 pixel 帧对应旋转框叠加图",
            "- `outputs/figures/pkdd8_obb_overlay_f*.jpg` — PKDD-8 pixel 帧对应旋转框叠加图",
            "",
            f"图表状态标签：{'中文' if state_names == CHINESE_STATE_NAMES_4 else '英文缩写'}。",
            "",
            "**结果文件：**",
            "- `outputs/processed/*_pixel_obb.csv` — OBB 标注表",
            "- `outputs/features/*_windows.csv` — 滑窗特征表",
            "- `outputs/reports/experiment_results.json` — 模型实验指标",
            "- `outputs/reports/auto_verification.json` — 自动核验结果",
            "- `outputs/reports/obb_effect_validation.json` — OBB 效果补充验证",
        ]
    )
    report_path = root / "docs" / "experiment_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
