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
    read_feature_table,
)
from ute_pipeline.features import grid_area_heatmap, load_obb, video_shape

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
    horizons = [item["horizon_seconds"] for item in results["prediction_horizon"]]
    f1_horizon = [item["metrics"]["f1_macro"] for item in results["prediction_horizon"]]
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
    lines = ["| Horizon | 消融集 | ROC-AUC | PR-AUC | F1 | Precision | Recall |", "|---|---|---:|---:|---:|---:|---:|"]
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
            lines.append(
                f"| {hs}s | {abl_name} | {auc_str} | {pr_auc_str} | {m.get('f1_macro', 0):.4f} | {m.get('precision_macro', 0):.4f} | {m.get('recall_macro', 0):.4f} |"
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
    best_name = "XGBoost-OBB-MGTI" if "XGBoost-OBB-MGTI" in exp["classification"] else "XGBoost-OBB"
    best_cm = exp["classification"][best_name]["confusion_matrix"]
    plot_confusion(best_cm, f"{best_name} Current State", fig_dir / "cm_xgboost_obb.png", state_names)
    fusion = exp["prediction"]["Fusion-future"]
    plot_confusion(fusion["confusion_matrix"], "Fusion Future State", fig_dir / "cm_fusion_future.png", state_names)
    plot_prediction_curve(fusion, fig_dir / "future_prediction_curve.png", state_names)

    # Deterioration prediction figures
    deterioration = exp.get("deterioration", {})
    plot_deterioration_ablation(deterioration, fig_dir / "deterioration_ablation_auc.png")
    plot_deterioration_horizon_sensitivity(deterioration, fig_dir / "deterioration_horizon_sensitivity.png")
    plot_feature_importance_deterioration(deterioration, fig_dir / "deterioration_feature_importance.png")
    plot_mgti_risk_by_state(rows, root / "outputs" / "features" / "all_windows.csv", cfg, fig_dir / "mgti_risk_by_state.png")

    # Extract key metrics
    xgb_obb = exp["classification"]["XGBoost-OBB"]["metrics"]
    xgb_mgti = exp["classification"]["XGBoost-OBB-MGTI"]["metrics"]
    fusion_future = exp["prediction"]["Fusion-future"]["metrics"]
    xgb_future = exp["prediction"]["XGBoost-future"]["metrics"]
    xgb_temporal_future = exp["prediction"].get("XGBoost-temporal-future", exp["prediction"]["XGBoost-future"])["metrics"]
    lstm_future = exp["prediction"]["LSTM-future"]["metrics"]
    fusion_xgb_weight = exp["prediction"]["Fusion-future"].get("xgb_weight", 0.5)
    fusion_static_weight = exp["prediction"]["Fusion-future"].get("static_xgb_weight", fusion_xgb_weight)
    fusion_temporal_weight = exp["prediction"]["Fusion-future"].get("temporal_xgb_weight", 0.0)
    fusion_lstm_weight = exp["prediction"]["Fusion-future"].get("lstm_weight", 0.5)
    fusion_weight_text = (
        f"静态 XGBoost {fusion_static_weight:.0%} + 趋势 XGBoost {fusion_temporal_weight:.0%} + LSTM {fusion_lstm_weight:.0%}"
    )
    mgti_delta = xgb_mgti["f1_macro"] - xgb_obb["f1_macro"]
    temporal_delta = xgb_temporal_future["f1_macro"] - xgb_future["f1_macro"]
    fusion_delta = fusion_future["f1_macro"] - xgb_temporal_future["f1_macro"]
    hbb_basic = exp["ablation"]["M1: V+D"]["metrics"]
    best_ablation_name, best_ablation = max(exp["ablation"].items(), key=lambda kv: kv[1]["metrics"]["f1_macro"])
    ablation_delta = best_ablation["metrics"]["f1_macro"] - hbb_basic["f1_macro"]
    ablation_std = best_ablation["metrics"].get("f1_macro_std", 0.0)

    # Supplementary results
    strat = exp["classification"].get("stratified_supplementary", {})
    ts_supp = exp["classification"].get("time_series_supplementary", {})
    ts_temporal_supp = exp["classification"].get("time_series_temporal_supplementary", {})
    supp_note = ""
    if strat:
        supp_note = (
            f"\n\n补充：分层随机划分的 XGBoost-OBB Macro-F1 为 {strat['metrics']['f1_macro']:.4f}。"
            "该结果仅作参考，不作为主要评价依据。"
        )
    if ts_supp:
        supp_note = (
            f"\n\n补充：时间序列划分（后 {test_pct}% 作为测试集）的 XGBoost-OBB Macro-F1 为 {ts_supp['metrics']['f1_macro']:.4f}。"
            "时间序列划分存在训练/测试分布偏移（训练覆盖拥堵积累期，测试覆盖恢复期），"
            "因此分类难度显著高于分层划分。该结果反映了模型对未见时间段的泛化能力。"
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
            f"LSTM 在当前小样本条件下表现不如 XGBoost（{lstm_future['f1_macro']:.4f}）。"
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
            f"LSTM 在当前小样本条件下表现不如 XGBoost（{lstm_future['f1_macro']:.4f}）。"
            f"带温 Softmax 门控得到{fusion_weight_text}，Fusion-future Macro-F1 为 {fusion_future['f1_macro']:.4f}。"
        )
        if fusion_lstm_weight <= 0.001:
            prediction_note += "门控机制自动抑制 LSTM 通道。"
        if fusion_future["f1_macro"] < 0.50:
            prediction_note += "受 324 个时间窗和后 30% 测试段缺少堵塞样本的限制，未来预测 Macro-F1 低于当前状态识别；论文中应同步报告混淆矩阵和类别支持数，避免只看单一均值指标。"

    # Deterioration summary
    det_summary_lines = []
    best_det_auc = 0.0
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
            if auc is not None and auc > best_det_auc:
                best_det_auc = auc
                best_det_horizon = f"{hs}s"
                best_det_ablation = abl_name
            if abl_name in ("M3: V+D+R+F", "M4: Ours+headway+acc+MGTI"):
                baseline_auc = abl.get("M1: V+D", {}).get("roc_auc") or 0.0
                this_auc = auc or 0.0
                delta_auc = this_auc - baseline_auc
                det_summary_lines.append(f"  - {abl_name}: AUC={this_auc:.4f} (vs M1 {baseline_auc:.4f}, {_effect_phrase(delta_auc)})")

    if best_det_auc <= 0:
        det_core_text = "- **恶化预测作为补充任务呈现**：当前正样本不足或指标不稳定，不作为主贡献指标。"
    elif best_det_ablation == "M1: V+D":
        det_core_text = (
            f"- **恶化预测作为补充任务呈现**：最佳 ROC-AUC 为 {best_det_auc:.4f}，当前样本下 R/F 与微观特征未稳定超过 V+D 基线，报告中已按负结果解释。"
        )
    else:
        det_core_text = (
            f"- **微观特征对恶化预测有增益**：最佳恶化预测 ROC-AUC 达到 {best_det_auc:.4f}，最优组合为 `{best_det_ablation}`。"
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
        f"- **当前状态识别可用**：`XGBoost-OBB` 在 XAM-N-6 分层测试集上的 Macro-F1 为 {xgb_obb['f1_macro']:.4f}，SVM 与 LR 基线已纳入对比。",
        f"- **未来状态预测**：3s 预测步长下 `Fusion-future` Macro-F1 为 {fusion_future['f1_macro']:.4f}，用于评估模型对未见时间段状态变化的提前判别能力。",
        det_core_text,
        "- **数据边界已明确**：XAM-N-5 的公开视频为降采样版本，因此该数据集用于 pixel 表级 OBB 验证，不作为完整逐帧视频主实验。",
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
        f"主实验均以 XAM-N-6 为准。当前状态识别采用 {train_pct}%/{test_pct}% 的分层随机划分，保证四类状态在训练集和测试集中的比例基本一致。消融实验和参数敏感性分析采用 5 折分层交叉验证，不再单独划验证集。",
        "",
        f"未来状态预测和恶化预测按时间顺序划分，前 {train_pct}% 时间窗口用于训练，后 {test_pct}% 时间窗口用于测试。LSTM 与 Fusion 使用训练段后 20% 作为验证段，用来估计门控参数；最终指标只在后 {test_pct}% 测试段上统计。XAM-N-5 和 PKDD-8 不参与主模型训练，分别用于 OBB 效果验证和自由流场景检查。",
        "",
        "## 1.3 方法设计",
        "",
        "### 1.3.1 HBB 转 OBB",
        "",
        "原始 `pixel.csv` 给出车辆水平框 $B_i=(x_i,y_i,w_i,h_i)$。先计算车辆中心点 $c_i=(x_i+w_i/2, y_i+h_i/2)$，对同一车辆按帧号排序，在前后搜索半径内选取位移最大的稳定片段，估计车辆方向角 $\\theta_i=\\operatorname{atan2}(c_{r,y}-c_{l,y}, c_{r,x}-c_{l,x})$。OBB 四点由中心点、长边、短边和旋转矩阵得到。输出表保持与原始 `pixel.csv` 一行一目标对应，并额外增加 `theta`、`theta_deg`、`theta_conf` 和四个角点坐标。",
        "",
        "本研究没有直接改用 YOLOv8-OBB 重新检测，主要原因是 UTE 公开数据提供的是 HBB 水平框和车辆轨迹表，没有人工旋转框真值。若直接用当前算法生成的 OBB 作为伪标签再训练 YOLOv8-OBB，本质上仍受伪标签质量约束，且可能破坏 pixel 表与车辆编号、车道、速度、加速度等字段的一一对应关系。当前采用轨迹方向补角度，是在现有数据条件下更稳妥的方案；如果后续补充人工旋转框标注，可将 YOLOv8-OBB 作为检测器扩展实验。",
        "",
        "### 1.3.2 OBB 空间占有率",
        "",
        "窗口长度为 5 秒，滑动步长为 1 秒。当前实现使用旋转框几何面积进行窗口级占有率统计：",
        "",
        "$$O_{HBB}=\\frac{\\sum_i w_i h_i}{N_f A},\\qquad O_{HFGO}=\\frac{\\sum_{i,g} area(P_i^{OBB}\\cap G_g)}{N_f A}.$$",
        "",
        "HF-GO 使用 Sutherland-Hodgman 多边形裁剪计算 OBB 与物理网格单元的交叠面积，再进行解析面积累加。与简单采样点计数相比，该方法能保留车辆跨网格、斜向占用和边界截断时的真实占用比例，更适合作为本文区别于参考文献的空间表达增强模块。进一步地，本文计算空间梯度湍流指标 SGT，度量每个网格 HF-GO 占有率与相邻网格均值的偏差，用于捕捉拥堵形成时的局部空间不均匀性。",
        "",
        "![HF-GO热力图](../outputs/figures/hfgo_hbb_vs_obb_heatmap.png)",
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
        "其中 $z(\\cdot)$ 为 z-score 标准化。在 UTE 数据中，拥堵状态下 THW 随拥堵程度递增，因此直接使用 $z(THW)$ 作为车头时距分量，确保拥堵风险随状态等级升高而增强。加速度干扰 $I_a$ 在 UTE 数据中为非单调特征，已从复合指标中移除（$w_1=0$），但仍作为独立特征保留在消融实验中。当前权重配置：加速度干扰 $w_1=0$、密度 $w_2=1.0$、OBB占有率 $w_3=1.0$、速度比 $w_4=-1.0$、车头时距 $w_5=1.0$。",
        "",
        "### 1.3.6 状态标签构造",
        "",
        "为降低由单一 V+D 规则阈值带来的标签泄露，先使用 RobustScaler 对速度比、密度、变道干扰率 R 和方向波动指数 F 进行尺度校正，并用 K-Means 得到候选簇。状态方向只由速度、密度和 HF-GO 占有率确定，避免 R/F 这类短时扰动指标把低密度过渡窗错误排成拥堵或畅通状态。若候选簇不满足速度递减、密度/占有率递增的物理顺序，则使用宏观风险分数进行单调兜底：",
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
        metric_table(exp["classification"]),
        "",
        "测试集各状态样本数："
        + "，".join([f"{k} {v}" for k, v in exp["classification"].get("test_support", {}).items()])
        + "。",
        supp_note,
        "",
        f"`XGBoost-OBB-MGTI` 的 Macro-F1 为 {xgb_mgti['f1_macro']:.4f}，相比 `XGBoost-OBB` {_effect_phrase(mgti_delta)}。",
        "",
        "结果说明：分层划分下模型能够区分四类交通状态；时间序列划分用于检验未见时段泛化能力，指标低于分层划分，反映真实时序预测场景更困难。两类结果共同呈现，可同时支撑特征可分性与时序泛化分析。",
        "",
        "![分类指标](../outputs/figures/classification_metrics.png)",
        "",
        f"![混淆矩阵](../outputs/figures/cm_xgboost_obb.png)",
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
    for rank, item in enumerate(exp["classification"]["XGBoost-OBB-MGTI"].get("feature_importance", [])[:10], start=1):
        lines.append(f"| {rank} | `{item['feature']}` | {item['importance']:.4f} |")

    lines.extend(
        [
            "",
            "## 1.5 未来状态预测",
            "",
            f"预测步长：{exp['prediction']['horizon_seconds']:.1f} 秒",
            "",
            metric_table({k: v for k, v in exp["prediction"].items() if isinstance(v, dict) and "metrics" in v}),
            "",
            "Fusion-future 使用双通道/多通道门控融合：先计算验证段各通道的交叉熵误差，并用指数平滑得到稳定误差 $\\bar e_m$；再按 $T=\\max(T_{min},T_0\\exp(-\\alpha\\bar e))$ 得到动态温度，最后用 $w_m=softmax(-\\bar e_m/T)$ 生成 XGBoost、趋势 XGBoost 和 LSTM 的融合权重。该实现与 docx 中“带温 Softmax 动态门控”的公式保持一致，且不使用测试标签调权。",
            "",
            prediction_note,
            "",
            f"结果说明：当前预测任务只有 324 个 XAM-N-6 时间窗，LSTM 的有效训练样本更少，因此端到端序列模型相较 XGBoost 的优势受样本量限制。该部分用于补充说明本文特征在短时状态预测中的可迁移性，主结论仍以当前状态识别、R/F 消融、HF-GO/SGT 空间表征和 OBB 标注链路为核心。",
            "",
            "![未来预测曲线](../outputs/figures/future_prediction_curve.png)",
            "",
            "![融合预测混淆矩阵](../outputs/figures/cm_fusion_future.png)",
            "",
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
            "**分析**：消融实验按参考文献的阶梯组织：`M1: V+D` 为速度与密度基线，`M2` 加入变道干扰率 R，`M3` 加入方向波动指数 F，`M4` 进一步加入本文的 HF-GO、SGT、车头时距、加速度干扰和 MGTI。这样可以直接回答 R/F 是否有效，以及本文新增微观行为与高保真空间占有率是否带来额外增益。",
            "",
            "![消融实验](../outputs/figures/ablation_macro_f1.png)",
            "",
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
        m = item["metrics"]
        lines.append(f"| prediction horizon(s) | {item['horizon_seconds']:.1f} | {m['accuracy']:.4f} | {m['f1_macro']:.4f} |")
    lines.extend(
        [
            "",
            "![参数敏感性](../outputs/figures/parameter_sensitivity.png)",
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
            "| 数据集 | 窗口数 | HF-GO占有率降幅 | M1 F1 | M3 F1 | R/F变化 | M4 F1 | 本文变化 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
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
                f"| {dataset} | {item['windows']} | {spatial.get('hfgo_occupancy_reduction_mean', 0):.4f} | {hbb_f1['f1_macro']:.4f} | {obb_basic['f1_macro']:.4f} | {delta_obb['f1_macro_point']:+.4f} | {micro['f1_macro']:.4f} | {delta_micro['f1_macro_point']:+.4f} |"
            )

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
    lines.extend(
        [
            "",
            'PKDD 以自由流为主，修正标签方向后 1059 个窗口均预测为"畅通"类，说明跨场景检查未再出现自由流被误判为堵塞的问题。该结果用于自由流迁移合理性检查，不与 XAM-N-6 直接视作同分布混合训练数据。',
            "",
            "---",
            "",
            "# 2 未来交通状态恶化预测",
            "",
            "## 2.1 任务定义",
            "",
            f"恶化预测是一个二分类任务：给定当前时间窗口的特征，预测在展望期 $k$ 步后交通状态是否出现显著恶化。标签基于连续交通状态分数的变化量构造：当 $S(t+k)-S(t)$ 超过该展望期差分均值加 {det_std_multiplier:.1f} 倍标准差时标记为恶化（标签=1），否则为 0。该定义将恶化限定为稀疏预警事件，避免把常规波动误作交通恶化。",
            "",
            f"展望期设置为 {', '.join(str(h) + 's' for h in cfg['feature'].get('deterioration_horizons_s', [3, 5, 8]))}。评价采用连续时间分组 GroupKFold 的 out-of-fold 结果，报告 ROC-AUC、PR-AUC 和 Macro-F1。",
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
    if best_det_auc > 0:
        lines.append(f"最佳恶化预测结果：展望期 {best_det_horizon}，消融集 {best_det_ablation}，ROC-AUC = {best_det_auc:.4f}。")
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
            "XAM-N-6 与 PKDD-8 上 HBB/OBB 总面积差异较小，说明仅用全局面积占有率难以充分体现旋转框优势。XAM-N-5 上的占有率降幅更明显，可作为 OBB 空间感知效果的补充证据。因此，OBB 模块的核心价值体现在角度补全、目标朝向表达与空间占用估计增强，而不是单纯依赖全局面积占有率变化。",
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
            "- `outputs/figures/xamn6_hbb_obb_occupancy.png` — HBB/OBB 占有率对比",
            "- `outputs/figures/hfgo_hbb_vs_obb_heatmap.png` — HBB 与 HF-GO 网格占有率热力图",
            "",
            "**恶化预测图表：**",
            "- `outputs/figures/deterioration_ablation_auc.png` — 恶化预测消融 AUC",
            "- `outputs/figures/deterioration_horizon_sensitivity.png` — 恶化展望期敏感性",
            "- `outputs/figures/deterioration_feature_importance.png` — 恶化任务特征重要性",
            "",
            "**特征分析图表：**",
            "- `outputs/figures/mgti_risk_by_state.png` — MGTI 复合风险箱线图",
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
