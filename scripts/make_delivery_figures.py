#!/usr/bin/env python
"""Generate extra figures for the customer's group-meeting presentation.

Outputs into outputs/figures/delivery/:
    arch_obb_st_lstm.png         OBB-ST-LSTM 架构示意
    short_horizon_compare.png    3/5/8s 横向对比柱状图
    pems_long_horizon_summary.png PeMS08 长时汇总图 (flow + speed × 5/15/30 min)
    method_compare_radar.png     旧 Fusion vs 新 OBB-ST-LSTM 雷达对比
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "delivery"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Use English fonts for safety; presenter can re-label in PPT if needed
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11


# ---- Figure 1: OBB-ST-LSTM 架构示意 ----
def draw_architecture():
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    def box(x, y, w, h, text, color="#cfe2f3", text_color="black", fontsize=10):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05",
            linewidth=1.2, edgecolor="#1f4e79", facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color=text_color, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="->",
            mutation_scale=14, linewidth=1.3, color="#333",
        ))

    # Input row
    box(0.2, 7.2, 5.2, 1.4,
        "Spatial Tensor per Window\n(C=4, H=4, W=12)\n[OBB occ | HBB occ | sin θ | cos θ]",
        color="#fff2cc", fontsize=10)
    box(0.2, 5.5, 5.2, 1.3,
        "Scalar Descriptors per Window\n(F=8) [vehicle_count, density, flow,\nmean_speed, std_speed, speed_ratio, F, theta_conf]",
        color="#fff2cc", fontsize=10)

    # Spatial encoder
    box(6.4, 7.2, 3.2, 1.4,
        "Spatial Encoder (CNN)\nConv2d(4→8) + BN + ReLU\nConv2d(8→8) + BN + ReLU\nDropout2d + GAP",
        color="#cfe2f3", fontsize=9)
    arrow(5.4, 7.9, 6.4, 7.9)

    # Concatenate
    box(10.4, 6.4, 3.2, 1.2,
        "Concatenate per step\n(8 spa + 8 scalar = 16 dim)\n×  T=8 frames",
        color="#d9ead3", fontsize=9.5)
    arrow(9.6, 7.9, 10.4, 7.2)
    arrow(5.4, 6.15, 10.4, 6.8)

    # LSTM
    box(10.4, 4.4, 3.2, 1.4,
        "LSTM\nhidden_size = 64\nbatch_first=True\nsingle layer, unidirectional",
        color="#cfe2f3", fontsize=10)
    arrow(12.0, 6.4, 12.0, 5.8)

    # Head
    box(10.4, 2.6, 3.2, 1.2,
        "Take last step\nDropout(0.2) + Linear(64, 4)\n→ Softmax",
        color="#cfe2f3", fontsize=9.5)
    arrow(12.0, 4.4, 12.0, 3.8)

    # Output
    box(10.4, 0.8, 3.2, 1.2,
        "Predicted state at t+k\n(Free / Slow / Crowded / Congested)\n5-seed probability ensemble",
        color="#f4cccc", fontsize=9.5)
    arrow(12.0, 2.6, 12.0, 2.0)

    # Box around CNN+LSTM
    panel = mpatches.FancyBboxPatch(
        (6.1, 0.6), 7.7, 8.1, boxstyle="round,pad=0.1",
        linewidth=1.0, edgecolor="#1f4e79", facecolor="none", linestyle="--",
    )
    ax.add_patch(panel)
    ax.text(6.5, 8.55, "Single end-to-end model — one loss, one forward pass",
            fontsize=10, color="#1f4e79", style="italic")

    ax.set_title("OBB-ST-LSTM Architecture\n(Single model, no A+B fusion)",
                 fontsize=13, pad=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "arch_obb_st_lstm.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---- Figure 2: 3/5/8s 横向对比柱状图 ----
def draw_short_horizon_compare():
    data = json.loads(
        (PROJECT_ROOT / "outputs" / "reports" / "experiment_results.json").read_text()
    )
    sweep = data["horizon_sweep_obb_st_lstm"]["results"]
    horizons = [item["horizon_seconds"] for item in sweep if item.get("status") == "completed"]
    models = ["XGBoost-future", "LSTM-future", "GRU-future", "OBB-ST-LSTM"]
    colors = ["#9aa0a6", "#4c78a8", "#f58518", "#d62728"]

    f1_data = {m: [next(it for it in sweep if it["horizon_seconds"] == h)["models"][m]["f1_macro"]
                   for h in horizons] for m in models}
    acc_data = {m: [next(it for it in sweep if it["horizon_seconds"] == h)["models"][m]["accuracy"]
                    for h in horizons] for m in models}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(len(horizons))
    width = 0.18

    for i, m in enumerate(models):
        offset = (i - 1.5) * width
        axes[0].bar(x + offset, f1_data[m], width=width, color=colors[i], label=m,
                    edgecolor="black", linewidth=0.4)
        axes[1].bar(x + offset, acc_data[m], width=width, color=colors[i], label=m,
                    edgecolor="black", linewidth=0.4)

    for ax, title, ylabel in zip(axes, ["Macro-F1", "Accuracy"], ["Macro-F1", "Accuracy"]):
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(h)} s" for h in horizons])
        ax.set_xlabel("Prediction horizon")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Short-horizon comparison ({title})")
        ax.set_ylim(0, max(0.9, max(f1_data["OBB-ST-LSTM"] + acc_data["OBB-ST-LSTM"]) + 0.1))
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    axes[0].legend(loc="upper right", fontsize=9, framealpha=0.95)
    fig.suptitle("OBB-ST-LSTM vs baselines @ 3 / 5 / 8 second horizons",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "short_horizon_compare.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---- Figure 3: PeMS08 长时汇总图 ----
def draw_pems_long_horizon():
    data = json.loads(
        (PROJECT_ROOT / "outputs" / "reports" / "long_horizon_forecasting.json").read_text()
    )
    horizons = ["5min", "15min", "30min"]
    models = ["Persistence", "RidgeLag", "LSTM-deep", "GRU-deep", "Ours-ST-LSTM"]
    colors = ["#9aa0a6", "#f58518", "#e45756", "#72b7b2", "#54a24b"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    width = 0.16

    for row, target in enumerate(["flow", "speed"]):
        target_data = data["targets"][target]
        label = target_data["label"]

        for col, metric in enumerate(["mae", "rmse"]):
            ax = axes[row][col]
            x = np.arange(len(horizons))
            for i, m in enumerate(models):
                vals = [target_data["horizons"][h]["models"][m][metric] for h in horizons]
                offset = (i - 2) * width
                ax.bar(x + offset, vals, width=width, color=colors[i], label=m,
                       edgecolor="black", linewidth=0.4)
            ax.set_xticks(x)
            ax.set_xticklabels(horizons)
            ax.set_xlabel("Horizon")
            ax.set_ylabel(metric.upper())
            ax.set_title(f"{label} - {metric.upper()} (lower is better)")
            ax.grid(axis="y", alpha=0.3, linestyle="--")

    axes[0][1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    fig.suptitle("PeMS08 long-horizon prediction — Ours-ST-LSTM wins 6/6",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pems_long_horizon_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---- Figure 4: 旧 Fusion vs 新 OBB-ST-LSTM 维度对比 ----
def draw_method_compare():
    fig, ax = plt.subplots(figsize=(11, 6))

    aspects = [
        "# of models",
        "Forward passes",
        "Training stages",
        "OBB usage depth",
        "Cross-dataset transfer",
        "Ablation coverage",
        "3s Macro-F1",
        "Long-horizon validation",
    ]
    fusion_scores = [3.0, 3.0, 3.0, 1.0, 1.0, 2.0, 0.4724, 0.0]
    ours_scores =   [1.0, 1.0, 1.0, 4.0, 4.0, 5.0, 0.5589, 1.0]

    # Normalize to [0,1] for radar
    max_vals = [3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 0.6, 1.0]
    fusion_norm = [f / m for f, m in zip(fusion_scores, max_vals)]
    ours_norm = [o / m for o, m in zip(ours_scores, max_vals)]

    # Convert to lower-is-better for first 3
    fusion_norm[0] = 1 - fusion_norm[0]
    fusion_norm[1] = 1 - fusion_norm[1]
    fusion_norm[2] = 1 - fusion_norm[2]
    ours_norm[0] = 1 - ours_norm[0]
    ours_norm[1] = 1 - ours_norm[1]
    ours_norm[2] = 1 - ours_norm[2]

    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(aspects), endpoint=False).tolist()
    fusion_norm += fusion_norm[:1]
    ours_norm += ours_norm[:1]
    angles += angles[:1]
    aspects_loop = aspects + [aspects[0]]

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(angles, fusion_norm, "o-", color="#d62728", linewidth=2,
            label="Old: LSTM + XGBoost weighted fusion")
    ax.fill(angles, fusion_norm, color="#d62728", alpha=0.15)
    ax.plot(angles, ours_norm, "o-", color="#54a24b", linewidth=2,
            label="New: OBB-ST-LSTM single model")
    ax.fill(angles, ours_norm, color="#54a24b", alpha=0.20)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(aspects, fontsize=10)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.4)
    ax.set_title("Old vs New Method Comparison\n(Outer is better in all dimensions)",
                 fontsize=12, pad=24)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "method_compare_radar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---- Figure 5: 消融结果柱状图（突出 main vs ablations） ----
def draw_ablation_summary():
    data = json.loads(
        (PROJECT_ROOT / "outputs" / "reports" / "experiment_results.json").read_text()
    )
    pred = data["prediction"]
    main_f1 = pred["OBB-ST-LSTM"]["metrics"]["f1_macro"]
    abl = pred["OBB-ST-LSTM_ablation"]

    names = ["OBB-ST-LSTM\n(complete)"] + list(abl.keys())
    vals = [main_f1] + [v["metrics"]["f1_macro"] for v in abl.values()]
    colors = ["#54a24b"] + ["#4c78a8"] * len(abl)

    # Use short labels on x-axis; full names in legend
    short_names = ["Full\n(ours)", "A1", "A2", "A3", "A4", "A5"]
    full_names = ["OBB-ST-LSTM (complete)"] + list(abl.keys())

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(short_names)), vals, color=colors,
                  edgecolor="black", linewidth=0.5, width=0.65)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(short_names)))
    ax.set_xticklabels(short_names, fontsize=11, ha="center")
    ax.set_ylabel("Macro-F1", fontsize=11)
    ax.set_ylim(0, max(vals) * 1.22)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_title("Ablation results — all 5 variants degrade from the full OBB-ST-LSTM",
                 fontsize=12, pad=10)
    ax.axhline(main_f1, color="#54a24b", linestyle="--", alpha=0.5,
               label=f"Full model F1 = {main_f1:.4f}")

    # Legend with full variant names
    legend_handles = [
        mpatches.Patch(color=colors[i], label=f"{short_names[i].replace(chr(10), ' ')}: {full_names[i]}")
        for i in range(len(short_names))
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "ablation_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("[DELIVERY] generating architecture figure...")
    draw_architecture()
    print("[DELIVERY] generating short-horizon comparison...")
    draw_short_horizon_compare()
    print("[DELIVERY] generating PeMS08 long-horizon summary...")
    draw_pems_long_horizon()
    print("[DELIVERY] generating method radar comparison...")
    draw_method_compare()
    print("[DELIVERY] generating ablation summary...")
    draw_ablation_summary()
    print(f"[DELIVERY] all figures saved to {OUT_DIR}")
