"""Lightweight matplotlib plotting helpers for delivery2."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion(cm: list[list[int]], state_names: list[str], out_path: Path, title: str = "Confusion Matrix") -> None:
    cm_arr = np.asarray(cm, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks(range(len(state_names)))
    ax.set_yticks(range(len(state_names)))
    ax.set_xticklabels(state_names, rotation=30, ha="right")
    ax.set_yticklabels(state_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, int(cm_arr[i, j]), ha="center", va="center",
                    color="black" if cm_arr[i, j] < cm_arr.max() * 0.6 else "white", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_feature_importance(importance: list[dict], out_path: Path, top_k: int = 12) -> None:
    items = importance[:top_k]
    names = [it["feature"] for it in items][::-1]
    vals = [it["importance"] for it in items][::-1]
    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.35 * len(items))))
    ax.barh(range(len(names)), vals, color="#4c78a8", edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("XGBoost gain importance")
    ax.set_title(f"Top {top_k} feature importance")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_state_distribution(labels: np.ndarray, state_names: list[str], out_path: Path) -> None:
    counts = [int(np.sum(labels == c)) for c in range(len(state_names))]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(state_names, counts, color=["#54a24b", "#f1cb29", "#f58518", "#d62728"][:len(state_names)],
                  edgecolor="black", linewidth=0.4)
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(counts) * 0.015, str(v),
                ha="center", fontsize=10)
    ax.set_ylabel("Windows")
    ax.set_title("State label distribution")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
