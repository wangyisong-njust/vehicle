#!/usr/bin/env python
"""Long-horizon traffic STATE forecasting on PeMS08 (4-class classification).

Speed thresholds aligned with UTE short-term experiment (converted to mph,
the native unit of PeMS08 California freeway data):

  State 0 (畅通 / Free-flow):  speed ≥ 62.14 mph  (≥ 100 km/h,  LOS A/B)
  State 1 (缓行 / Synchronized): 49.71 ≤ speed < 62.14 mph (80–100 km/h, LOS C/D)
  State 2 (拥挤 / Congested):  37.28 ≤ speed < 49.71 mph (60–80 km/h,  LOS E)
  State 3 (堵塞 / Jam):        speed < 37.28 mph           (< 60 km/h,   LOS F)

Metrics: Macro-F1 and Accuracy — identical to the UTE short-term experiment.
Models: Persistence, HistMode, LSTM, GRU, Ours-ST-LSTM (3-seed ensemble).
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ute_pipeline.models.obb_st_lstm import FocalLoss  # noqa: E402
from ute_pipeline.models.st_lstm_pems import (  # noqa: E402
    GRUClassifier,
    LSTMClassifier,
    STLSTMClassifier,
    fit_classifier,
)

DATASETS = {
    "PEMS08": {
        "url": "https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/master/data/PEMS08/PEMS08.npz",
        "interval_minutes": 5,
        "description": "PeMSD8 traffic flow/occupancy/speed data, 170 sensors, 5-minute interval.",
    },
}

SPEED_CHANNEL = 2  # channel index: 0=flow, 1=occupancy, 2=speed
N_CLASSES = 4

# Thresholds in mph (converted from [100, 80, 60] km/h)
_THR = [62.14, 49.71, 37.28]

STATE_NAMES = ["畅通(S0)", "缓行(S1)", "拥挤(S2)", "堵塞(S3)"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def download_if_needed(dataset: str, data_dir: Path, auto_download: bool) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / f"{dataset}.npz"
    if out_path.exists():
        return out_path
    if not auto_download:
        raise FileNotFoundError(
            f"{out_path} not found. Rerun with --auto-download or place the npz file there manually."
        )
    url = DATASETS[dataset]["url"]
    print(f"[LONG] downloading {dataset} from {url}")
    urllib.request.urlretrieve(url, out_path)
    return out_path


def speed_to_state(speed: np.ndarray) -> np.ndarray:
    """Map speed (mph) → 4-class state label (0=畅通 … 3=堵塞)."""
    out = np.full(speed.shape, 3, dtype=np.int64)
    out[speed >= _THR[2]] = 2
    out[speed >= _THR[1]] = 1
    out[speed >= _THR[0]] = 0
    return out


def cls_metrics(true: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    flat_true = true.ravel()
    flat_pred = pred.ravel()
    if mask is not None:
        m = mask.ravel().astype(bool)
        flat_true = flat_true[m]
        flat_pred = flat_pred[m]
    if flat_true.size == 0:
        return {"macro_f1": 0.0, "accuracy": 0.0, "n_samples": 0}
    return {
        "macro_f1": float(f1_score(flat_true, flat_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(flat_true, flat_pred)),
        "n_samples": int(flat_true.size),
    }


def _modal_state_per_sensor(arr: np.ndarray) -> np.ndarray:
    """arr: (T, N) int array → (N,) modal state per sensor (vectorised over N)."""
    counts = np.stack([np.sum(arr == s, axis=0) for s in range(N_CLASSES)], axis=0)  # (4, N)
    return counts.argmax(axis=0).astype(np.int64)  # (N,)


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------

def persistence_baseline(state_data: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Predict the current observed state as the future state."""
    return state_data[positions]  # (B, N)


def historical_mode_baseline(
    state_data: np.ndarray,
    train_end: int,
    target_positions: np.ndarray,
    slots_per_day: int,
) -> np.ndarray:
    """For each target time slot, predict the modal state seen during training."""
    train_states = state_data[:train_end]
    slot_indices = np.arange(train_end) % slots_per_day
    slot_modes: dict[int, np.ndarray] = {}
    for slot in range(slots_per_day):
        mask = slot_indices == slot
        if mask.any():
            slot_modes[slot] = _modal_state_per_sensor(train_states[mask])
    global_mode = _modal_state_per_sensor(train_states)
    preds = np.stack(
        [slot_modes.get(int(t % slots_per_day), global_mode) for t in target_positions],
        axis=0,
    )  # (B, N)
    return preds


# ---------------------------------------------------------------------------
# dataset builder
# ---------------------------------------------------------------------------

def _build_seq_dataset(
    speed: np.ndarray,
    states: np.ndarray,
    horizon_steps: int,
    seq_len: int,
    train_end: int,
    test_start: int,
) -> dict:
    """Slide a seq_len window over (T, N) speed; target is state label at t+horizon.

    Returns dict with:
      x  (samples, seq_len, N)  float32 speed sequences (raw, normalise outside)
      y  (samples, N)           int64 state labels
    """
    n_t = speed.shape[0]
    end_positions = np.arange(seq_len - 1, n_t - horizon_steps, dtype=np.int64)
    xs = np.stack([speed[e - seq_len + 1 : e + 1] for e in end_positions], axis=0)
    ys = states[end_positions + horizon_steps]  # (B, N) int64
    train_mask = end_positions < train_end
    val_mask = (end_positions >= train_end) & (end_positions < test_start)
    test_mask = end_positions >= test_start
    return {
        "x": xs.astype(np.float32),
        "y": ys,
        "end_positions": end_positions,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "n_sensors": int(speed.shape[1]),
    }


# ---------------------------------------------------------------------------
# deep model training helpers
# ---------------------------------------------------------------------------

def _train_classifier_seeds(
    model_factory,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    seeds: list[int],
    epochs: int,
    batch_size: int,
    loss_fn=None,
) -> np.ndarray:
    """Average softmax probabilities over multiple seeds; return argmax labels."""
    probs_list = []
    for s in seeds:
        torch.manual_seed(s)
        model = model_factory()
        probs = fit_classifier(
            model, train_x, train_y, test_x, s,
            epochs=epochs, batch_size=batch_size, loss_fn=loss_fn,
        )
        probs_list.append(probs)
    avg_probs = np.mean(np.stack(probs_list, axis=0), axis=0)  # (B, N, 4)
    return avg_probs.argmax(axis=-1).astype(np.int64)  # (B, N)


def _make_focal_loss(train_labels: np.ndarray) -> FocalLoss:
    """Build FocalLoss(gamma=2.0) with mildly imbalance-aware class weights.

    PeMS08 has extreme class imbalance (>75% free-flow). Pure inverse-frequency
    weights over-correct and collapse majority-class accuracy; we use sqrt-scaled
    weights to keep the dynamic range bounded while still up-weighting minorities.
    """
    counts = np.bincount(train_labels.ravel(), minlength=N_CLASSES).astype(np.float64)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    weights = weights / max(1e-6, float(weights.mean()))
    return FocalLoss(gamma=2.0, weight=torch.tensor(weights, dtype=torch.float32))


# ---------------------------------------------------------------------------
# core experiment
# ---------------------------------------------------------------------------

def run_state_forecasting(
    speed_data: np.ndarray,
    interval_minutes: int,
    train_end: int,
    test_start: int,
    slots_per_day: int,
) -> dict:
    """Run 4-class state classification at 5 / 15 / 30 min prediction horizons."""
    state_data = speed_to_state(speed_data)  # (T, N)
    n_sensors = int(speed_data.shape[1])

    train_flat = state_data[:train_end].ravel()
    train_class_dist = {STATE_NAMES[i]: int(np.sum(train_flat == i)) for i in range(N_CLASSES)}

    seq_len = 12
    seeds = [7, 42, 73, 115, 211, 337, 503]
    epochs = 50
    batch_size = 128

    horizons_out: dict = {}

    for horizon_minutes in [5, 15, 30]:
        horizon_steps = max(1, int(round(horizon_minutes / interval_minutes)))
        print(f"[LONG] horizon={horizon_minutes}min ({horizon_steps} steps)")

        ds = _build_seq_dataset(speed_data, state_data, horizon_steps, seq_len, train_end, test_start)
        train_idx = np.where(ds["train_mask"])[0]
        test_idx = np.where(ds["test_mask"])[0]

        # Normalise speed inputs using training statistics
        x_all = ds["x"]
        train_mean = float(x_all[train_idx].mean())
        train_std = max(float(x_all[train_idx].std()), 1e-6)
        x_norm = (x_all - train_mean) / train_std

        y_test = ds["y"][test_idx]  # (B_test, N) ground-truth states
        end_pos_test = ds["end_positions"][test_idx]
        target_pos_test = end_pos_test + horizon_steps

        # ---- Baseline: Persistence ----
        pers_pred = persistence_baseline(state_data, end_pos_test)

        # ---- Baseline: Historical mode ----
        hist_pred = historical_mode_baseline(state_data, train_end, target_pos_test, slots_per_day)

        # ---- Deep models ----
        train_x_t = torch.tensor(x_norm[train_idx], dtype=torch.float32)   # (B, T, N)
        train_y_t = torch.tensor(ds["y"][train_idx], dtype=torch.long)      # (B, N)
        test_x_t = torch.tensor(x_norm[test_idx], dtype=torch.float32)

        # ST-LSTM expects (B, T, C=1, 1, N)
        train_x_st = train_x_t.unsqueeze(2).unsqueeze(3)
        test_x_st = test_x_t.unsqueeze(2).unsqueeze(3)

        # FocalLoss with class weights derived from this horizon's training labels
        focal_loss = _make_focal_loss(ds["y"][train_idx])

        lstm_pred = _train_classifier_seeds(
            lambda: LSTMClassifier(input_dim=n_sensors, sensors=n_sensors, hidden_size=64),
            train_x_t, train_y_t, test_x_t, seeds, epochs, batch_size,
            loss_fn=focal_loss,
        )
        gru_pred = _train_classifier_seeds(
            lambda: GRUClassifier(input_dim=n_sensors, sensors=n_sensors, hidden_size=64),
            train_x_t, train_y_t, test_x_t, [s + 17 for s in seeds], epochs, batch_size,
            loss_fn=focal_loss,
        )
        st_pred = _train_classifier_seeds(
            lambda: STLSTMClassifier(
                in_channels=1, sensors=n_sensors, hidden_size=64,
                conv_channels=(16, 32), bidirectional=True,
            ),
            train_x_st, train_y_t, test_x_st, seeds, epochs, batch_size,
            loss_fn=focal_loss,
        )

        # Transition mask: where the future state actually differs from current
        current_state_test = state_data[end_pos_test]  # (B, N)
        trans_mask = (y_test != current_state_test)
        n_trans = int(trans_mask.sum())
        n_total = int(trans_mask.size)

        all_preds = {
            "Persistence": pers_pred,
            "HistMode": hist_pred,
            "LSTM": lstm_pred,
            "GRU": gru_pred,
            "Ours-ST-LSTM": st_pred,
        }
        models_block: dict = {}
        for name, pred in all_preds.items():
            models_block[name] = {
                "all": cls_metrics(y_test, pred),
                "transitions": cls_metrics(y_test, pred, mask=trans_mask),
            }

        best_all = max(models_block.items(), key=lambda kv: kv[1]["all"]["macro_f1"])
        best_trans = max(models_block.items(), key=lambda kv: kv[1]["transitions"]["macro_f1"])

        horizons_out[f"{horizon_minutes}min"] = {
            "horizon_minutes": horizon_minutes,
            "horizon_steps": int(horizon_steps),
            "effective_minutes": int(horizon_steps * interval_minutes),
            "train_samples": int(train_idx.size),
            "test_samples": int(test_idx.size),
            "transition_samples": n_trans,
            "transition_ratio": float(n_trans / max(n_total, 1)),
            "seq_len": int(seq_len),
            "epochs": int(epochs),
            "seeds": seeds,
            "loss": "FocalLoss(gamma=2.0, class_weighted)",
            "models": models_block,
            "best_model_all_by_macro_f1": best_all[0],
            "best_model_transitions_by_macro_f1": best_trans[0],
            "ours_leads_all": bool(best_all[0] == "Ours-ST-LSTM"),
            "ours_leads_transitions": bool(best_trans[0] == "Ours-ST-LSTM"),
        }

    return {
        "task": "4-class traffic state classification",
        "state_definitions": {
            STATE_NAMES[0]: "speed >= 100 km/h (>= 62.14 mph)",
            STATE_NAMES[1]: "80 <= speed < 100 km/h (49.71–62.14 mph)",
            STATE_NAMES[2]: "60 <= speed < 80 km/h (37.28–49.71 mph)",
            STATE_NAMES[3]: "speed < 60 km/h (< 37.28 mph)",
        },
        "train_class_distribution": train_class_dist,
        "horizons": horizons_out,
    }


# ---------------------------------------------------------------------------
# dataset runner
# ---------------------------------------------------------------------------

def run_dataset(dataset: str, auto_download: bool) -> dict:
    meta = DATASETS[dataset]
    interval_minutes = int(meta["interval_minutes"])
    data_path = download_if_needed(dataset, PROJECT_ROOT / "data" / "long_horizon", auto_download)
    raw = np.load(data_path)["data"].astype(np.float32)
    total_steps = int(raw.shape[0])
    sensors = int(raw.shape[1])
    train_end = int(total_steps * 0.6)
    test_start = int(total_steps * 0.8)
    slots_per_day = int(round(24 * 60 / interval_minutes))

    speed_data = raw[:, :, SPEED_CHANNEL]  # (T, N)
    state_results = run_state_forecasting(speed_data, interval_minutes, train_end, test_start, slots_per_day)

    return {
        "dataset": dataset,
        "source": meta["url"],
        "description": meta["description"],
        "interval_minutes": interval_minutes,
        "shape": {
            "time_steps": total_steps,
            "sensors": sensors,
            "channels": int(raw.shape[2]),
        },
        "split": {
            "train_ratio": 0.6,
            "validation_ratio": 0.2,
            "test_ratio": 0.2,
            "train_end_step": train_end,
            "test_start_step": test_start,
        },
        **state_results,
    }


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def plot_results(report: dict, out_path: Path) -> None:
    horizons = list(report["horizons"].keys())
    model_order = ["Persistence", "HistMode", "LSTM", "GRU", "Ours-ST-LSTM"]
    colors = {
        "Persistence": "#9aa0a6",
        "HistMode": "#4c78a8",
        "LSTM": "#e45756",
        "GRU": "#72b7b2",
        "Ours-ST-LSTM": "#54a24b",
    }
    models = [m for m in model_order if m in report["horizons"][horizons[0]]["models"]]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    x = np.arange(len(horizons))
    width = min(0.16, 0.78 / max(len(models), 1))

    for i, model in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * width
        f1_all = [report["horizons"][h]["models"][model]["all"]["macro_f1"] for h in horizons]
        acc_all = [report["horizons"][h]["models"][model]["all"]["accuracy"] for h in horizons]
        f1_tr = [report["horizons"][h]["models"][model]["transitions"]["macro_f1"] for h in horizons]
        acc_tr = [report["horizons"][h]["models"][model]["transitions"]["accuracy"] for h in horizons]
        axes[0][0].bar(x + offset, f1_all, width=width, label=model, color=colors.get(model))
        axes[0][1].bar(x + offset, acc_all, width=width, label=model, color=colors.get(model))
        axes[1][0].bar(x + offset, f1_tr, width=width, label=model, color=colors.get(model))
        axes[1][1].bar(x + offset, acc_tr, width=width, label=model, color=colors.get(model))

    panel_meta = [
        (axes[0][0], "Macro-F1", f"{report['dataset']} all samples — Macro-F1 (higher is better)"),
        (axes[0][1], "Accuracy", f"{report['dataset']} all samples — Accuracy (higher is better)"),
        (axes[1][0], "Macro-F1", f"{report['dataset']} state-transition samples — Macro-F1 (higher is better)"),
        (axes[1][1], "Accuracy", f"{report['dataset']} state-transition samples — Accuracy (higher is better)"),
    ]
    for ax, ylabel, title in panel_meta:
        ax.set_xticks(x)
        ax.set_xticklabels(horizons)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Prediction horizon")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        ax.set_title(title)

    axes[0][1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Long-horizon 4-class traffic state forecasting on PeMS."
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="PEMS08")
    parser.add_argument("--auto-download", action="store_true", help="Download npz if missing.")
    args = parser.parse_args()

    report = run_dataset(args.dataset, args.auto_download)
    report_dir = PROJECT_ROOT / "outputs" / "reports"
    fig_dir = PROJECT_ROOT / "outputs" / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_json = report_dir / "long_horizon_forecasting.json"
    out_png = fig_dir / "long_horizon_forecasting.png"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_results(report, out_png)
    print(f"[LONG] wrote {out_json}")
    print(f"[LONG] wrote {out_png}")


if __name__ == "__main__":
    main()
