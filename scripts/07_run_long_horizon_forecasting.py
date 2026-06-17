#!/usr/bin/env python
"""Long-horizon PeMS08 flow/speed regression + classifier-free state mapping.

The long-horizon experiment is a *regression* task (not 4-class classification):

  - targets : flow and speed
  - horizons: 5 / 15 / 30 min  (1 / 3 / 6 steps at 5-min interval)
  - metric  : MAE / RMSE
  - model   : GTSEP-DL = 1D spatial CNN + LSTM residual regressor with a
              persistence prior (src/ute_pipeline/models/gtsep_dl_pems.py)

Baselines: Persistence, RidgeLag, LSTM-deep, GRU-deep.

State-mapping supplement (§6): the regression output is mapped to four traffic
states *purely by fixed speed thresholds* -- there is NO trained classifier and
no post-processing. The predicted speed of each model is passed through the same
threshold function used on the ground-truth speed, so the discrete-state quality
is fully determined by the underlying speed-regression accuracy. This keeps the
supplement consistent with the paper: the model contains no classification head.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from sklearn.linear_model import Ridge

from ute_pipeline.models.gtsep_dl_pems import (
    GRURegressor,
    LSTMRegressor,
    GTSEPDLRegressorV2,
    fit_regressor,
    fit_regressor_es,
)

OURS_SEEDS = [0, 31, 73]  # 3-seed ensemble offsets for GTSEP-DL

DATASETS = {
    "PEMS08": {
        "url": "https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/master/data/PEMS08/PEMS08.npz",
        "interval_minutes": 5,
        "description": "PeMSD8 traffic flow/occupancy/speed data, 170 sensors, 5-minute interval.",
    },
}

MODEL_ORDER = ["Persistence", "RidgeLag", "LSTM-deep", "GRU-deep", "GTSEP-DL"]
TARGET_CHANNEL = {"flow": 0, "speed": 2}  # PeMS08 channels: 0=flow, 1=occupancy, 2=speed
HORIZON_STEPS = {"5min": 1, "15min": 3, "30min": 6}
SEQ_LEN = 12

# Fixed traffic-engineering speed thresholds (mph) -> 4 ordinal states.
# 0 = 畅通(free), 1 = 缓行(slow), 2 = 拥挤(congested), 3 = 堵塞(jam).
SPEED_STATE_THRESHOLDS = [60.0, 45.0, 30.0]


def load_pems08() -> np.ndarray:
    arr = np.load(PROJECT_ROOT / "data" / "long_horizon" / "PEMS08.npz")["data"]
    return arr.astype(np.float32)  # (T, N, C)


def speed_to_state(speed: np.ndarray) -> np.ndarray:
    """Map speed (mph) to 4 ordinal states via fixed thresholds (no classifier)."""
    state = np.zeros_like(speed, dtype=np.int64)
    state[speed < SPEED_STATE_THRESHOLDS[0]] = 1
    state[speed < SPEED_STATE_THRESHOLDS[1]] = 2
    state[speed < SPEED_STATE_THRESHOLDS[2]] = 3
    return state


def macro_f1_and_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> tuple[float, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    accuracy = float((y_true == y_pred).mean())
    f1s = []
    for c in range(n_classes):
        tp = float(((y_pred == c) & (y_true == c)).sum())
        fp = float(((y_pred == c) & (y_true != c)).sum())
        fn = float(((y_pred != c) & (y_true == c)).sum())
        if tp + fp + fn == 0:
            continue  # class absent in both -> excluded from macro mean
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return macro_f1, accuracy


def build_windows(values: np.ndarray, seq_len: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Return window end-indices t and their prediction target indices t+horizon.

    A window covers [t-seq_len+1, t]; the target is the value at t+horizon.
    """
    last_valid = values.shape[0] - horizon
    ends = np.arange(seq_len - 1, last_valid)
    return ends, ends + horizon


def mae_rmse(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    err = pred - true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }


def run_target_horizon(
    data: np.ndarray,
    target: str,
    horizon: int,
    train_end: int,
    test_start: int,
    device: str,
    seed: int,
) -> dict[str, dict[str, float]]:
    T, N, C = data.shape
    tgt_c = TARGET_CHANNEL[target]

    # Train-only normalisation stats per channel.
    train_slab = data[:train_end]
    ch_mean = train_slab.reshape(-1, C).mean(axis=0)
    ch_std = train_slab.reshape(-1, C).std(axis=0) + 1e-6
    norm = (data - ch_mean) / ch_std  # (T, N, C)

    # Per-step disturbance descriptor: cross-sensor mean absolute change of
    # occupancy (channel 1) in normalised space -> a scalar local-volatility
    # signal that drives the disturbance gate. dist_time[0] = 0.
    occ = norm[:, :, 1]  # (T, N)
    dist_time = np.zeros(T, dtype=np.float32)
    dist_time[1:] = np.mean(np.abs(occ[1:] - occ[:-1]), axis=1)

    ends, tgt_idx = build_windows(data[:, 0, 0], SEQ_LEN, horizon)
    train_mask = ends < train_end
    val_mask = (ends >= train_end) & (ends < test_start)
    test_mask = ends >= test_start
    ends_tr, tgt_tr = ends[train_mask], tgt_idx[train_mask]
    ends_va, tgt_va = ends[val_mask], tgt_idx[val_mask]
    ends_te, tgt_te = ends[test_mask], tgt_idx[test_mask]

    # Ground-truth future target in original units.
    true_te = data[tgt_te][:, :, tgt_c]  # (Bte, N)
    last_obs_te = data[ends_te][:, :, tgt_c]  # persistence prediction (original units)
    tgt_mean, tgt_std = ch_mean[tgt_c], ch_std[tgt_c]

    def make_seq(ends_arr: np.ndarray) -> np.ndarray:
        # (B, T, N, C) normalised
        idx = ends_arr[:, None] + np.arange(-SEQ_LEN + 1, 1)[None, :]
        return norm[idx]  # (B, seq, N, C)

    def make_dist(ends_arr: np.ndarray) -> np.ndarray:
        idx = ends_arr[:, None] + np.arange(-SEQ_LEN + 1, 1)[None, :]
        return dist_time[idx][:, :, None]  # (B, seq, 1)

    seq_tr = make_seq(ends_tr)
    seq_va = make_seq(ends_va)
    seq_te = make_seq(ends_te)
    y_tr = norm[tgt_tr][:, :, tgt_c]  # (Btr, N) normalised target
    y_va = norm[tgt_va][:, :, tgt_c]

    results: dict[str, dict[str, float]] = {}

    # --- Persistence ---
    results["Persistence"] = mae_rmse(last_obs_te, true_te)

    # --- RidgeLag (own-channel lag regression, weights shared across sensors) ---
    # seq_*[:,:,:,tgt_c] is (B, seq, N); move the time axis last so each row is
    # one sensor's seq_len lags aligned with its label (B-major, N-minor order).
    lag_tr = seq_tr[:, :, :, tgt_c].transpose(0, 2, 1).reshape(-1, SEQ_LEN)  # (Btr*N, seq)
    lab_tr = y_tr.reshape(-1)
    ridge = Ridge(alpha=10.0)
    ridge.fit(lag_tr, lab_tr)
    lag_te = seq_te[:, :, :, tgt_c].transpose(0, 2, 1).reshape(-1, SEQ_LEN)
    pred_ridge = ridge.predict(lag_te).reshape(true_te.shape) * tgt_std + tgt_mean
    results["RidgeLag"] = mae_rmse(pred_ridge, true_te)

    # --- LSTM-deep / GRU-deep: flatten all channels per timestep ---
    flat_tr = torch.tensor(seq_tr.reshape(seq_tr.shape[0], SEQ_LEN, N * C), dtype=torch.float32)
    flat_te = torch.tensor(seq_te.reshape(seq_te.shape[0], SEQ_LEN, N * C), dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)

    lstm = LSTMRegressor(input_dim=N * C, sensors=N, hidden_size=64)
    pred_lstm = fit_regressor(lstm, flat_tr, y_tr_t, flat_te, seed, epochs=40, batch_size=64, device=device)
    results["LSTM-deep"] = mae_rmse(pred_lstm * tgt_std + tgt_mean, true_te)

    gru = GRURegressor(input_dim=N * C, sensors=N, hidden_size=64)
    pred_gru = fit_regressor(gru, flat_tr, y_tr_t, flat_te, seed, epochs=40, batch_size=64, device=device)
    results["GRU-deep"] = mae_rmse(pred_gru * tgt_std + tgt_mean, true_te)

    # --- GTSEP-DL (V2): AR trend prior + gated nonlinear residual +
    # disturbance-gated LSTM, trained with L1 loss, validation early stopping
    # and a 3-seed ensemble. Input (B, T, C, 1, N) with the target channel
    # placed first so the AR prior reads the target history. ---
    ch_order = [tgt_c] + [c for c in range(C) if c != tgt_c]

    def to_ours(seq: np.ndarray) -> torch.Tensor:
        return torch.tensor(
            seq[:, :, :, ch_order].transpose(0, 1, 3, 2)[:, :, :, None, :], dtype=torch.float32
        )  # (B, T, C, 1, N)

    ours_tr, ours_va, ours_te = to_ours(seq_tr), to_ours(seq_va), to_ours(seq_te)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)
    d_tr = torch.tensor(make_dist(ends_tr), dtype=torch.float32)
    d_va = torch.tensor(make_dist(ends_va), dtype=torch.float32)
    d_te = torch.tensor(make_dist(ends_te), dtype=torch.float32)

    ours_probs = []
    for off in OURS_SEEDS:
        model = GTSEPDLRegressorV2(
            in_channels=C, sensors=N, seq_len=SEQ_LEN, hidden_size=64, conv_channels=(8, 8)
        )
        pred = fit_regressor_es(
            model, ours_tr, y_tr_t, ours_va, y_va_t, ours_te, seed + off,
            max_epochs=80, patience=10, batch_size=64, device=device,
            train_disturbance=d_tr, val_disturbance=d_va, test_disturbance=d_te,
        )
        ours_probs.append(pred)
    pred_ours = np.mean(np.stack(ours_probs, axis=0), axis=0)
    pred_ours_orig = pred_ours * tgt_std + tgt_mean
    results["GTSEP-DL"] = mae_rmse(pred_ours_orig, true_te)

    # Stash speed predictions (original units) for the state-mapping supplement.
    if target == "speed":
        results["_speed_pred"] = {
            "true": true_te,
            "Persistence": last_obs_te,
            "GTSEP-DL": pred_ours_orig,
        }
    return results


def build_report(dataset: str, device: str, seed: int) -> dict:
    data = load_pems08()
    T = data.shape[0]
    train_end = int(round(T * 0.6))
    test_start = int(round(T * 0.8))

    results: dict[str, dict[str, object]] = {}
    speed_preds: dict[str, dict[str, np.ndarray]] = {}
    for target in ("flow", "speed"):
        target_block = {}
        for horizon_name, horizon in HORIZON_STEPS.items():
            raw = run_target_horizon(data, target, horizon, train_end, test_start, device, seed)
            if "_speed_pred" in raw:
                speed_preds[horizon_name] = raw.pop("_speed_pred")
            model_block = {name: {**vals, "mape": None} for name, vals in raw.items()}
            best_mae = min(model_block.items(), key=lambda kv: kv[1]["mae"])[0]
            best_rmse = min(model_block.items(), key=lambda kv: kv[1]["rmse"])[0]
            target_block[horizon_name] = {
                "models": {k: model_block[k] for k in MODEL_ORDER},
                "best_model_by_mae": best_mae,
                "best_model_by_rmse": best_rmse,
                "ours_leads_mae": best_mae == "GTSEP-DL",
                "ours_leads_rmse": best_rmse == "GTSEP-DL",
            }
            print(f"[LONG] {target} {horizon_name}: " + ", ".join(
                f"{k}={model_block[k]['mae']:.3f}" for k in MODEL_ORDER))
        results[target] = target_block

    # --- Classifier-free speed -> state mapping supplement ---
    supplement = {"thresholds_mph": SPEED_STATE_THRESHOLDS, "n_states": 4, "horizons": {}}
    for horizon_name, preds in speed_preds.items():
        true_state = speed_to_state(preds["true"])
        block = {}
        for model in ("Persistence", "GTSEP-DL"):
            pred_state = speed_to_state(preds[model])
            mf1, acc = macro_f1_and_accuracy(true_state, pred_state)
            block[model] = {"macro_f1": round(mf1, 4), "accuracy": round(acc, 4)}
        best = max(block.items(), key=lambda kv: kv[1]["macro_f1"])[0]
        block["best_model_by_macro_f1"] = best
        supplement["horizons"][horizon_name] = block
        print(f"[STATE] {horizon_name}: Persistence mF1={block['Persistence']['macro_f1']:.4f} "
              f"Ours mF1={block['GTSEP-DL']['macro_f1']:.4f} -> best={best}")

    ours_mae = sum(results[t][h]["ours_leads_mae"] for t in results for h in results[t])
    ours_rmse = sum(results[t][h]["ours_leads_rmse"] for t in results for h in results[t])

    return {
        "dataset": dataset,
        "source": DATASETS[dataset]["url"],
        "description": DATASETS[dataset]["description"],
        "interval_minutes": DATASETS[dataset]["interval_minutes"],
        "shape": {"time_steps": int(data.shape[0]), "sensors": int(data.shape[1]), "channels": int(data.shape[2])},
        "split": {
            "train_ratio": 0.6, "validation_ratio": 0.2, "test_ratio": 0.2,
            "train_end_step": train_end, "test_start_step": test_start,
        },
        "task": "long-horizon traffic flow/speed regression",
        "targets": ["flow", "speed"],
        "horizons_minutes": [5, 15, 30],
        "seq_len": SEQ_LEN,
        "models": MODEL_ORDER,
        "ours_model": "1D spatial CNN + LSTM residual regressor with persistence prior",
        "metrics": ["MAE", "RMSE"],
        "seed": seed,
        "results": results,
        "state_mapping_supplement": supplement,
        "summary": {
            "ours_leads_mae_count": int(ours_mae),
            "ours_leads_rmse_count": int(ours_rmse),
            "completed_settings": 6,
            "note": "GTSEP-DL MAE/RMSE on flow/speed x 5/15/30min; state mapping is classifier-free.",
        },
    }


def plot_results(report: dict, out_path: Path) -> None:
    horizons = ["5min", "15min", "30min"]
    colors = {
        "Persistence": "#8a8f98", "RidgeLag": "#4c78a8", "LSTM-deep": "#e45756",
        "GRU-deep": "#72b7b2", "GTSEP-DL": "#54a24b",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    panels = [
        ("flow", "mae", axes[0][0], "Flow MAE"), ("flow", "rmse", axes[0][1], "Flow RMSE"),
        ("speed", "mae", axes[1][0], "Speed MAE"), ("speed", "rmse", axes[1][1], "Speed RMSE"),
    ]
    x = np.arange(len(horizons))
    width = 0.15
    for target, metric, ax, title in panels:
        for i, model in enumerate(MODEL_ORDER):
            values = [report["results"][target][h]["models"][model][metric] for h in horizons]
            offset = (i - (len(MODEL_ORDER) - 1) / 2) * width
            ax.bar(x + offset, values, width=width, label=model, color=colors[model])
        ax.set_xticks(x)
        ax.set_xticklabels(["5 min", "15 min", "30 min"])
        ax.set_title(title)
        ax.set_xlabel("Prediction horizon")
        ax.set_ylabel(metric.upper())
        ax.grid(axis="y", alpha=0.25)
    axes[0][1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("PeMS08 long-horizon regression (real run)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PeMS08 long-horizon regression + state mapping.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="PEMS08")
    parser.add_argument("--auto-download", action="store_true", help="Accepted for compatibility.")
    parser.add_argument("--device", default=None, help="cuda / cuda:0 / cpu (default: auto)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    report = build_report(args.dataset, device, args.seed)
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
