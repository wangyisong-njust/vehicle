#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "PEMS08": {
        "url": "https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/master/data/PEMS08/PEMS08.npz",
        "interval_minutes": 5,
        "description": "PeMSD8 traffic flow data, 170 sensors, 5-minute interval.",
    },
    "PEMS04": {
        "url": "https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/master/data/PEMS04/PEMS04.npz",
        "interval_minutes": 5,
        "description": "PeMSD4 traffic flow data, 307 sensors, 5-minute interval.",
    },
}


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


def make_supervised(data: np.ndarray, horizon_steps: int, lags: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_lag = max(lags)
    positions = np.arange(max_lag, data.shape[0] - horizon_steps, dtype=np.int64)
    x = np.concatenate([data[positions - lag] for lag in lags], axis=1)
    y = data[positions + horizon_steps]
    return positions, x, y


def masked_mape(true: np.ndarray, pred: np.ndarray, threshold: float = 10.0) -> float:
    mask = np.abs(true) > threshold
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100.0)


def metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    diff = pred - true
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(math.sqrt(np.mean(diff * diff))),
        "mape": masked_mape(true, pred),
    }


def historical_average(data: np.ndarray, train_end: int, target_positions: np.ndarray, slots_per_day: int) -> np.ndarray:
    slot_sum: dict[int, np.ndarray] = {}
    slot_count: dict[int, int] = {}
    for t in range(train_end):
        slot = t % slots_per_day
        if slot not in slot_sum:
            slot_sum[slot] = np.zeros(data.shape[1], dtype=np.float64)
            slot_count[slot] = 0
        slot_sum[slot] += data[t]
        slot_count[slot] += 1
    global_mean = data[:train_end].mean(axis=0)
    preds = []
    for t in target_positions:
        slot = int(t % slots_per_day)
        if slot in slot_sum and slot_count[slot] > 0:
            preds.append(slot_sum[slot] / slot_count[slot])
        else:
            preds.append(global_mean)
    return np.asarray(preds, dtype=np.float32)


def run_dataset(dataset: str, auto_download: bool) -> dict[str, object]:
    meta = DATASETS[dataset]
    interval_minutes = int(meta["interval_minutes"])
    data_path = download_if_needed(dataset, PROJECT_ROOT / "data" / "long_horizon", auto_download)
    raw = np.load(data_path)["data"].astype(np.float32)
    flow = raw[:, :, 0]
    total_steps, sensors = flow.shape
    train_end = int(total_steps * 0.6)
    test_start = int(total_steps * 0.8)
    slots_per_day = int(round(24 * 60 / interval_minutes))
    lags = [1, 2, 3, 6, 12]

    results: dict[str, object] = {
        "dataset": dataset,
        "source": meta["url"],
        "description": meta["description"],
        "interval_minutes": interval_minutes,
        "shape": {
            "time_steps": int(total_steps),
            "sensors": int(sensors),
            "channels": int(raw.shape[2]) if raw.ndim == 3 else 1,
        },
        "split": {
            "train_ratio": 0.6,
            "validation_ratio": 0.2,
            "test_ratio": 0.2,
            "train_end_step": train_end,
            "test_start_step": test_start,
        },
        "target": "traffic_flow_channel_0",
        "lags_steps": lags,
        "horizons": {},
    }

    for horizon_minutes in [15, 30, 60]:
        horizon_steps = max(1, int(round(horizon_minutes / interval_minutes)))
        positions, x, y = make_supervised(flow, horizon_steps, lags)
        train_mask = positions < train_end
        test_mask = positions >= test_start
        x_train, y_train = x[train_mask], y[train_mask]
        x_test, y_test = x[test_mask], y[test_mask]
        target_positions = positions[test_mask] + horizon_steps

        persistence_pred = flow[positions[test_mask]]
        ha_pred = historical_average(flow, train_end, target_positions, slots_per_day)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        ridge = Ridge(alpha=10.0)
        ridge.fit(x_train_scaled, y_train)
        ridge_pred = ridge.predict(x_test_scaled)

        results["horizons"][f"{horizon_minutes}min"] = {
            "horizon_minutes": horizon_minutes,
            "horizon_steps": int(horizon_steps),
            "train_samples": int(x_train.shape[0]),
            "test_samples": int(x_test.shape[0]),
            "models": {
                "Persistence": metrics(y_test, persistence_pred),
                "HistoricalAverage": metrics(y_test, ha_pred),
                "RidgeLag": metrics(y_test, ridge_pred),
            },
        }
    return results


def plot_results(report: dict[str, object], out_path: Path) -> None:
    horizons = list(report["horizons"].keys())
    models = ["Persistence", "HistoricalAverage", "RidgeLag"]
    x = np.arange(len(horizons))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["#9aa0a6", "#4c78a8", "#f58518"]
    for i, model in enumerate(models):
        mae = [report["horizons"][h]["models"][model]["mae"] for h in horizons]
        rmse = [report["horizons"][h]["models"][model]["rmse"] for h in horizons]
        axes[0].bar(x + (i - 1) * width, mae, width=width, label=model, color=colors[i])
        axes[1].bar(x + (i - 1) * width, rmse, width=width, label=model, color=colors[i])
    for ax, ylabel in [(axes[0], "MAE"), (axes[1], "RMSE")]:
        ax.set_xticks(x)
        ax.set_xticklabels(horizons)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Prediction horizon")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title(f"{report['dataset']} long-horizon flow forecasting")
    axes[1].set_title("Lower is better")
    axes[1].legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an optional PeMS long-horizon traffic flow forecasting benchmark.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="PEMS08")
    parser.add_argument("--auto-download", action="store_true", help="Download the public npz dataset if missing.")
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
