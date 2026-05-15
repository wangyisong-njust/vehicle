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
        "description": "PeMSD8 traffic flow/occupancy/speed data, 170 sensors, 5-minute interval.",
    },
    "PEMS04": {
        "url": "https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/master/data/PEMS04/PEMS04.npz",
        "interval_minutes": 5,
        "description": "PeMSD4 traffic flow/occupancy/speed data, 307 sensors, 5-minute interval.",
    },
}

TARGETS = {
    "flow": {
        "channel": 0,
        "label": "Traffic flow",
        "mape_threshold": 10.0,
        "unit": "veh/5min",
    },
    "speed": {
        "channel": 2,
        "label": "Traffic speed",
        "mape_threshold": 1.0,
        "unit": "mph",
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
    max_lag = max(max(lags), horizon_steps)
    positions = np.arange(max_lag, data.shape[0] - horizon_steps, dtype=np.int64)
    x = np.concatenate([data[positions - lag] for lag in lags], axis=1)
    y = data[positions + horizon_steps]
    return positions, x, y


def masked_mape(true: np.ndarray, pred: np.ndarray, threshold: float = 10.0) -> float:
    mask = np.abs(true) > threshold
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100.0)


def metrics(true: np.ndarray, pred: np.ndarray, mape_threshold: float = 10.0) -> dict[str, float]:
    diff = pred - true
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(math.sqrt(np.mean(diff * diff))),
        "mape": masked_mape(true, pred, threshold=mape_threshold),
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


def seasonal_persistence(data: np.ndarray, fallback_end: int, target_positions: np.ndarray, slots_per_day: int) -> np.ndarray:
    fallback = data[:fallback_end].mean(axis=0)
    preds = []
    for t in target_positions:
        source_t = int(t) - slots_per_day
        if source_t >= 0:
            preds.append(data[source_t])
        else:
            preds.append(fallback)
    return np.asarray(preds, dtype=np.float32)


def simplex_partitions(n_parts: int, total_units: int, prefix: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    if n_parts == 1:
        return [prefix + (total_units,)]
    out: list[tuple[int, ...]] = []
    for value in range(total_units + 1):
        out.extend(simplex_partitions(n_parts - 1, total_units - value, prefix + (value,)))
    return out


def optimize_fusion_weights(true: np.ndarray, preds: dict[str, np.ndarray]) -> dict[str, float]:
    names = list(preds)
    best_score = float("inf")
    best_weights = {name: 1.0 / len(names) for name in names}
    # A 0.05 simplex grid keeps the fusion validation-only and deterministic.
    for partition in simplex_partitions(len(names), total_units=20):
        weights = np.asarray(partition, dtype=np.float32) / 20.0
        pred = sum(weights[i] * preds[name] for i, name in enumerate(names))
        score = metrics(true, pred)["mae"]
        if score < best_score:
            best_score = score
            best_weights = {name: float(weights[i]) for i, name in enumerate(names)}
    return best_weights


def weighted_sum(preds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    first = next(iter(preds.values()))
    out = np.zeros_like(first, dtype=np.float32)
    for name, pred in preds.items():
        out += float(weights.get(name, 0.0)) * pred
    return out


def run_target(
    data: np.ndarray,
    target_meta: dict[str, object],
    interval_minutes: int,
    train_end: int,
    test_start: int,
    slots_per_day: int,
    lags: list[int],
) -> dict[str, object]:
    out: dict[str, object] = {
        "channel": int(target_meta["channel"]),
        "label": str(target_meta["label"]),
        "unit": str(target_meta["unit"]),
        "mape_threshold": float(target_meta["mape_threshold"]),
        "horizons": {},
    }
    mape_threshold = float(target_meta["mape_threshold"])

    for horizon_minutes in [3, 5, 15, 30]:
        horizon_steps = max(1, int(round(horizon_minutes / interval_minutes)))
        positions, x, y = make_supervised(data, horizon_steps, lags)
        train_mask = positions < train_end
        val_mask = (positions >= train_end) & (positions < test_start)
        test_mask = positions >= test_start
        x_train, y_train = x[train_mask], y[train_mask]
        x_val, y_val = x[val_mask], y[val_mask]
        x_test, y_test = x[test_mask], y[test_mask]
        val_target_positions = positions[val_mask] + horizon_steps
        target_positions = positions[test_mask] + horizon_steps

        persistence_val = data[positions[val_mask]]
        persistence_pred = data[positions[test_mask]]
        seasonal_val = seasonal_persistence(data, train_end, val_target_positions, slots_per_day)
        seasonal_pred = seasonal_persistence(data, train_end, target_positions, slots_per_day)
        ha_val = historical_average(data, train_end, val_target_positions, slots_per_day)
        ha_pred = historical_average(data, train_end, target_positions, slots_per_day)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        x_test_scaled = scaler.transform(x_test)
        ridge = Ridge(alpha=10.0)
        ridge.fit(x_train_scaled, y_train)
        ridge_val = ridge.predict(x_val_scaled)
        ridge_pred = ridge.predict(x_test_scaled)
        val_preds = {
            "Persistence": persistence_val,
            "SeasonalPersistence": seasonal_val,
            "HistoricalAverage": ha_val,
            "RidgeLag": ridge_val,
        }
        test_preds = {
            "Persistence": persistence_pred,
            "SeasonalPersistence": seasonal_pred,
            "HistoricalAverage": ha_pred,
            "RidgeLag": ridge_pred,
        }
        fusion_weights = optimize_fusion_weights(y_val, val_preds)
        fusion_pred = weighted_sum(test_preds, fusion_weights)

        out["horizons"][f"{horizon_minutes}min"] = {
            "horizon_minutes": horizon_minutes,
            "horizon_steps": int(horizon_steps),
            "effective_minutes": int(horizon_steps * interval_minutes),
            "train_samples": int(x_train.shape[0]),
            "validation_samples": int(x_val.shape[0]),
            "test_samples": int(x_test.shape[0]),
            "fusion_weights": fusion_weights,
            "models": {
                "Persistence": metrics(y_test, persistence_pred, mape_threshold),
                "SeasonalPersistence": metrics(y_test, seasonal_pred, mape_threshold),
                "HistoricalAverage": metrics(y_test, ha_pred, mape_threshold),
                "RidgeLag": metrics(y_test, ridge_pred, mape_threshold),
                "Ours-TSFusion": metrics(y_test, fusion_pred, mape_threshold),
            },
        }
    return out


def run_dataset(dataset: str, auto_download: bool) -> dict[str, object]:
    meta = DATASETS[dataset]
    interval_minutes = int(meta["interval_minutes"])
    data_path = download_if_needed(dataset, PROJECT_ROOT / "data" / "long_horizon", auto_download)
    raw = np.load(data_path)["data"].astype(np.float32)
    total_steps, sensors = raw[:, :, 0].shape
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
        "lags_steps": lags,
        "targets": {},
    }

    for target_name, target_meta in TARGETS.items():
        channel = int(target_meta["channel"])
        if raw.ndim == 3 and channel < raw.shape[2]:
            results["targets"][target_name] = run_target(
                raw[:, :, channel],
                target_meta,
                interval_minutes,
                train_end,
                test_start,
                slots_per_day,
                lags,
            )
    # Backward-compatible alias for existing report consumers.
    if "flow" in results["targets"]:
        results["target"] = "traffic_flow_channel_0"
        results["horizons"] = results["targets"]["flow"]["horizons"]
    return results


def plot_results(report: dict[str, object], out_path: Path) -> None:
    targets = report.get("targets") or {"flow": {"label": "Traffic flow", "horizons": report["horizons"]}}
    model_order = ["Persistence", "SeasonalPersistence", "HistoricalAverage", "RidgeLag", "Ours-TSFusion"]
    fig, axes = plt.subplots(len(targets), 2, figsize=(12, 4 * len(targets)), squeeze=False)
    colors = {
        "Persistence": "#9aa0a6",
        "SeasonalPersistence": "#b279a2",
        "HistoricalAverage": "#4c78a8",
        "RidgeLag": "#f58518",
        "Ours-TSFusion": "#54a24b",
    }
    for row, (_target_name, target_data) in enumerate(targets.items()):
        horizons = list(target_data["horizons"].keys())
        models = [m for m in model_order if all(m in target_data["horizons"][h]["models"] for h in horizons)]
        x = np.arange(len(horizons))
        width = min(0.16, 0.78 / max(len(models), 1))
        for i, model in enumerate(models):
            mae = [target_data["horizons"][h]["models"][model]["mae"] for h in horizons]
            rmse = [target_data["horizons"][h]["models"][model]["rmse"] for h in horizons]
            offset = (i - (len(models) - 1) / 2) * width
            axes[row][0].bar(x + offset, mae, width=width, label=model, color=colors.get(model))
            axes[row][1].bar(x + offset, rmse, width=width, label=model, color=colors.get(model))
        for ax, ylabel in [(axes[row][0], "MAE"), (axes[row][1], "RMSE")]:
            ax.set_xticks(x)
            ax.set_xticklabels(horizons)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Prediction horizon")
            ax.grid(axis="y", alpha=0.25)
        axes[row][0].set_title(f"{report['dataset']} {target_data.get('label', 'target')} forecasting")
        axes[row][1].set_title("Lower is better")
    axes[0][1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
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
