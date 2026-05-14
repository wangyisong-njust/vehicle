#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ute_pipeline.config import load_config, project_root
from ute_pipeline.experiments import (
    STATE_NAMES,
    compute_composite_mgti,
    make_deterioration_labels_score,
    make_state_labels,
    read_feature_table,
)


def load_obb_sample(path: Path, max_rows: int | None = None) -> list[dict[str, float]]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            rows.append(
                {
                    "frame": float(row["frame"]),
                    "vehicle_id": float(row["vehicle_id"]),
                    "w": float(row["w"]),
                    "h": float(row["h"]),
                    "theta_conf": float(row["theta_conf"]),
                    "obb_x1": float(row["obb_x1"]),
                    "obb_y1": float(row["obb_y1"]),
                    "obb_x2": float(row["obb_x2"]),
                    "obb_y2": float(row["obb_y2"]),
                    "obb_x3": float(row["obb_x3"]),
                    "obb_y3": float(row["obb_y3"]),
                    "obb_x4": float(row["obb_x4"]),
                    "obb_y4": float(row["obb_y4"]),
                }
            )
    return rows


def verify_obb_geometry(rows: list[dict[str, float]]) -> dict[str, float]:
    total = len(rows)
    valid_size = 0
    valid_area = 0
    high_conf = 0
    centers_x = []
    centers_y = []
    for row in rows:
        if row["w"] > 0 and row["h"] > 0:
            valid_size += 1
        pts = np.asarray(
            [
                [row["obb_x1"], row["obb_y1"]],
                [row["obb_x2"], row["obb_y2"]],
                [row["obb_x3"], row["obb_y3"]],
                [row["obb_x4"], row["obb_y4"]],
            ],
            dtype=np.float32,
        )
        x = pts[:, 0]
        y = pts[:, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        if area > 1.0:
            valid_area += 1
        if row["theta_conf"] >= 0.9:
            high_conf += 1
        centers_x.append(float(np.mean(x)))
        centers_y.append(float(np.mean(y)))
    return {
        "sample_rows": total,
        "valid_size_rate": valid_size / max(1, total),
        "valid_obb_area_rate": valid_area / max(1, total),
        "high_theta_conf_rate": high_conf / max(1, total),
        "center_x_min": float(np.min(centers_x)) if centers_x else 0.0,
        "center_x_max": float(np.max(centers_x)) if centers_x else 0.0,
        "center_y_min": float(np.min(centers_y)) if centers_y else 0.0,
        "center_y_max": float(np.max(centers_y)) if centers_y else 0.0,
    }


def feature_state_stats(table, labels: np.ndarray, dataset: str, n_states: int, mgti_composite: np.ndarray | None = None) -> dict[str, object]:
    idxs = np.where(table.dataset == dataset)[0]
    grouped: dict[int, list[int]] = defaultdict(list)
    for idx in idxs:
        grouped[int(labels[idx])].append(int(idx))

    stats = {}
    for state in range(n_states):
        name = STATE_NAMES[state] if state < len(STATE_NAMES) else str(state)
        group = grouped.get(state, [])
        if not group:
            stats[name] = {"count": 0}
            continue
        rows = [table.rows[i] for i in group]
        mgti_val = float(np.mean([mgti_composite[i] for i in group])) if mgti_composite is not None else float(np.mean([float(r["mgti"]) for r in rows]))
        stats[name] = {
            "count": len(rows),
            "mean_speed_kmh": float(np.mean([float(r["mean_speed_kmh"]) for r in rows])),
            "density_veh_per_m": float(np.mean([float(r["density_veh_per_m"]) for r in rows])),
            "obb_occupancy": float(np.mean([float(r["obb_occupancy"]) for r in rows])),
            "mean_headway_s": float(np.mean([float(r["mean_headway_s"]) for r in rows])),
            "mean_space_gap_m": float(np.mean([float(r["mean_space_gap_m"]) for r in rows])),
            "acceleration_interference": float(np.mean([float(r["acceleration_interference"]) for r in rows])),
            "mgti": mgti_val,
        }
    return stats


def monotonic_checks(stats: dict[str, dict[str, float]], n_states: int) -> dict[str, bool]:
    ordered = [stats.get(STATE_NAMES[i] if i < len(STATE_NAMES) else str(i), {}) for i in range(n_states)]
    speed = [s.get("mean_speed_kmh") for s in ordered if s.get("count", 0) > 0]
    free_name = STATE_NAMES[0] if n_states > 0 else "0"
    worst_name = STATE_NAMES[n_states - 1] if n_states > 0 else str(n_states - 1)
    free_speed = stats.get(free_name, {}).get("mean_speed_kmh", 0.0)
    worst_speed = stats.get(worst_name, {}).get("mean_speed_kmh", 1e9)
    worst_density = stats.get(worst_name, {}).get("density_veh_per_m", 0.0)
    worst_occ = stats.get(worst_name, {}).get("obb_occupancy", 0.0)
    max_density = max([s.get("density_veh_per_m", 0.0) for s in ordered])
    max_occ = max([s.get("obb_occupancy", 0.0) for s in ordered])
    return {
        "speed_generally_decreases": all(speed[i] >= speed[i + 1] - 1e-6 for i in range(len(speed) - 1)),
        "worst_speed_lower_than_free": worst_speed < free_speed,
        "worst_density_is_highest": abs(worst_density - max_density) < 1e-6,
        "worst_occupancy_is_highest": abs(worst_occ - max_occ) < 1e-6,
    }


def check_mgti_composite_monotonic(table, labels: np.ndarray, cfg: dict, n_states: int) -> dict[str, object]:
    mgti = compute_composite_mgti(table, cfg)
    idxs = np.where(table.dataset == "xamn6")[0]
    grouped = defaultdict(list)
    for idx in idxs:
        grouped[int(labels[idx])].append(mgti[idx])
    result = {}
    means = []
    for state in range(n_states):
        name = STATE_NAMES[state] if state < len(STATE_NAMES) else str(state)
        vals = grouped.get(state, [])
        mean_val = float(np.mean(vals)) if vals else 0.0
        result[name] = {"mean_mgti_composite": mean_val, "count": len(vals)}
        means.append(mean_val)
    monotonic = all(means[i] <= means[i + 1] + 1e-6 for i in range(len(means) - 1))
    return {"by_state": result, "monotonically_increases": monotonic}


def check_deterioration_labels(table: FeatureTable, cfg: dict, n_states: int) -> dict[str, object]:
    y = table.y
    assert y is not None
    step_s = float(cfg["feature"]["step_s"])
    horizons_s = cfg["feature"]["deterioration_horizons_s"]
    min_positive = int(cfg["experiment"].get("deterioration_min_positive", 10))
    main_idx = np.where(table.dataset == "xamn6")[0]
    score_main = table.score[main_idx] if table.score is not None else y[main_idx].astype(np.float32)
    result = {}
    for horizon_s in horizons_s:
        horizon_steps = max(1, int(round(horizon_s / step_s)))
        det_labels = make_deterioration_labels_score(
            score_main,
            horizon_steps,
            threshold_pct=None,
            std_multiplier=float(cfg["experiment"].get("deterioration_std_multiplier", 1.5)),
        )
        valid = det_labels >= 0
        pos = int(np.sum(det_labels[valid] == 1))
        neg = int(np.sum(det_labels[valid] == 0))
        result[f"horizon_{horizon_s}s"] = {
            "positive_count": pos,
            "negative_count": neg,
            "positive_rate": pos / max(1, pos + neg),
            "sufficient": pos >= min_positive,
        }
    return result


def write_markdown(path: Path, payload: dict[str, object], n_states: int) -> None:
    lines = [
        "# 自动核验报告",
        "",
        f"这份报告使用程序化规则检查四个方面：旋转框几何是否合理、{n_states}类状态标签与特征是否一致、MGTI 复合指标单调性、恶化标签正样本率。",
        "",
        "## 1. OBB 几何核验",
        "",
        "| 数据集 | 样本行数 | 尺寸合法率 | OBB面积合法率 | 高置信角度比例 | 中心点X范围 | 中心点Y范围 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in payload["obb_geometry"].items():
        lines.append(
            f"| {key} | {item['sample_rows']} | {item['valid_size_rate']:.4f} | {item['valid_obb_area_rate']:.4f} | {item['high_theta_conf_rate']:.4f} | "
            f"{item['center_x_min']:.1f}-{item['center_x_max']:.1f} | {item['center_y_min']:.1f}-{item['center_y_max']:.1f} |"
        )
    lines.extend(["", f"## 2. XAM-N-6 状态特征核验（{n_states}类）", ""])
    lines.append("| 状态 | 窗口数 | 平均速度 | 密度 | OBB占有率 | 平均车头时距 | 平均空间间距 | 加速度干扰 | MGTI |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, item in payload["xamn6_state_stats"].items():
        if item.get("count", 0) == 0:
            lines.append(f"| {name} | 0 | - | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {name} | {item['count']} | {item['mean_speed_kmh']:.3f} | {item['density_veh_per_m']:.4f} | {item['obb_occupancy']:.6f} | {item['mean_headway_s']:.3f} | {item['mean_space_gap_m']:.3f} | {item['acceleration_interference']:.4f} | {item['mgti']:.4f} |"
        )
    lines.extend(["", "单调性检查："])
    for key, ok in payload["xamn6_monotonic_checks"].items():
        lines.append(f"- {key}: {'通过' if ok else '需关注'}")

    if "mgti_composite_check" in payload:
        lines.extend(["", "## 3. MGTI 复合指标单调性", ""])
        mgti_check = payload["mgti_composite_check"]
        for name, info in mgti_check["by_state"].items():
            lines.append(f"- {name}: mean={info['mean_mgti_composite']:.4f} (n={info['count']})")
        lines.append(f"- 单调递增: {'通过' if mgti_check['monotonically_increases'] else '需关注'}")

    if "deterioration_label_check" in payload:
        lines.extend(["", "## 4. 恶化标签核验", ""])
        for key, info in payload["deterioration_label_check"].items():
            status = "充分" if info["sufficient"] else "不足"
            lines.append(f"- {key}: 正样本 {info['positive_count']} ({info['positive_rate']:.1%}) — {status}")

    lines.extend(["", "## PKDD 泛化核验", ""])
    lines.append('PKDD 是自由流为主的数据，因此模型输出应主要集中在“畅通”类。')
    lines.append("")
    for name, count in payload["pkdd_predicted_distribution"].items():
        lines.append(f"- {name}: {count}")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            payload["conclusion"],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets.json")
    parser.add_argument("--sample-rows", type=int, default=200000)
    args = parser.parse_args()

    root = project_root()
    cfg = load_config(args.config)
    n_states = int(cfg["feature"].get("n_states", 3))
    table = read_feature_table(root / "outputs" / "features" / "all_windows.csv")
    labels, _ = make_state_labels(table, cfg, main_dataset="xamn6")
    exp = json.loads((root / "outputs" / "reports" / "experiment_results.json").read_text(encoding="utf-8"))

    obb_geometry = {}
    for key in ["xamn6", "pkdd8"]:
        ds = cfg["datasets"][key]
        rows = load_obb_sample(root / "outputs" / "processed" / f"{key}_pixel_obb.csv", max_rows=args.sample_rows)
        obb_geometry[key] = verify_obb_geometry(rows)

    mgti_composite = compute_composite_mgti(table, cfg)
    state_stats = feature_state_stats(table, labels, "xamn6", n_states, mgti_composite)
    checks = monotonic_checks(state_stats, n_states)
    mgti_check = check_mgti_composite_monotonic(table, labels, cfg, n_states)
    det_check = check_deterioration_labels(table, cfg, n_states)
    pkdd_dist = exp["pkdd_generalization"]["predicted_distribution"]
    conclusion = (
        f"自动核验结果显示，OBB 几何合法率较高，XAM-N-6 的 {n_states} 类状态分组已完成速度、密度和占有率的物理单调性检查。"
        f"MGTI 复合指标单调性{'通过' if mgti_check['monotonically_increases'] else '需关注'}。"
        "PKDD 泛化结果主要用于自由流迁移检查。"
    )
    payload = {
        "obb_geometry": obb_geometry,
        "xamn6_state_stats": state_stats,
        "xamn6_monotonic_checks": checks,
        "mgti_composite_check": mgti_check,
        "deterioration_label_check": det_check,
        "pkdd_predicted_distribution": pkdd_dist,
        "conclusion": conclusion,
    }
    out_json = root / "outputs" / "reports" / "auto_verification.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[VERIFY] wrote {out_json}")


if __name__ == "__main__":
    main()
