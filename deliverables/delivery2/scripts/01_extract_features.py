#!/usr/bin/env python
"""第一步：从 OBB 标注表与轨迹表中提取滑窗特征 + 4 通道网格张量。

输入：
  - data/UTE/datasets/{ds}/frenet.csv
  - data/UTE/datasets/{ds}/sample video.mp4
  - inputs/{ds}_pixel_obb.csv          ← 来自第一部分交付包

输出：
  - outputs/features/{ds}_windows.csv     ← 滑窗特征表
  - outputs/features/{ds}_grid_tensors.npz ← 每窗 4×4×12 张量
  - outputs/features/all_windows.csv      ← 全部数据集合并
  - outputs/reports/feature_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from features import extract_window_features, load_frenet, load_obb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "datasets.json"))
    parser.add_argument("--datasets", nargs="+", default=["xamn6", "xamn5", "pkdd8"])
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    feature_cfg = cfg["feature"]

    fieldnames = None
    all_rows: list[dict] = []
    summary: dict[str, object] = {}

    for key in args.datasets:
        ds = cfg["datasets"][key]
        ds_root = ROOT / ds["root"]
        print(f"[FEATURE] loading {ds['name']}")
        frenet = load_frenet(ds_root / ds["frenet_csv"], float(ds["fps"]), int(ds["frame_offset"]))
        obb_csv = ROOT / "inputs" / f"{key}_pixel_obb.csv"
        if not obb_csv.exists():
            raise FileNotFoundError(
                f"Missing OBB pixel table {obb_csv}. "
                "Run delivery 1 (hbb_to_obb_minimal) first and copy the produced "
                f"outputs/processed/{key}_pixel_obb.csv into inputs/."
            )
        obb = load_obb(obb_csv)
        fields, rows, grid_tensors = extract_window_features(
            dataset_key=key,
            dataset_name=ds["name"],
            frenet=frenet,
            obb=obb,
            video_path=ds_root / ds["video"],
            segment_length_m=float(ds["segment_length_m"]),
            speed_limit_kmh=float(ds["speed_limit_kmh"]),
            window_s=float(feature_cfg["window_s"]),
            step_s=float(feature_cfg["step_s"]),
            safe_headway_s=float(feature_cfg.get("safe_headway_s", 2.0)),
            mgti_sensitivity=float(feature_cfg.get("mgti_sensitivity", 1.0)),
            grid_cols=int(feature_cfg.get("grid_cols", 12)),
            grid_rows=int(feature_cfg.get("grid_rows", 4)),
        )
        fieldnames = fields
        out_csv = ROOT / "outputs" / "features" / f"{key}_windows.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[FEATURE] wrote {len(rows)} windows -> {out_csv}")

        tensor_path = ROOT / "outputs" / "features" / f"{key}_grid_tensors.npz"
        start_s = np.asarray([float(r["start_s"]) for r in rows], dtype=np.float32)
        window_id = np.asarray([int(r["window_id"]) for r in rows], dtype=np.int32)
        np.savez_compressed(
            tensor_path,
            tensors=grid_tensors,
            start_s=start_s,
            window_id=window_id,
            channel_names=np.asarray(["obb_occupancy", "hbb_occupancy", "theta_sin", "theta_cos"]),
        )
        print(f"[FEATURE] wrote tensors {grid_tensors.shape} -> {tensor_path}")

        all_rows.extend(rows)
        summary[key] = {
            "windows": len(rows),
            "feature_csv": str(out_csv.relative_to(ROOT)),
            "grid_tensor_npz": str(tensor_path.relative_to(ROOT)),
            "grid_tensor_shape": list(grid_tensors.shape),
        }

    merged_path = ROOT / "outputs" / "features" / "all_windows.csv"
    with merged_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    summary["merged"] = {"windows": len(all_rows), "feature_csv": str(merged_path.relative_to(ROOT))}

    summary_path = ROOT / "outputs" / "reports" / "feature_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[FEATURE] summary -> {summary_path}")


if __name__ == "__main__":
    main()
