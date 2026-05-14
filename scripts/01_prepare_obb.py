#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ute_pipeline.config import load_config, project_root
from ute_pipeline.features import load_frenet, vehicle_size_map
from ute_pipeline.obb import compute_theta, load_pixel_table, save_overlay, write_obb_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets.json")
    parser.add_argument("--datasets", nargs="+", default=["xamn6", "xamn5", "pkdd8"])
    args = parser.parse_args()

    root = project_root()
    cfg = load_config(args.config)
    feature_cfg = cfg["feature"]
    summary = {}
    for key in args.datasets:
        ds = cfg["datasets"][key]
        ds_root = root / ds["root"]
        pixel_path = ds_root / ds["pixel_csv"]
        video_path = ds_root / ds["video"]
        print(f"[OBB] loading {ds['name']} from {pixel_path}")
        table = load_pixel_table(
            pixel_path,
            ds["pixel_columns"],
            float(ds["fps"]),
            int(ds["frame_offset"]),
        )
        theta, conf, stats = compute_theta(
            table,
            min_motion_px=float(feature_cfg["min_motion_px"]),
            search_radius=int(feature_cfg["theta_search_radius"]),
        )
        frenet = load_frenet(ds_root / ds["frenet_csv"], float(ds["fps"]), int(ds["frame_offset"]))
        sizes = vehicle_size_map(frenet)
        side_ratio = np.empty(table.frame.shape[0], dtype=np.float32)
        for i, vid in enumerate(table.vehicle_id):
            length_m, width_m = sizes.get(int(vid), (4.5, 1.8))
            side_ratio[i] = float(width_m / max(length_m, 1e-6))
        out_csv = root / "outputs" / "processed" / f"{key}_pixel_obb.csv"
        print(f"[OBB] writing {out_csv}")
        write_obb_csv(out_csv, table, theta, conf, side_ratio=side_ratio)

        frames = np.unique(table.frame)
        sample_frames = [
            int(frames[len(frames) // 4]),
            int(frames[len(frames) // 2]),
            int(frames[(len(frames) * 3) // 4]),
        ]
        overlays = []
        for fr in sample_frames:
            out_img = root / "outputs" / "figures" / f"{key}_obb_overlay_f{fr}.jpg"
            if save_overlay(video_path, out_img, table, theta, fr, int(ds["frame_offset"]), side_ratio=side_ratio):
                overlays.append(str(out_img.relative_to(root)))
        stats["sample_overlays"] = overlays
        summary[key] = stats
        print(f"[OBB] {ds['name']} direct theta rate: {stats['direct_theta_rate']:.3f}")

    summary_path = root / "outputs" / "reports" / "obb_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OBB] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
