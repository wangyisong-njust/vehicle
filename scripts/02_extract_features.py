#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json

from ute_pipeline.config import load_config, project_root
from ute_pipeline.features import extract_window_features, load_frenet, load_obb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets.json")
    parser.add_argument("--datasets", nargs="+", default=["xamn6", "xamn5", "pkdd8"])
    args = parser.parse_args()

    root = project_root()
    cfg = load_config(args.config)
    feature_cfg = cfg["feature"]
    all_rows = []
    fieldnames = None
    summary = {}
    for key in args.datasets:
        ds = cfg["datasets"][key]
        ds_root = root / ds["root"]
        print(f"[FEATURE] loading {ds['name']}")
        frenet = load_frenet(ds_root / ds["frenet_csv"], float(ds["fps"]), int(ds["frame_offset"]))
        obb = load_obb(root / "outputs" / "processed" / f"{key}_pixel_obb.csv")
        fields, rows = extract_window_features(
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
        out_path = root / "outputs" / "features" / f"{key}_windows.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[FEATURE] wrote {len(rows)} windows to {out_path}")
        all_rows.extend(rows)
        summary[key] = {"windows": len(rows), "feature_csv": str(out_path.relative_to(root))}

    merged_path = root / "outputs" / "features" / "all_windows.csv"
    with merged_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    summary["merged"] = {"windows": len(all_rows), "feature_csv": str(merged_path.relative_to(root))}
    summary_path = root / "outputs" / "reports" / "feature_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[FEATURE] merged features saved to {merged_path}")


if __name__ == "__main__":
    main()
