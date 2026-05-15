#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PixelTable:
    frame: np.ndarray
    vehicle_id: np.ndarray
    x: np.ndarray
    y: np.ndarray
    w: np.ndarray
    h: np.ndarray
    cls: np.ndarray
    time_s: np.ndarray


def _read_float(row: dict[str, str], key: str | None, default: float = 0.0) -> float:
    if not key:
        return default
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def _read_int(row: dict[str, str], key: str | None, default: int = 0) -> int:
    return int(round(_read_float(row, key, float(default))))


def resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def load_pixel_table(
    path: Path,
    columns: dict[str, str],
    fps: float,
    frame_offset: int,
    max_rows: int | None = None,
) -> PixelTable:
    frame: list[int] = []
    vehicle_id: list[int] = []
    x: list[float] = []
    y: list[float] = []
    w: list[float] = []
    h: list[float] = []
    cls: list[int] = []
    time_s: list[float] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            fr = _read_int(row, columns["frame"])
            frame.append(fr)
            vehicle_id.append(_read_int(row, columns["vehicle_id"]))
            x.append(_read_float(row, columns["x"]))
            y.append(_read_float(row, columns["y"]))
            w.append(_read_float(row, columns["w"]))
            h.append(_read_float(row, columns["h"]))
            cls.append(_read_int(row, columns.get("class"), 0))
            if columns.get("time"):
                time_s.append(_read_float(row, columns["time"]))
            else:
                time_s.append((fr - frame_offset + 1) / fps)

    return PixelTable(
        frame=np.asarray(frame, dtype=np.int32),
        vehicle_id=np.asarray(vehicle_id, dtype=np.int32),
        x=np.asarray(x, dtype=np.float32),
        y=np.asarray(y, dtype=np.float32),
        w=np.asarray(w, dtype=np.float32),
        h=np.asarray(h, dtype=np.float32),
        cls=np.asarray(cls, dtype=np.int16),
        time_s=np.asarray(time_s, dtype=np.float32),
    )


def load_vehicle_size_map(path: Path, fps: float, frame_offset: int) -> dict[int, tuple[float, float]]:
    if not path.exists():
        return {}
    values: dict[int, list[tuple[float, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vehicle_id = _read_int(row, "vehicleID", -1)
            length = _read_float(row, "vehicleLength(m)", 4.5)
            width = _read_float(row, "vehicleWidth(m)", 1.8)
            if vehicle_id >= 0 and length > 0 and width > 0:
                values[vehicle_id].append((length, width))
    result: dict[int, tuple[float, float]] = {}
    for vehicle_id, pairs in values.items():
        arr = np.asarray(pairs, dtype=np.float32)
        result[vehicle_id] = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))
    return result


def compute_side_ratio(table: PixelTable, size_map: dict[int, tuple[float, float]]) -> np.ndarray:
    ratio = np.empty(table.frame.shape[0], dtype=np.float32)
    for i, vehicle_id in enumerate(table.vehicle_id):
        length_m, width_m = size_map.get(int(vehicle_id), (4.5, 1.8))
        ratio[i] = float(width_m / max(length_m, 1e-6))
    return ratio


def _fill_vehicle_angles(raw: np.ndarray, fallback: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    theta = raw.copy()
    valid = ~np.isnan(theta)
    if not valid.any():
        return np.full_like(theta, fallback, dtype=np.float32), np.zeros_like(theta, dtype=np.float32)

    theta[valid] = np.unwrap(theta[valid])
    last = np.nan
    for i in range(theta.shape[0]):
        if np.isnan(theta[i]):
            theta[i] = last
        else:
            last = theta[i]

    nxt = np.nan
    for i in range(theta.shape[0] - 1, -1, -1):
        if np.isnan(theta[i]):
            theta[i] = nxt if not np.isnan(nxt) else fallback
        else:
            nxt = theta[i]

    theta = (theta + math.pi) % (2 * math.pi) - math.pi
    confidence = np.where(valid, 1.0, 0.55).astype(np.float32)
    return theta.astype(np.float32), confidence


def compute_theta(
    table: PixelTable,
    min_motion_px: float = 0.75,
    search_radius: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    cx = table.x + table.w / 2.0
    cy = table.y + table.h / 2.0
    theta = np.full(table.frame.shape[0], np.nan, dtype=np.float32)
    confidence = np.zeros(table.frame.shape[0], dtype=np.float32)
    by_vehicle: dict[int, list[int]] = defaultdict(list)
    for i, vehicle_id in enumerate(table.vehicle_id):
        by_vehicle[int(vehicle_id)].append(i)

    direct = 0
    filled = 0
    for indices in by_vehicle.values():
        indices.sort(key=lambda i: int(table.frame[i]))
        raw = np.full(len(indices), np.nan, dtype=np.float32)
        for pos, idx in enumerate(indices):
            best_delta: tuple[float, float] | None = None
            best_dist = -1.0
            for radius in range(1, search_radius + 1):
                left = max(0, pos - radius)
                right = min(len(indices) - 1, pos + radius)
                if left == right:
                    continue
                dx = float(cx[indices[right]] - cx[indices[left]])
                dy = float(cy[indices[right]] - cy[indices[left]])
                dist = math.hypot(dx, dy)
                if dist > best_dist:
                    best_dist = dist
                    best_delta = (dx, dy)
                if dist >= min_motion_px:
                    break
            if best_delta is not None and best_dist >= min_motion_px:
                raw[pos] = math.atan2(best_delta[1], best_delta[0])
                direct += 1
        filled_theta, conf = _fill_vehicle_angles(raw)
        for local, idx in enumerate(indices):
            theta[idx] = filled_theta[local]
            confidence[idx] = conf[local]
            if conf[local] < 1.0:
                filled += 1

    stats = {
        "rows": int(table.frame.shape[0]),
        "vehicles": int(len(by_vehicle)),
        "direct_theta_rows": int(direct),
        "filled_theta_rows": int(filled),
        "direct_theta_rate": float(direct / max(1, table.frame.shape[0])),
    }
    return theta, confidence, stats


def obb_side_lengths(
    w: np.ndarray,
    h: np.ndarray,
    side_ratio: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    long_side = np.maximum(w, h)
    if side_ratio is None:
        short_side = np.minimum(w, h)
    else:
        ratio = np.clip(side_ratio.astype(np.float32), 0.15, 0.95)
        short_side = np.minimum(np.minimum(w, h), long_side * ratio)
    return long_side.astype(np.float32), short_side.astype(np.float32)


def obb_corners(
    cx: np.ndarray,
    cy: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    theta: np.ndarray,
    side_ratio: np.ndarray | None = None,
) -> np.ndarray:
    long_side, short_side = obb_side_lengths(w, h, side_ratio)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    base = np.asarray([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], dtype=np.float32)
    corners = np.empty((cx.shape[0], 4, 2), dtype=np.float32)
    for i, (sx, sy) in enumerate(base):
        px = sx * long_side
        py = sy * short_side
        corners[:, i, 0] = cos_t * px - sin_t * py + cx
        corners[:, i, 1] = sin_t * px + cos_t * py + cy
    return corners


def polygon_area(corners: np.ndarray) -> np.ndarray:
    x = corners[:, :, 0]
    y = corners[:, :, 1]
    return (0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1))).astype(np.float32)


def write_obb_csv(
    path: Path,
    table: PixelTable,
    theta: np.ndarray,
    confidence: np.ndarray,
    side_ratio: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cx = table.x + table.w / 2.0
    cy = table.y + table.h / 2.0
    obb_w, obb_h = obb_side_lengths(table.w, table.h, side_ratio)
    corners = obb_corners(cx, cy, table.w, table.h, theta, side_ratio)
    areas = polygon_area(corners)

    fieldnames = [
        "frame",
        "time_s",
        "vehicle_id",
        "x",
        "y",
        "w",
        "h",
        "class",
        "cx",
        "cy",
        "obb_w",
        "obb_h",
        "obb_area",
        "obb_aspect_ratio",
        "theta",
        "theta_deg",
        "theta_conf",
        "obb_x1",
        "obb_y1",
        "obb_x2",
        "obb_y2",
        "obb_x3",
        "obb_y3",
        "obb_x4",
        "obb_y4",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(table.frame.shape[0]):
            row = {
                "frame": int(table.frame[i]),
                "time_s": f"{float(table.time_s[i]):.6f}",
                "vehicle_id": int(table.vehicle_id[i]),
                "x": f"{float(table.x[i]):.3f}",
                "y": f"{float(table.y[i]):.3f}",
                "w": f"{float(table.w[i]):.3f}",
                "h": f"{float(table.h[i]):.3f}",
                "class": int(table.cls[i]),
                "cx": f"{float(cx[i]):.3f}",
                "cy": f"{float(cy[i]):.3f}",
                "obb_w": f"{float(obb_w[i]):.3f}",
                "obb_h": f"{float(obb_h[i]):.3f}",
                "obb_area": f"{float(areas[i]):.3f}",
                "obb_aspect_ratio": f"{float(obb_h[i] / max(obb_w[i], 1e-6)):.6f}",
                "theta": f"{float(theta[i]):.8f}",
                "theta_deg": f"{math.degrees(float(theta[i])):.3f}",
                "theta_conf": f"{float(confidence[i]):.3f}",
            }
            for j in range(4):
                row[f"obb_x{j + 1}"] = f"{float(corners[i, j, 0]):.3f}"
                row[f"obb_y{j + 1}"] = f"{float(corners[i, j, 1]):.3f}"
            writer.writerow(row)


def _orientation_delta(angle: np.ndarray, base: float) -> np.ndarray:
    return np.abs((angle - base + math.pi / 2.0) % math.pi - math.pi / 2.0)


def _smooth_overlay_theta(theta: np.ndarray, cy: np.ndarray) -> np.ndarray:
    theta_vis = theta.copy()
    if theta_vis.size < 8:
        return theta_vis
    band_edges = np.quantile(cy, np.linspace(0.0, 1.0, min(5, theta_vis.size) + 1))
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        mask = (cy >= lo) & (cy <= hi)
        if int(mask.sum()) < 4:
            continue
        band_theta = theta_vis[mask]
        sin2 = float(np.mean(np.sin(2.0 * band_theta)))
        cos2 = float(np.mean(np.cos(2.0 * band_theta)))
        concentration = math.hypot(sin2, cos2)
        if concentration < 0.35:
            continue
        dominant = 0.5 * math.atan2(sin2, cos2)
        dev = _orientation_delta(band_theta, dominant)
        local = np.where(mask)[0]
        theta_vis[local[dev > math.radians(30.0)]] = dominant
    return theta_vis


def save_overlay(
    video_path: Path,
    out_path: Path,
    table: PixelTable,
    theta: np.ndarray,
    frame: int,
    frame_offset: int,
    side_ratio: np.ndarray | None = None,
    max_boxes: int = 160,
) -> bool:
    if not video_path.exists():
        return False
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    video_idx = max(0, frame - frame_offset)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if video_idx >= frame_count and frame_count > 0 and video_fps > 0:
        frame_rows = np.where(table.frame == frame)[0]
        if frame_rows.size > 0:
            time_s = float(np.median(table.time_s[frame_rows]))
            video_idx = int(round(time_s * video_fps)) - 1
    video_idx = min(max(0, video_idx), max(0, frame_count - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_idx)
    ok, image = cap.read()
    cap.release()
    if not ok:
        return False

    idxs = np.where(table.frame == frame)[0][:max_boxes]
    if idxs.size == 0:
        return False
    cx = table.x[idxs] + table.w[idxs] / 2.0
    cy = table.y[idxs] + table.h[idxs] / 2.0
    ratio = side_ratio[idxs] if side_ratio is not None else None
    theta_vis = _smooth_overlay_theta(theta[idxs], cy)
    corners = obb_corners(cx, cy, table.w[idxs], table.h[idxs], theta_vis, ratio)

    img_h, img_w = image.shape[:2]
    coord_x1 = float(np.min(table.x))
    coord_y1 = float(np.min(table.y))
    coord_x2 = float(np.max(table.x + table.w))
    coord_y2 = float(np.max(table.y + table.h))
    scale_x = img_w / max(coord_x2, 1.0)
    scale_y = img_h / max(coord_y2 - coord_y1, 1.0)
    needs_alignment = (
        abs(coord_y1) > 1.0
        or abs(coord_x2 - img_w) > 1.0
        or abs((coord_y2 - coord_y1) - img_h) > 1.0
    )
    if needs_alignment:
        corners = corners.copy()
        corners[:, :, 0] = corners[:, :, 0] * scale_x
        corners[:, :, 1] = (corners[:, :, 1] - coord_y1) * scale_y
        cx = cx * scale_x
        cy = (cy - coord_y1) * scale_y

    for local, idx in enumerate(idxs):
        pts = np.round(corners[local]).astype(np.int32)
        cv2.polylines(image, [pts], True, (0, 255, 255), 2)
        cv2.circle(image, (int(round(cx[local])), int(round(cy[local]))), 2, (0, 0, 255), -1)
        cv2.putText(
            image,
            str(int(table.vehicle_id[idx])),
            (int(round(cx[local])), int(round(cy[local])) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), image))


def process_dataset(config_path: Path, config: dict, key: str, max_rows: int | None = None) -> dict[str, object]:
    base_dir = config_path.parent
    dataset = config["datasets"][key]
    dataset_root = resolve(base_dir, dataset["root"])
    output_root = resolve(base_dir, config.get("output_dir", "outputs"))
    pixel_path = dataset_root / dataset["pixel_csv"]
    frenet_path = dataset_root / dataset.get("frenet_csv", "")
    video_path = dataset_root / dataset.get("video", "")

    theta_cfg = config.get("theta", {})
    print(f"[OBB] loading {dataset['name']} pixel table: {pixel_path}")
    table = load_pixel_table(
        pixel_path,
        dataset["pixel_columns"],
        float(dataset["fps"]),
        int(dataset["frame_offset"]),
        max_rows=max_rows,
    )
    theta, confidence, stats = compute_theta(
        table,
        min_motion_px=float(theta_cfg.get("min_motion_px", 0.75)),
        search_radius=int(theta_cfg.get("search_radius", 8)),
    )

    size_map = load_vehicle_size_map(frenet_path, float(dataset["fps"]), int(dataset["frame_offset"]))
    side_ratio = compute_side_ratio(table, size_map)

    out_csv = output_root / "processed" / f"{key}_pixel_obb.csv"
    print(f"[OBB] writing OBB csv: {out_csv}")
    write_obb_csv(out_csv, table, theta, confidence, side_ratio=side_ratio)

    overlays: list[str] = []
    unique_frames = np.unique(table.frame)
    if unique_frames.size > 0:
        sample_frames = [
            int(unique_frames[len(unique_frames) // 4]),
            int(unique_frames[len(unique_frames) // 2]),
            int(unique_frames[(len(unique_frames) * 3) // 4]),
        ]
        for frame in sample_frames:
            out_img = output_root / "figures" / f"{key}_obb_overlay_f{frame}.jpg"
            if save_overlay(video_path, out_img, table, theta, frame, int(dataset["frame_offset"]), side_ratio=side_ratio):
                overlays.append(str(out_img.relative_to(base_dir)))
                print(f"[OBB] wrote overlay: {out_img}")

    result: dict[str, object] = {
        **stats,
        "dataset": dataset["name"],
        "obb_csv": str(out_csv.relative_to(base_dir)),
        "sample_overlays": overlays,
    }
    print(f"[OBB] {dataset['name']} direct theta rate: {result['direct_theta_rate']:.3f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert UTE HBB pixel.csv annotations to OBB annotations.")
    parser.add_argument("--config", default="config_ute_sample.json", help="Path to config JSON.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["xamn5", "xamn6", "pkdd8"],
        help="Dataset keys in config JSON. Defaults to all configured datasets.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional small trial mode; omit for full conversion.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    summary: dict[str, object] = {}
    for key in args.datasets:
        if key not in config["datasets"]:
            raise KeyError(f"Dataset key not found in config: {key}")
        summary[key] = process_dataset(config_path, config, key, max_rows=args.max_rows)

    output_root = resolve(config_path.parent, config.get("output_dir", "outputs"))
    summary_path = output_root / "reports" / "obb_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OBB] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
