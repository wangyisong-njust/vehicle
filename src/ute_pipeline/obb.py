from __future__ import annotations

import csv
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
    header: list[str]
    raw_rows: list[dict[str, str]]


def _get(row: dict[str, str], key: str | None, default: float = 0.0) -> float:
    if not key:
        return default
    val = row.get(key, "")
    if val == "":
        return default
    return float(val)


def load_pixel_table(path: Path, columns: dict[str, str], fps: float, frame_offset: int) -> PixelTable:
    rows: list[dict[str, str]] = []
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
        header = reader.fieldnames or []
        for row in reader:
            fr = int(round(_get(row, columns["frame"])))
            rows.append(row)
            frame.append(fr)
            vehicle_id.append(int(round(_get(row, columns["vehicle_id"]))))
            x.append(_get(row, columns["x"]))
            y.append(_get(row, columns["y"]))
            w.append(_get(row, columns["w"]))
            h.append(_get(row, columns["h"]))
            cls.append(int(round(_get(row, columns.get("class"), 0))))
            if columns.get("time"):
                time_s.append(_get(row, columns["time"]))
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
        header=header,
        raw_rows=rows,
    )


def _angle_from_delta(dx: float, dy: float) -> float:
    return math.atan2(dy, dx)


def _fill_vehicle_angles(raw: np.ndarray, fallback: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    theta = raw.copy()
    valid = ~np.isnan(theta)
    if not valid.any():
        return np.full_like(theta, fallback, dtype=np.float32), np.zeros_like(theta, dtype=np.float32)

    valid_values = theta[valid]
    valid_values = np.unwrap(valid_values)
    theta[valid] = valid_values

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
    for idx, vid in enumerate(table.vehicle_id):
        by_vehicle[int(vid)].append(idx)

    direct = 0
    filled = 0
    for indices in by_vehicle.values():
        indices.sort(key=lambda i: int(table.frame[i]))
        raw = np.full(len(indices), np.nan, dtype=np.float32)
        for pos, idx in enumerate(indices):
            best = None
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
                    best = (dx, dy)
                if dist >= min_motion_px:
                    break
            if best is not None and best_dist >= min_motion_px:
                raw[pos] = _angle_from_delta(best[0], best[1])
                direct += 1
        filled_theta, conf = _fill_vehicle_angles(raw)
        for local, idx in enumerate(indices):
            theta[idx] = filled_theta[local]
            confidence[idx] = conf[local]
            if conf[local] < 1.0:
                filled += 1
    stats = {
        "rows": float(table.frame.shape[0]),
        "vehicles": float(len(by_vehicle)),
        "direct_theta_rows": float(direct),
        "filled_theta_rows": float(filled),
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
    cx = table.x[idxs] + table.w[idxs] / 2.0
    cy = table.y[idxs] + table.h[idxs] / 2.0
    ratio = side_ratio[idxs] if side_ratio is not None else None
    theta_vis = theta[idxs].copy()
    if theta_vis.size >= 8:
        # The published sample videos are cropped/downsampled views of the annotation canvas.
        # For visualization only, suppress isolated trajectory-angle outliers by snapping them
        # to the dominant orientation of nearby horizontal road bands. The CSV keeps raw theta.
        def orientation_delta(a: np.ndarray, b: float) -> np.ndarray:
            return np.abs((a - b + math.pi / 2.0) % math.pi - math.pi / 2.0)

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
            dev = orientation_delta(band_theta, dominant)
            local = np.where(mask)[0]
            theta_vis[local[dev > math.radians(30.0)]] = dominant
    corners = obb_corners(cx, cy, table.w[idxs], table.h[idxs], theta_vis, ratio)
    img_h, img_w = image.shape[:2]
    if table.x.size:
        coord_x1 = float(np.min(table.x))
        coord_y1 = float(np.min(table.y))
        coord_x2 = float(np.max(table.x + table.w))
        coord_y2 = float(np.max(table.y + table.h))
    else:
        coord_x1, coord_y1, coord_x2, coord_y2 = 0.0, 0.0, float(img_w), float(img_h)
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
