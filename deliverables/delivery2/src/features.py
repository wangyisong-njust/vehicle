from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FrenetData:
    frame: np.ndarray
    time_s: np.ndarray
    vehicle_id: np.ndarray
    lane_id: np.ndarray
    longitudinal_m: np.ndarray
    lateral_m: np.ndarray
    speed_kmh: np.ndarray
    acceleration: np.ndarray
    vehicle_length_m: np.ndarray
    vehicle_width_m: np.ndarray


@dataclass
class ObbData:
    frame: np.ndarray
    time_s: np.ndarray
    vehicle_id: np.ndarray
    x: np.ndarray
    y: np.ndarray
    w: np.ndarray
    h: np.ndarray
    obb_area: np.ndarray
    theta: np.ndarray
    theta_conf: np.ndarray


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    val = row.get(key, "")
    if val == "":
        return default
    return float(val)


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    val = row.get(key, "")
    if val == "":
        return default
    return int(round(float(val)))


def load_frenet(path: Path, fps: float, frame_offset: int) -> FrenetData:
    frame: list[int] = []
    time_s: list[float] = []
    vehicle_id: list[int] = []
    lane_id: list[int] = []
    longitudinal: list[float] = []
    lateral: list[float] = []
    speed: list[float] = []
    acc: list[float] = []
    length: list[float] = []
    width: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("frame", "") != "":
                fr = _int(row, "frame", 0)
            else:
                fr = int(round(_float(row, "time(s)", 0.0) * fps)) + frame_offset - 1
            frame.append(fr)
            time_s.append(_float(row, "time(s)", (fr - frame_offset + 1) / fps))
            vehicle_id.append(_int(row, "vehicleID"))
            lane_id.append(_int(row, "laneID"))
            longitudinal.append(_float(row, "longitudinalDistance(m)"))
            lateral.append(_float(row, "lateralDistance(m)"))
            speed.append(_float(row, "speed(km/h)"))
            acc.append(_float(row, "acceleration(m/s^2)"))
            length.append(_float(row, "vehicleLength(m)", 4.5))
            width.append(_float(row, "vehicleWidth(m)", 1.8))
    return FrenetData(
        frame=np.asarray(frame, dtype=np.int32),
        time_s=np.asarray(time_s, dtype=np.float32),
        vehicle_id=np.asarray(vehicle_id, dtype=np.int32),
        lane_id=np.asarray(lane_id, dtype=np.int16),
        longitudinal_m=np.asarray(longitudinal, dtype=np.float32),
        lateral_m=np.asarray(lateral, dtype=np.float32),
        speed_kmh=np.asarray(speed, dtype=np.float32),
        acceleration=np.asarray(acc, dtype=np.float32),
        vehicle_length_m=np.asarray(length, dtype=np.float32),
        vehicle_width_m=np.asarray(width, dtype=np.float32),
    )


def load_obb(path: Path) -> ObbData:
    frame: list[int] = []
    time_s: list[float] = []
    vehicle_id: list[int] = []
    x: list[float] = []
    y: list[float] = []
    w: list[float] = []
    h: list[float] = []
    obb_area: list[float] = []
    theta: list[float] = []
    theta_conf: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame.append(_int(row, "frame"))
            time_s.append(_float(row, "time_s"))
            vehicle_id.append(_int(row, "vehicle_id"))
            x.append(_float(row, "x"))
            y.append(_float(row, "y"))
            w.append(_float(row, "w"))
            h.append(_float(row, "h"))
            obb_area.append(_float(row, "obb_area", _float(row, "w") * _float(row, "h")))
            theta.append(_float(row, "theta"))
            theta_conf.append(_float(row, "theta_conf", 1.0))
    return ObbData(
        frame=np.asarray(frame, dtype=np.int32),
        time_s=np.asarray(time_s, dtype=np.float32),
        vehicle_id=np.asarray(vehicle_id, dtype=np.int32),
        x=np.asarray(x, dtype=np.float32),
        y=np.asarray(y, dtype=np.float32),
        w=np.asarray(w, dtype=np.float32),
        h=np.asarray(h, dtype=np.float32),
        obb_area=np.asarray(obb_area, dtype=np.float32),
        theta=np.asarray(theta, dtype=np.float32),
        theta_conf=np.asarray(theta_conf, dtype=np.float32),
    )


def video_shape(video_path: Path) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, fps, frames


def vehicle_size_map(frenet: FrenetData) -> dict[int, tuple[float, float]]:
    sizes: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for vid, length, width in zip(frenet.vehicle_id, frenet.vehicle_length_m, frenet.vehicle_width_m):
        if length > 0 and width > 0:
            sizes[int(vid)].append((float(length), float(width)))
    result = {}
    for vid, vals in sizes.items():
        arr = np.asarray(vals, dtype=np.float32)
        result[vid] = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))
    return result


def lane_change_times(frenet: FrenetData) -> np.ndarray:
    by_vehicle: dict[int, list[int]] = defaultdict(list)
    for idx, vid in enumerate(frenet.vehicle_id):
        by_vehicle[int(vid)].append(idx)
    events: list[float] = []
    for indices in by_vehicle.values():
        indices.sort(key=lambda i: float(frenet.time_s[i]))
        last_lane = None
        for idx in indices:
            lane = int(frenet.lane_id[idx])
            if last_lane is not None and lane != last_lane:
                events.append(float(frenet.time_s[idx]))
            last_lane = lane
    return np.asarray(sorted(events), dtype=np.float32)


def _window_slices(times: np.ndarray, starts: np.ndarray, window_s: float) -> list[tuple[int, int]]:
    order = np.argsort(times)
    sorted_times = times[order]
    slices = []
    for start in starts:
        left = int(np.searchsorted(sorted_times, start, side="left"))
        right = int(np.searchsorted(sorted_times, start + window_s, side="left"))
        slices.append((left, right))
    return order, slices


def _safe_mean(values: np.ndarray, default: float = 0.0) -> float:
    if values.size == 0:
        return default
    return float(np.mean(values))


def _safe_std(values: np.ndarray, default: float = 0.0) -> float:
    if values.size <= 1:
        return default
    return float(np.std(values))


def _circular_std(theta: np.ndarray) -> float:
    if theta.size <= 1:
        return 0.0
    c = float(np.mean(np.cos(theta)))
    s = float(np.mean(np.sin(theta)))
    r = min(1.0, max(1e-8, math.hypot(c, s)))
    return float(math.sqrt(max(0.0, -2.0 * math.log(r))))


def _headway_metrics(frenet: FrenetData, indices: np.ndarray, default_headway_s: float) -> dict[str, float]:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx in indices:
        groups[(int(frenet.frame[idx]), int(frenet.lane_id[idx]))].append(int(idx))

    time_headways: list[float] = []
    space_gaps: list[float] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda i: float(frenet.longitudinal_m[i]))
        for follower, leader in zip(group[:-1], group[1:]):
            gap = float(frenet.longitudinal_m[leader] - frenet.longitudinal_m[follower])
            gap -= 0.5 * float(frenet.vehicle_length_m[leader] + frenet.vehicle_length_m[follower])
            if gap <= 0:
                continue
            speed_mps = max(float(frenet.speed_kmh[follower]) / 3.6, 0.1)
            thw = gap / speed_mps
            if math.isfinite(thw) and 0.0 < thw < 20.0:
                time_headways.append(thw)
                space_gaps.append(gap)

    if not time_headways:
        return {
            "mean_headway_s": float(default_headway_s),
            "min_headway_s": float(default_headway_s),
            "mean_space_gap_m": 0.0,
            "headway_sample_count": 0.0,
        }
    thw_arr = np.asarray(time_headways, dtype=np.float32)
    gap_arr = np.asarray(space_gaps, dtype=np.float32)
    return {
        "mean_headway_s": float(np.mean(thw_arr)),
        "min_headway_s": float(np.min(thw_arr)),
        "mean_space_gap_m": float(np.mean(gap_arr)),
        "headway_sample_count": float(thw_arr.size),
    }


def _grid_stats_from_points(
    px: np.ndarray,
    py: np.ndarray,
    width: int,
    height: int,
    grid_cols: int,
    grid_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, samples = px.shape
    cell_count = grid_cols * grid_rows
    gx = np.floor(px / max(width, 1) * grid_cols).astype(np.int32)
    gy = np.floor(py / max(height, 1) * grid_rows).astype(np.int32)
    valid = (gx >= 0) & (gx < grid_cols) & (gy >= 0) & (gy < grid_rows)
    cell_id = gy * grid_cols + gx

    counts = np.zeros((n, cell_count), dtype=np.uint8)
    row_ids = np.repeat(np.arange(n, dtype=np.int32), samples)
    flat_valid = valid.reshape(-1)
    np.add.at(counts, (row_ids[flat_valid], cell_id.reshape(-1)[flat_valid]), 1)

    totals = counts.sum(axis=1).astype(np.float32)
    active = (counts > 0).sum(axis=1).astype(np.float32)
    peak = counts.max(axis=1).astype(np.float32) / np.maximum(totals, 1.0)
    probs = counts.astype(np.float32) / np.maximum(totals[:, None], 1.0)
    entropy = -(np.where(probs > 0, probs * np.log(probs + 1e-9), 0.0).sum(axis=1))
    entropy = entropy / math.log(cell_count)
    no_samples = totals <= 0
    active[no_samples] = 0.0
    peak[no_samples] = 0.0
    entropy[no_samples] = 0.0
    return active, entropy.astype(np.float32), peak


def _polygon_area(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    acc = 0.0
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        acc += x1 * y2 - x2 * y1
    return abs(acc) * 0.5


def _clip_polygon_to_rect(
    poly: list[tuple[float, float]],
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> list[tuple[float, float]]:
    def clip_edge(
        points: list[tuple[float, float]],
        inside,
        intersect,
    ) -> list[tuple[float, float]]:
        if not points:
            return []
        out: list[tuple[float, float]] = []
        prev = points[-1]
        prev_in = inside(prev)
        for curr in points:
            curr_in = inside(curr)
            if curr_in:
                if not prev_in:
                    out.append(intersect(prev, curr))
                out.append(curr)
            elif prev_in:
                out.append(intersect(prev, curr))
            prev, prev_in = curr, curr_in
        return out

    def ix(x_edge: float):
        def fn(p1: tuple[float, float], p2: tuple[float, float]) -> tuple[float, float]:
            x1, y1 = p1
            x2, y2 = p2
            t = (x_edge - x1) / (x2 - x1 + 1e-12)
            return (x_edge, y1 + t * (y2 - y1))
        return fn

    def iy(y_edge: float):
        def fn(p1: tuple[float, float], p2: tuple[float, float]) -> tuple[float, float]:
            x1, y1 = p1
            x2, y2 = p2
            t = (y_edge - y1) / (y2 - y1 + 1e-12)
            return (x1 + t * (x2 - x1), y_edge)
        return fn

    clipped = clip_edge(poly, lambda p: p[0] >= left, ix(left))
    clipped = clip_edge(clipped, lambda p: p[0] <= right, ix(right))
    clipped = clip_edge(clipped, lambda p: p[1] >= top, iy(top))
    clipped = clip_edge(clipped, lambda p: p[1] <= bottom, iy(bottom))
    return clipped


def _rect_grid_stats(
    polygons: list[list[tuple[float, float]]],
    width: int,
    height: int,
    grid_cols: int,
    grid_rows: int,
    keep_cells: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    n = len(polygons)
    cell_count = grid_cols * grid_rows
    cell_w = width / max(1, grid_cols)
    cell_h = height / max(1, grid_rows)
    cell_area = cell_w * cell_h
    active = np.zeros(n, dtype=np.float32)
    entropy = np.zeros(n, dtype=np.float32)
    peak = np.zeros(n, dtype=np.float32)
    clipped_area = np.zeros(n, dtype=np.float32)
    cell_matrix = np.zeros((n, cell_count), dtype=np.float32) if keep_cells else None

    for i, poly in enumerate(polygons):
        if not poly:
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        col0 = max(0, int(math.floor(min(xs) / max(cell_w, 1e-6))))
        col1 = min(grid_cols - 1, int(math.floor(max(xs) / max(cell_w, 1e-6))))
        row0 = max(0, int(math.floor(min(ys) / max(cell_h, 1e-6))))
        row1 = min(grid_rows - 1, int(math.floor(max(ys) / max(cell_h, 1e-6))))
        if col1 < col0 or row1 < row0:
            continue
        cell_areas: list[float] = []
        for gy in range(row0, row1 + 1):
            top = gy * cell_h
            bottom = min(height, (gy + 1) * cell_h)
            for gx in range(col0, col1 + 1):
                left = gx * cell_w
                right = min(width, (gx + 1) * cell_w)
                area = _polygon_area(_clip_polygon_to_rect(poly, left, top, right, bottom))
                if area > 1e-6:
                    frac = area / max(cell_area, 1e-6)
                    cell_areas.append(frac)
                    if cell_matrix is not None:
                        cell_matrix[i, gy * grid_cols + gx] = frac
        if not cell_areas:
            continue
        vals = np.asarray(cell_areas, dtype=np.float32)
        total = float(np.sum(vals))
        probs = vals / max(total, 1e-9)
        active[i] = float(vals.size)
        peak[i] = float(np.max(vals) / max(total, 1e-9))
        entropy[i] = float(-(probs * np.log(probs + 1e-9)).sum() / math.log(cell_count))
        clipped_area[i] = total * cell_area
    return active, entropy, peak, clipped_area, cell_matrix


def _spatial_gradient_turbulence(cell_occ: np.ndarray, grid_cols: int, grid_rows: int) -> float:
    if cell_occ.size == 0:
        return 0.0
    grid = cell_occ.reshape(grid_rows, grid_cols)
    diffs: list[float] = []
    for gy in range(grid_rows):
        for gx in range(grid_cols):
            neighbors = []
            if gy > 0:
                neighbors.append(grid[gy - 1, gx])
            if gy + 1 < grid_rows:
                neighbors.append(grid[gy + 1, gx])
            if gx > 0:
                neighbors.append(grid[gy, gx - 1])
            if gx + 1 < grid_cols:
                neighbors.append(grid[gy, gx + 1])
            if neighbors:
                diffs.append(abs(float(grid[gy, gx]) - float(np.mean(neighbors))))
    return float(np.mean(diffs)) if diffs else 0.0


def high_fidelity_grid_stats(
    obb: ObbData,
    width: int,
    height: int,
    grid_cols: int,
    grid_rows: int,
    chunk_size: int = 200_000,
) -> dict[str, np.ndarray]:
    n = obb.frame.shape[0]
    hbb_active = np.zeros(n, dtype=np.float32)
    hbb_entropy = np.zeros(n, dtype=np.float32)
    hbb_peak = np.zeros(n, dtype=np.float32)
    obb_active = np.zeros(n, dtype=np.float32)
    obb_entropy = np.zeros(n, dtype=np.float32)
    obb_peak = np.zeros(n, dtype=np.float32)
    hbb_area_px = np.zeros(n, dtype=np.float32)
    obb_area_px = np.zeros(n, dtype=np.float32)
    hbb_cell_frac = np.zeros((n, grid_cols * grid_rows), dtype=np.float32)
    obb_cell_frac = np.zeros((n, grid_cols * grid_rows), dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        sl = slice(start, end)
        hbb_polys: list[list[tuple[float, float]]] = []
        obb_polys: list[list[tuple[float, float]]] = []
        for i in range(start, end):
            x = float(obb.x[i])
            y = float(obb.y[i])
            w = float(obb.w[i])
            h = float(obb.h[i])
            hbb_polys.append([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])

            cx = x + w / 2.0
            cy = y + h / 2.0
            long_side = max(w, h)
            short_side = min(w, h)
            dx = long_side / 2.0
            dy = short_side / 2.0
            theta = float(obb.theta[i])
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            local = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
            obb_polys.append([
                (cx + cos_t * lx - sin_t * ly, cy + sin_t * lx + cos_t * ly)
                for lx, ly in local
            ])

        hbb_a, hbb_e, hbb_p, hbb_area, hbb_cells = _rect_grid_stats(
            hbb_polys, width, height, grid_cols, grid_rows, keep_cells=True
        )
        obb_a, obb_e, obb_p, obb_area, obb_cells = _rect_grid_stats(
            obb_polys, width, height, grid_cols, grid_rows, keep_cells=True
        )
        hbb_active[sl], hbb_entropy[sl], hbb_peak[sl], hbb_area_px[sl] = hbb_a, hbb_e, hbb_p, hbb_area
        obb_active[sl], obb_entropy[sl], obb_peak[sl], obb_area_px[sl] = obb_a, obb_e, obb_p, obb_area
        if obb_cells is not None:
            obb_cell_frac[sl] = obb_cells
        if hbb_cells is not None:
            hbb_cell_frac[sl] = hbb_cells

    return {
        "hbb_grid_active": hbb_active,
        "hbb_grid_entropy": hbb_entropy,
        "hbb_grid_peak_frac": hbb_peak,
        "hbb_grid_area_px": hbb_area_px,
        "obb_grid_active": obb_active,
        "obb_grid_entropy": obb_entropy,
        "obb_grid_peak_frac": obb_peak,
        "obb_grid_area_px": obb_area_px,
        "hbb_grid_cell_frac": hbb_cell_frac,
        "obb_grid_cell_frac": obb_cell_frac,
    }


def grid_area_heatmap(
    obb: ObbData,
    width: int,
    height: int,
    grid_cols: int,
    grid_rows: int,
    max_rows: int = 120_000,
) -> tuple[np.ndarray, np.ndarray]:
    limit = min(max_rows, int(obb.frame.shape[0]))
    cell_w = width / max(1, grid_cols)
    cell_h = height / max(1, grid_rows)
    hbb_map = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    obb_map = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    for i in range(limit):
        x = float(obb.x[i])
        y = float(obb.y[i])
        w = float(obb.w[i])
        h = float(obb.h[i])
        hbb_poly = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        cx = x + w / 2.0
        cy = y + h / 2.0
        dx = max(w, h) / 2.0
        dy = min(w, h) / 2.0
        theta = float(obb.theta[i])
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        obb_poly = [
            (cx + cos_t * lx - sin_t * ly, cy + sin_t * lx + cos_t * ly)
            for lx, ly in [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
        ]
        for target, poly in [(hbb_map, hbb_poly), (obb_map, obb_poly)]:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            col0 = max(0, int(math.floor(min(xs) / max(cell_w, 1e-6))))
            col1 = min(grid_cols - 1, int(math.floor(max(xs) / max(cell_w, 1e-6))))
            row0 = max(0, int(math.floor(min(ys) / max(cell_h, 1e-6))))
            row1 = min(grid_rows - 1, int(math.floor(max(ys) / max(cell_h, 1e-6))))
            for gy in range(row0, row1 + 1):
                top = gy * cell_h
                bottom = min(height, (gy + 1) * cell_h)
                for gx in range(col0, col1 + 1):
                    left = gx * cell_w
                    right = min(width, (gx + 1) * cell_w)
                    target[gy, gx] += _polygon_area(_clip_polygon_to_rect(poly, left, top, right, bottom))
    return hbb_map, obb_map


def extract_window_features(
    dataset_key: str,
    dataset_name: str,
    frenet: FrenetData,
    obb: ObbData,
    video_path: Path,
    segment_length_m: float,
    speed_limit_kmh: float,
    window_s: float,
    step_s: float,
    safe_headway_s: float = 2.0,
    mgti_sensitivity: float = 1.0,
    grid_cols: int = 12,
    grid_rows: int = 4,
) -> tuple[list[str], list[dict[str, object]], np.ndarray]:
    video_width, video_height, _, _ = video_shape(video_path)
    coord_width = int(math.ceil(max(float(video_width), float(np.max(obb.x + obb.w)))))
    coord_height = int(math.ceil(max(float(video_height), float(np.max(obb.y + obb.h)))))
    image_area = float(coord_width * coord_height)
    size_map = vehicle_size_map(frenet)
    lane_events = lane_change_times(frenet)
    grid_stats = high_fidelity_grid_stats(obb, coord_width, coord_height, grid_cols, grid_rows)

    min_t = max(float(np.min(frenet.time_s)), float(np.min(obb.time_s)))
    max_t = min(float(np.max(frenet.time_s)), float(np.max(obb.time_s)))
    starts = np.arange(math.ceil(min_t / step_s) * step_s, max_t - window_s, step_s, dtype=np.float32)

    fr_order, fr_slices = _window_slices(frenet.time_s, starts, window_s)
    ob_order, ob_slices = _window_slices(obb.time_s, starts, window_s)
    sorted_lane_events = np.sort(lane_events)

    rows: list[dict[str, object]] = []
    n_cells = grid_rows * grid_cols
    grid_tensor_list: list[np.ndarray] = []
    for win_id, start in enumerate(starts):
        fr_left, fr_right = fr_slices[win_id]
        ob_left, ob_right = ob_slices[win_id]
        fr_idx = fr_order[fr_left:fr_right]
        ob_idx = ob_order[ob_left:ob_right]

        speed = frenet.speed_kmh[fr_idx]
        acc = frenet.acceleration[fr_idx]
        headway = _headway_metrics(frenet, fr_idx, default_headway_s=safe_headway_s * 2.0)
        unique_vehicles = np.unique(frenet.vehicle_id[fr_idx]) if fr_idx.size else np.asarray([], dtype=np.int32)
        unique_frames = np.unique(obb.frame[ob_idx]) if ob_idx.size else np.asarray([], dtype=np.int32)

        hbb_area = obb.w[ob_idx] * obb.h[ob_idx]
        obb_area = obb.obb_area[ob_idx]

        frame_count = max(1, int(unique_frames.size))
        hbb_occ = float(np.sum(hbb_area) / (frame_count * image_area)) if hbb_area.size else 0.0
        obb_occ = float(np.sum(obb_area) / (frame_count * image_area)) if obb_area.size else 0.0
        occ_reduction = (hbb_occ - obb_occ) / hbb_occ if hbb_occ > 1e-9 else 0.0
        hbb_hfgo_occ = float(np.sum(grid_stats["hbb_grid_area_px"][ob_idx]) / (frame_count * image_area)) if ob_idx.size else 0.0
        hfgo_occ = float(np.sum(grid_stats["obb_grid_area_px"][ob_idx]) / (frame_count * image_area)) if ob_idx.size else 0.0
        hfgo_reduction = (hbb_hfgo_occ - hfgo_occ) / hbb_hfgo_occ if hbb_hfgo_occ > 1e-9 else 0.0

        ev_left = int(np.searchsorted(sorted_lane_events, start, side="left"))
        ev_right = int(np.searchsorted(sorted_lane_events, start + window_s, side="left"))
        lane_changes = ev_right - ev_left

        direction_std = _circular_std(obb.theta[ob_idx])
        acceleration_interference = _safe_std(acc)
        compression = max(0.0, safe_headway_s - headway["mean_headway_s"]) / max(mgti_sensitivity, 1e-6)
        mgti = float(acceleration_interference * math.exp(min(6.0, compression)))
        hbb_active_mean = _safe_mean(grid_stats["hbb_grid_active"][ob_idx])
        obb_active_mean = _safe_mean(grid_stats["obb_grid_active"][ob_idx])
        hbb_entropy_mean = _safe_mean(grid_stats["hbb_grid_entropy"][ob_idx])
        obb_entropy_mean = _safe_mean(grid_stats["obb_grid_entropy"][ob_idx])
        hbb_peak_mean = _safe_mean(grid_stats["hbb_grid_peak_frac"][ob_idx])
        obb_peak_mean = _safe_mean(grid_stats["obb_grid_peak_frac"][ob_idx])
        if ob_idx.size:
            hbb_cell_occ = np.sum(grid_stats["hbb_grid_cell_frac"][ob_idx], axis=0) / max(1, frame_count)
            cell_occ = np.sum(grid_stats["obb_grid_cell_frac"][ob_idx], axis=0) / max(1, frame_count)
            sgt_hfgo = _spatial_gradient_turbulence(cell_occ, grid_cols, grid_rows)
            local_abs_diff = np.abs(hbb_cell_occ - cell_occ)
            active_cells = np.maximum(hbb_cell_occ, cell_occ) > 1e-9
            if np.any(active_cells):
                local_rel_diff = local_abs_diff[active_cells] / np.maximum(hbb_cell_occ[active_cells], 1e-6)
                hfgo_lgar_005 = float(np.mean(local_rel_diff > 0.05))
            else:
                hfgo_lgar_005 = 0.0
            hfgo_local_diff_mean = float(np.mean(local_abs_diff))
            hfgo_local_diff_max = float(np.max(local_abs_diff))

            cell_frac = grid_stats["obb_grid_cell_frac"][ob_idx]
            obb_areas = obb.obb_area[ob_idx].astype(np.float32)
            weights = cell_frac * obb_areas[:, None]
            cell_weight_total = np.sum(weights, axis=0)
            sin_t = np.sin(obb.theta[ob_idx]).astype(np.float32)
            cos_t = np.cos(obb.theta[ob_idx]).astype(np.float32)
            denom = np.maximum(cell_weight_total, 1e-9)
            sin_cell = np.where(cell_weight_total > 1e-9, np.sum(weights * sin_t[:, None], axis=0) / denom, 0.0)
            cos_cell = np.where(cell_weight_total > 1e-9, np.sum(weights * cos_t[:, None], axis=0) / denom, 0.0)
        else:
            sgt_hfgo = 0.0
            hfgo_lgar_005 = 0.0
            hfgo_local_diff_mean = 0.0
            hfgo_local_diff_max = 0.0
            hbb_cell_occ = np.zeros(n_cells, dtype=np.float32)
            cell_occ = np.zeros(n_cells, dtype=np.float32)
            sin_cell = np.zeros(n_cells, dtype=np.float32)
            cos_cell = np.zeros(n_cells, dtype=np.float32)
        window_tensor = np.stack([
            cell_occ.reshape(grid_rows, grid_cols),
            hbb_cell_occ.reshape(grid_rows, grid_cols),
            sin_cell.reshape(grid_rows, grid_cols),
            cos_cell.reshape(grid_rows, grid_cols),
        ], axis=0).astype(np.float32)
        grid_tensor_list.append(window_tensor)
        row = {
            "dataset": dataset_key,
            "dataset_name": dataset_name,
            "window_id": win_id,
            "start_s": f"{float(start):.3f}",
            "end_s": f"{float(start + window_s):.3f}",
            "sample_rows": int(fr_idx.size),
            "frame_count": int(frame_count),
            "vehicle_count": int(unique_vehicles.size),
            "density_veh_per_m": f"{float(unique_vehicles.size / max(segment_length_m, 1e-6)):.6f}",
            "flow_veh_per_s": f"{float(unique_vehicles.size / max(window_s, 1e-6)):.6f}",
            "mean_speed_kmh": f"{_safe_mean(speed):.6f}",
            "std_speed_kmh": f"{_safe_std(speed):.6f}",
            "speed_ratio": f"{_safe_mean(speed) / max(speed_limit_kmh, 1e-6):.6f}",
            "mean_abs_acc": f"{_safe_mean(np.abs(acc)):.6f}",
            "std_acc": f"{_safe_std(acc):.6f}",
            "mean_headway_s": f"{headway['mean_headway_s']:.6f}",
            "min_headway_s": f"{headway['min_headway_s']:.6f}",
            "mean_space_gap_m": f"{headway['mean_space_gap_m']:.6f}",
            "headway_sample_count": int(headway["headway_sample_count"]),
            "acceleration_interference": f"{acceleration_interference:.6f}",
            "mgti": f"{mgti:.6f}",
            "lane_change_count": int(lane_changes),
            "lane_change_rate": f"{float(lane_changes / max(1, unique_vehicles.size)):.6f}",
            "hbb_occupancy": f"{hbb_occ:.8f}",
            "obb_occupancy": f"{obb_occ:.8f}",
            "occupancy_reduction": f"{occ_reduction:.8f}",
            "hbb_hfgo_occupancy": f"{hbb_hfgo_occ:.8f}",
            "hfgo_occupancy": f"{hfgo_occ:.8f}",
            "hfgo_occupancy_reduction": f"{hfgo_reduction:.8f}",
            "hbb_grid_active_mean": f"{hbb_active_mean:.6f}",
            "obb_grid_active_mean": f"{obb_active_mean:.6f}",
            "grid_active_reduction": f"{((hbb_active_mean - obb_active_mean) / hbb_active_mean if hbb_active_mean > 1e-9 else 0.0):.6f}",
            "hbb_grid_entropy_mean": f"{hbb_entropy_mean:.6f}",
            "obb_grid_entropy_mean": f"{obb_entropy_mean:.6f}",
            "grid_entropy_reduction": f"{((hbb_entropy_mean - obb_entropy_mean) / hbb_entropy_mean if hbb_entropy_mean > 1e-9 else 0.0):.6f}",
            "hbb_grid_peak_frac_mean": f"{hbb_peak_mean:.6f}",
            "obb_grid_peak_frac_mean": f"{obb_peak_mean:.6f}",
            "grid_peak_gain": f"{(obb_peak_mean - hbb_peak_mean):.6f}",
            "sgt_hfgo": f"{sgt_hfgo:.8f}",
            "hfgo_lgar_005": f"{hfgo_lgar_005:.8f}",
            "hfgo_local_diff_mean": f"{hfgo_local_diff_mean:.8f}",
            "hfgo_local_diff_max": f"{hfgo_local_diff_max:.8f}",
            "direction_fluctuation": f"{direction_std:.8f}",
            "theta_conf_mean": f"{_safe_mean(obb.theta_conf[ob_idx], 0.0):.6f}",
        }
        rows.append(row)

    fieldnames = [
        "dataset",
        "dataset_name",
        "window_id",
        "start_s",
        "end_s",
        "sample_rows",
        "frame_count",
        "vehicle_count",
        "density_veh_per_m",
        "flow_veh_per_s",
        "mean_speed_kmh",
        "std_speed_kmh",
        "speed_ratio",
        "mean_abs_acc",
        "std_acc",
        "mean_headway_s",
        "min_headway_s",
        "mean_space_gap_m",
        "headway_sample_count",
        "acceleration_interference",
        "mgti",
        "lane_change_count",
        "lane_change_rate",
        "hbb_occupancy",
        "obb_occupancy",
        "occupancy_reduction",
        "hbb_hfgo_occupancy",
        "hfgo_occupancy",
        "hfgo_occupancy_reduction",
        "hbb_grid_active_mean",
        "obb_grid_active_mean",
        "grid_active_reduction",
        "hbb_grid_entropy_mean",
        "obb_grid_entropy_mean",
        "grid_entropy_reduction",
        "hbb_grid_peak_frac_mean",
        "obb_grid_peak_frac_mean",
        "grid_peak_gain",
        "sgt_hfgo",
        "hfgo_lgar_005",
        "hfgo_local_diff_mean",
        "hfgo_local_diff_max",
        "direction_fluctuation",
        "theta_conf_mean",
    ]
    grid_tensors = (
        np.stack(grid_tensor_list, axis=0)
        if grid_tensor_list
        else np.zeros((0, 4, grid_rows, grid_cols), dtype=np.float32)
    )
    return fieldnames, rows, grid_tensors
