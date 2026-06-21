#!/usr/bin/env python
"""Build a Table-5-style scene figure for the paper.

For several test scenes it shows two drone frames a few seconds apart with the
*same vehicle* boxed in both and connected by an arrow (illustrating vehicle
motion / tracking), alongside the ground-truth traffic state and the GTSEP-DL
predicted state. Frames, boxes and states are all taken from the real pipeline
(XAM-N-6 video + pixel.csv tracks + experiment_results.json predictions).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CJK_FONT = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
if Path(CJK_FONT).exists():
    font_manager.fontManager.addfont(CJK_FONT)
    _droid = font_manager.FontProperties(fname=CJK_FONT).get_name()
    # Latin/digits from DejaVu Sans, CJK glyphs fall back to Droid (per-glyph).
    plt.rcParams["font.family"] = ["DejaVu Sans", _droid]
plt.rcParams["axes.unicode_minus"] = False

FPS = 60.0
FRAME_OFFSET = 3901
SEGMENT_LENGTH_M = 140.0  # XAM-N-6 analysed road-segment length (configs/datasets.json)
STATE_NAMES = ["畅通", "缓行", "拥挤", "堵塞"]
GAP = 14  # pixel gap between the two stacked frames


def load_assets():
    exp = json.loads((PROJECT_ROOT / "outputs/reports/experiment_results.json").read_text())
    pred = exp["prediction"]
    horizon = int(pred["horizon_steps"])
    tp = np.asarray(pred["test_positions"])
    true = np.asarray(pred["GTSEP-DL"]["true"])
    pr = np.asarray(pred["GTSEP-DL"]["pred"])
    win = pd.read_csv(PROJECT_ROOT / "outputs/features/xamn6_windows.csv")
    px = pd.read_csv(PROJECT_ROOT / "data_check/XAM-N/XAM-N-6/pixel.csv")
    return horizon, tp, true, pr, win, px


def coord_bounds(px: pd.DataFrame):
    return (
        float((px.x + px.w).max()),   # coord_x2
        float(px.y.min()),            # coord_y1
        float((px.y + px.h).max()),   # coord_y2
    )


def grab_frame(cap, t_orig: float):
    video_idx = int(round(t_orig * FPS)) - FRAME_OFFSET
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_idx = min(max(0, video_idx), n - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_idx)
    ok, img = cap.read()
    return img if ok else None


def pixel_frame_for(t_orig: float) -> int:
    return int(round(t_orig * FPS))


def box_in_video(row, sx, sy, cy1, img_w, img_h):
    x = row.x * sx
    y = (row.y - cy1) * sy
    w = row.w * sx
    h = row.h * sy
    x2, y2 = x + w, y + h
    x = max(0, min(x, img_w - 1)); x2 = max(0, min(x2, img_w - 1))
    y = max(0, min(y, img_h - 1)); y2 = max(0, min(y2, img_h - 1))
    return int(round(x)), int(round(y)), int(round(x2)), int(round(y2))


def pick_representative_vehicle(px, f1, f2, mean_speed_kmh, m_per_unit, dt_s):
    """Pick the vehicle present in both frames whose own 5 s travel speed is
    closest to the scene mean speed — i.e. a vehicle that *represents* the
    scene's traffic state, rather than the fastest/slowest outlier.

    Returns (vid, row1, row2, disp_m, speed_kmh) or None.
    """
    r1 = px[px.frame == f1].set_index("vehicleID")
    r2 = px[px.frame == f2].set_index("vehicleID")
    common = r1.index.intersection(r2.index)
    if len(common) == 0:
        return None
    best, best_gap = None, float("inf")
    for vid in common:
        a, b = r1.loc[vid], r2.loc[vid]
        if getattr(a, "ndim", 1) > 1:
            a = a.iloc[0]
        if getattr(b, "ndim", 1) > 1:
            b = b.iloc[0]
        # Skip tiny boxes (noisy tracks).
        if float(a.w) * float(a.h) < 800:
            continue
        cxa, cya = float(a.x) + float(a.w) / 2, float(a.y) + float(a.h) / 2
        cxb, cyb = float(b.x) + float(b.w) / 2, float(b.y) + float(b.h) / 2
        disp_units = float(np.hypot(cxb - cxa, cyb - cya))
        disp_m = disp_units * m_per_unit
        speed_kmh = disp_m / max(dt_s, 1e-6) * 3.6
        gap = abs(speed_kmh - mean_speed_kmh)
        if gap < best_gap:
            best_gap = gap
            best = (vid, a, b, round(disp_m, 1), round(speed_kmh, 1))
    return best


def render_scene(cap, px, t1, t2, scales, img_dims, mean_speed_kmh, m_per_unit):
    sx, sy, cy1 = scales
    img_w, img_h = img_dims
    f1, f2 = pixel_frame_for(t1), pixel_frame_for(t2)
    veh = pick_representative_vehicle(px, f1, f2, mean_speed_kmh, m_per_unit, t2 - t1)
    im1, im2 = grab_frame(cap, t1), grab_frame(cap, t2)
    if im1 is None or im2 is None:
        return None
    centers = []
    for im, row in ((im1, None if veh is None else veh[1]), (im2, None if veh is None else veh[2])):
        if row is not None:
            x1, y1, x2, y2 = box_in_video(row, sx, sy, cy1, img_w, img_h)
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 3)
            centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
        else:
            centers.append(None)
    # Stack vertically with a white gap.
    canvas = np.full((img_h * 2 + GAP, img_w, 3), 255, np.uint8)
    canvas[:img_h] = im1
    canvas[img_h + GAP:] = im2
    # Arrow connecting the same vehicle from top frame to bottom frame.
    if veh is not None and centers[0] and centers[1]:
        p1 = (centers[0][0], centers[0][1])
        p2 = (centers[1][0], img_h + GAP + centers[1][1])
        cv2.arrowedLine(canvas, p1, p2, (0, 165, 255), 3, tipLength=0.03)
    img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    if veh is None:
        return img, None, None, None
    return img, int(veh[0]), veh[3], veh[4]  # img, vid, disp_m, speed_kmh


def select_scenes(tp, true, pr, win, horizon, n=5):
    """Pick up to n test windows that are varied and well separated in time.

    Strategy: one correct case for each available true state (free->slow->crowded),
    then one honest boundary case where GTSEP-DL disagrees with the truth (which
    also surfaces the 堵塞 label that is absent from the test ground truth),
    then fill any remaining slots with additional time-separated correct cases.
    """
    def qstart(k):
        return float(win.iloc[tp[k] + horizon].start_s)

    chosen = []

    def far_enough(k, gap=8.0):
        return all(abs(qstart(k) - qstart(c)) > gap for c in chosen)

    for s in range(4):  # correct case per available true state
        cands = [k for k in range(len(tp)) if true[k] == s and pr[k] == s and tp[k] + horizon < len(win)]
        if cands and far_enough(cands[len(cands) // 2]):
            chosen.append(cands[len(cands) // 2])

    # one boundary/mismatch case (prefer crowded<->jam confusion to show 堵塞)
    wrong = [k for k in range(len(tp)) if true[k] != pr[k] and tp[k] + horizon < len(win)]
    wrong.sort(key=lambda k: (0 if 3 in (true[k], pr[k]) else 1))
    for k in wrong:
        if len(chosen) >= n:
            break
        if far_enough(k):
            chosen.append(k)
            break

    extra = [k for k in range(len(tp)) if pr[k] == true[k] and tp[k] + horizon < len(win) and k not in chosen]
    for k in sorted(extra, key=qstart):
        if len(chosen) >= n:
            break
        if far_enough(k):
            chosen.append(k)

    return sorted(chosen, key=qstart)[:n]


def main():
    horizon, tp, true, pr, win, px = load_assets()
    cx2, cy1, cy2 = coord_bounds(px)
    x_min = float(px.x.min())
    # Coord-unit -> metre scale: the analysed road segment (140 m) spans the
    # along-road (x) extent of all tracked vehicles.
    m_per_unit = SEGMENT_LENGTH_M / max(cx2 - x_min, 1.0)
    cap = cv2.VideoCapture(str(PROJECT_ROOT / "data_check/XAM-N/XAM-N-6/sample video.mp4"))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx = img_w / max(cx2, 1.0)
    sy = img_h / max(cy2 - cy1, 1.0)

    scenes = select_scenes(tp, true, pr, win, horizon)
    rows = []
    out_dir = PROJECT_ROOT / "outputs/figures/scenes"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    for i, k in enumerate(scenes, start=1):
        q = tp[k] + horizon
        wrow = win.iloc[q]
        t1 = float(wrow.start_s)
        t2 = float(wrow.end_s)
        out = render_scene(cap, px, t1, t2, (sx, sy, cy1), (img_w, img_h),
                           float(wrow.mean_speed_kmh), m_per_unit)
        if out is None:
            continue
        img, vid, disp_m, veh_speed = out
        true_s = STATE_NAMES[int(true[k])]
        pred_s = STATE_NAMES[int(pr[k])]
        correct = true_s == pred_s

        # One standalone figure per scene.
        fig = plt.figure(figsize=(6.6, 5.6))
        ax = fig.add_axes([0.02, 0.06, 0.96, 0.82]); ax.axis("off")
        ax.imshow(img)
        ax.text(0.008, 0.985, f"t₁ = {t1:.0f} s", transform=ax.transAxes, ha="left", va="top",
                fontsize=10, color="yellow", bbox=dict(facecolor="black", alpha=0.5, pad=2, edgecolor="none"))
        ax.text(0.008, 0.49, f"t₂ = {t2:.0f} s", transform=ax.transAxes, ha="left", va="top",
                fontsize=10, color="yellow", bbox=dict(facecolor="black", alpha=0.5, pad=2, edgecolor="none"))
        if disp_m is not None:
            ax.text(0.5, -0.015,
                    f"代表车辆 ID={vid}：5 s 内沿道路前进约 {disp_m:.0f} m（≈ {veh_speed:.0f} km·h⁻¹），"
                    f"与场景平均车速 {wrow.mean_speed_kmh:.0f} km·h⁻¹ 相符",
                    transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#333")
        mark = "√" if correct else "×"
        color = "#1a7f37" if correct else "#c0392b"
        title = (f"场景 {i}（箭头连接的矩形框内为同一车辆 ID={vid}）\n"
                 f"状态真值：{true_s}    GTSEP-DL 预测：{pred_s} {mark}")
        fig.suptitle(title, fontsize=12, y=0.985, color=color if not correct else "black")
        fp = out_dir / f"scene_{i}.png"
        fig.savefig(fp, dpi=200, bbox_inches="tight")
        plt.close(fig)

        meta.append({
            "scene": i, "t1_s": round(t1, 1), "t2_s": round(t2, 1), "vehicle_id": vid,
            "veh_displacement_m": disp_m, "veh_speed_kmh": veh_speed,
            "mean_speed_kmh": round(float(wrow.mean_speed_kmh), 2),
            "density_veh_per_m": round(float(wrow.density_veh_per_m), 3),
            "vehicle_count": int(wrow.vehicle_count),
            "flow_veh_per_s": round(float(wrow.flow_veh_per_s), 3),
            "true": true_s, "pred": pred_s, "correct": correct,
            "file": str(fp.relative_to(PROJECT_ROOT)),
        })
        print(f"[FIG] scene {i}: {fp.name}  V={wrow.mean_speed_kmh:.1f}km/h "
              f"D={wrow.density_veh_per_m:.3f}veh/m N={int(wrow.vehicle_count)} "
              f"vehID={vid} disp={disp_m}m vspd={veh_speed}km/h "
              f"true={true_s} pred={pred_s} {'OK' if correct else 'WRONG'}")
    cap.release()

    meta_path = PROJECT_ROOT / "outputs/reports/scene_cases.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[FIG] wrote {len(meta)} scene images to {out_dir} and metadata to {meta_path}")


if __name__ == "__main__":
    main()
