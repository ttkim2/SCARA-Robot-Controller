"""
Drawing path preview — no robot needed.
Runs the exact same extract_segments() / densify() logic as main.py and
renders the result to assets/outputs/path_preview.png so you can verify
contours before running the robot.

Usage:
    python preview_drawing.py
"""

import sys
import os
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from trajectory import densify
from image_select import select_image


# ---------------------------------------------------------------------------
# Visvalingam-Whyatt simplification (mirrors main.py exactly)
# ---------------------------------------------------------------------------
def _visvalingam_whyatt(pts_in, threshold, closed=True):
    import heapq
    pts = list(pts_in)
    n   = len(pts)
    if n < 3:
        return pts

    def _area(i, p, nx):
        p0, p1, p2 = pts[p], pts[i], pts[nx]
        return 0.5 * abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1]))

    prev_idx = list(range(-1, n - 1))
    next_idx = list(range(1, n + 1))
    if closed:
        prev_idx[0]   = n - 1
        next_idx[n-1] = 0
    else:
        prev_idx[0]   = 0
        next_idx[n-1] = n - 1

    areas     = [float('inf')] * n
    heap      = []
    start_i, end_i = (0, n) if closed else (1, n - 1)
    for i in range(start_i, end_i):
        areas[i] = _area(i, prev_idx[i], next_idx[i])
        heapq.heappush(heap, (areas[i], i))

    removed   = [False] * n
    remaining = n
    max_seen  = 0.0

    while heap and remaining > 3:
        a, i = heapq.heappop(heap)
        if removed[i] or a < areas[i]:
            continue
        eff = max(a, max_seen)
        if eff > threshold:
            break
        max_seen   = max(max_seen, eff)
        removed[i] = True
        remaining -= 1

        p  = prev_idx[i]
        nx = next_idx[i]
        next_idx[p]  = nx
        prev_idx[nx] = p

        for j in (p, nx):
            if not closed and (j == 0 or j == n - 1):
                continue
            new_a    = max(_area(j, prev_idx[j], next_idx[j]), max_seen)
            areas[j] = new_a
            heapq.heappush(heap, (new_a, j))

    return [pts[i] for i in range(n) if not removed[i]]


# ---------------------------------------------------------------------------
# CONFIG — mirrors main.py
# ---------------------------------------------------------------------------
IMAGE_PATH   = select_image()
CANVAS_MM    = 160.0
DENSITY_PX   = 4
MAX_SEG_MM   = 3.0
LINE_THRESH  = 128   # 128 = black-on-white. 110 = UIUC navy logo. 0 = auto-Otsu.

# ---------------------------------------------------------------------------
# Exact copy of extract_segments from main.py
# ---------------------------------------------------------------------------
def extract_segments(image_path, workspace_mm, density_px, line_thresh=0):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] Cannot open image: {os.path.abspath(image_path)}")
        return []
    PAD = 6
    img = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD,
                             cv2.BORDER_CONSTANT, value=255)
    h, w = img.shape
    orig_w, orig_h = w - 2 * PAD, h - 2 * PAD
    scale_x = workspace_mm / orig_w
    scale_y = workspace_mm / orig_h

    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    if line_thresh == 0:
        _, binary = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(blurred, line_thresh, 255, cv2.THRESH_BINARY_INV)

    # Morphological closing: merges thin JPEG-compression ringing bands
    # (the dark halos around high-contrast edges) into the main ink region.
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    min_area = (orig_w * orig_h) * 0.0003

    segments = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x_pts = cnt[:, 0, 0]
        y_pts = cnt[:, 0, 1]
        if x_pts.min() <= 2 or y_pts.min() <= 2 or x_pts.max() >= w - 2 or y_pts.max() >= h - 2:
            continue
        step    = max(1, min(density_px, len(cnt) // 12))
        sampled = cnt[::step] if len(cnt) > step else cnt
        sampled = np.array(sampled, dtype=np.float32)
        if len(sampled) < 3:
            continue
        arc     = cv2.arcLength(sampled, closed=True)
        epsilon = 0.0008 * arc if arc < 200 else 0.0015 * arc
        # OLD — Douglas-Peucker (uncomment to revert):
        # approx = cv2.approxPolyDP(sampled, epsilon, closed=True)
        # if len(approx) < 3:
        #     continue
        # pts = [((float(p[0][0]) - PAD) * scale_x,
        #         (float(p[0][1]) - PAD) * scale_y) for p in approx]
        # NEW — Visvalingam-Whyatt
        vw_thresh = 0.5 * epsilon * epsilon
        pts_raw   = [(float(p[0][0]), float(p[0][1])) for p in sampled]
        pts_raw   = _visvalingam_whyatt(pts_raw, vw_thresh, closed=True)
        if len(pts_raw) < 3:
            continue
        pts = [((p[0] - PAD) * scale_x, (p[1] - PAD) * scale_y) for p in pts_raw]
        segments.append(pts)

    return segments

# ---------------------------------------------------------------------------
# Extract + densify
# ---------------------------------------------------------------------------
segments = extract_segments(IMAGE_PATH, CANVAS_MM, DENSITY_PX, LINE_THRESH)
if not segments:
    print("[ERROR] No contours found.")
    sys.exit(1)

dense_segments = [densify(s, MAX_SEG_MM) for s in segments]
total_pts = sum(len(d) for d in dense_segments)
print(f"[OK] {len(segments)} stroke(s), {total_pts} total waypoints after densification")

# ---------------------------------------------------------------------------
# Render preview — canvas coordinates (image-mm space)
# ---------------------------------------------------------------------------
SCALE = 6
PAD   = 30
W     = int(CANVAS_MM * SCALE + PAD * 2)
H     = int(CANVAS_MM * SCALE + PAD * 2)
canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

colors = [
    (220,  60,  60),
    ( 60, 160,  60),
    ( 60,  60, 220),
    (180, 120,  40),
    (120,  40, 180),
    ( 40, 180, 180),
]

for seg_idx, dense in enumerate(dense_segments):
    col = colors[seg_idx % len(colors)]
    for i, (px_mm, py_mm) in enumerate(dense):
        px = int(PAD + px_mm * SCALE)
        py = int(PAD + py_mm * SCALE)
        cv2.circle(canvas, (px, py), 1, col, -1)
        if i > 0:
            prev = dense[i - 1]
            cv2.line(canvas,
                     (int(PAD + prev[0]*SCALE), int(PAD + prev[1]*SCALE)),
                     (px, py), col, 1, cv2.LINE_AA)
    sx = int(PAD + dense[0][0] * SCALE)
    sy = int(PAD + dense[0][1] * SCALE)
    cv2.circle(canvas, (sx, sy), 4, (0, 0, 0), -1)
    cv2.putText(canvas, str(seg_idx + 1), (sx + 5, sy - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

cv2.rectangle(canvas, (PAD, PAD),
              (int(PAD + CANVAS_MM * SCALE), int(PAD + CANVAS_MM * SCALE)),
              (200, 200, 200), 1)

out_path = 'assets/outputs/path_preview.png'
cv2.imwrite(out_path, canvas)
print(f"[OK] Preview saved to {out_path}")

try:
    display = cv2.resize(canvas, (500, 500), interpolation=cv2.INTER_AREA)
    cv2.imshow("Drawing path preview (press any key to close)", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception:
    print("     (no display available — open the PNG file to view)")
