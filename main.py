"""
UIUC SCARA Robot — Drawing Controller
======================================
Homes the robot, plans a drawing path from any input image, shows a
path preview, then executes the drawing with pipelined IK and adaptive
per-waypoint speed control.

Pipeline
--------
  1. Home all axes
  2. Extract contours from image (threshold → morphological close → approxPolyDP)
  3. Densify contours to max MAX_SEG_MM spacing (trajectory.py)
  4. Smooth paths with Savitzky-Golay filter (control.py)
  5. Pre-solve inverse kinematics for every waypoint
  6. Sort strokes by nearest-neighbour to minimise pen-up travel
  7. Show scaled path preview, wait for user confirmation
  8. Move arm to drawing position, confirm iPad placement
  9. Draw: for each stroke — pen up → travel → pen down → pipeline waypoints

Key files
---------
  src/main.cpp       Firmware (AccelStepper, MultiStepper, blend queue)
  core/kinematics.py 2R planar IK/FK with DH matrices and FK verification
  trajectory.py      Linear densification of contour segments
  control.py         Savitzky-Golay smoothing and curvature-adaptive speeds
  preview_drawing.py Standalone path preview (no robot required)
"""

import serial
import time
import sys
import os
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SERIAL_PORT    = 'COM3'
BAUD_RATE      = 115200
L1, L2         = 220.0, 212.5
MAX_ROTATION   = 330.0
DRAW_SPEED     = 2000    # steps/s. Stay above ~300 to avoid elbow stepper resonance
                        # (resonance shows as zigzag on vertical/elbow-heavy lines).
                        # If waves return try 500; if stalls occur try 350.
MIN_DRAW_SPEED = 300    # steps/s floor on curves.  Must stay above the stepper
                        # resonance band (~300).  Raise to 400 if stalls occur in
                        # tight corners; lower toward 300 for maximum corner control.
CURVE_SENSITIVITY = 120  # How aggressively curvature slows the pen.
                        # speed = DRAW_SPEED / (1 + CURVE_SENSITIVITY × κ_mm⁻¹)
                        #   0   → disable (constant speed)
                        #  30   → gentle   (designed for DRAW_SPEED ≈ 400)
                        #  60   → moderate (good default for DRAW_SPEED = 800)
                        # 120   → aggressive (useful for tight logos / small text)
                        # Example at DRAW_SPEED=2000, CURVE_SENSITIVITY=120:
                        #   Straight line  (κ ≈ 0)      → 2000 steps/s
                        #   Circle R=80mm  (κ = 0.0125) → 2000/(1+1.5) ≈ 800 steps/s
                        #   Tight corner R=10mm (κ=0.1) → 2000/(1+12.0)  ≈ 167 → clamped to MIN
TRAVEL_SPEED   = 2000   # steps/s for pen-up travel between strokes

# Drawing position — physical joint angles when end-effector is at drawing centre.
SH_DRAW        = 195.0   # physical shoulder angle at drawing centre
EL_DRAW        = 67.0    # physical elbow angle at drawing centre
DRAW_CENTER_MM = 250.0   # distance from shoulder to drawing centre (mm)

# Canvas rotation correction.
# If the drawing looks tilted clockwise on the paper, increase this value.
# If tilted counter-clockwise, decrease it (use a negative number).
CANVAS_ROTATION_DEG = 0.0

# Y-axis scale correction.
# If the drawing looks squished top-to-bottom (y-direction), increase above 1.0.
# If stretched top-to-bottom, decrease below 1.0.
# The centre of the canvas is always fixed; only the extent changes.
# Example: CANVAS_SCALE_Y = 1.10 adds 10 % stretch in the depth direction.
CANVAS_SCALE_Y = 1.3

# Image / canvas parameters
CANVAS_MM      = 130.0   # square canvas size in mm (drawing fits inside this box)
DENSITY_PX     = 4       # contour sub-sample: lower = more waypoints, smoother curves
MAX_SEG_MM     = 3        # max Cartesian distance between IK waypoints (mm)

# Contrast threshold for isolating drawn lines from the background.
#   > 0  : explicit grayscale cutoff — pixels ≤ value are treated as ink.
#           128  → black lines on white paper / simple black-and-white drawings.
#           110  → dark navy outline of UIUC logo (ignores orange fill ≈ 130).
#   = 0  : auto-detect threshold via Otsu's method — works for any image with
#           a clear foreground/background contrast difference.
LINE_THRESH    = 128

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from image_select import select_image
IMAGE_PATH = select_image()

from core.kinematics import KinematicsEngine, BacklashCompensator

from trajectory import densify
from control import smooth_path, adaptive_speeds

# ---------------------------------------------------------------------------
# Serial helpers
# ---------------------------------------------------------------------------
def send(cmd: str):
    arduino.write((cmd.strip() + '\n').encode('utf-8'))

def wait_for_done(timeout_s: float = 30.0, label: str = "") -> bool:
    """Block until MOVE_DONE. Ignores QUEUED responses."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if arduino.in_waiting > 0:
            raw = arduino.readline().decode('utf-8', errors='ignore').strip()
            if raw:
                print(f"  [ARD] {raw}")
                if "MOVE_DONE" in raw:
                    return True
    print(f"  [WARN] Timeout {timeout_s}s waiting for {label}")
    return False

# Alias so non-drawing call sites don't need renaming.
wait_for_done_with_queued = wait_for_done

# ---------------------------------------------------------------------------
# Connect + compute offsets
# ---------------------------------------------------------------------------
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    kin     = KinematicsEngine(L1=L1, L2=L2, max_rotation=MAX_ROTATION)
    print(f"[OK] Connected on {SERIAL_PORT}")
except Exception as e:
    print(f"[ERROR] {e}")
    sys.exit(1)

# Offsets map IK angles → physical Arduino angles.
# Anchored so IK(DRAW_CENTER_MM, 0) maps exactly to (SH_DRAW, EL_DRAW).
_ik_c = kin.inverse_kinematics(DRAW_CENTER_MM, 0.0)
if _ik_c is None:
    print(f"[ERROR] IK cannot reach drawing centre at {DRAW_CENTER_MM} mm.")
    sys.exit(1)
sh_offset = SH_DRAW - _ik_c[0]
el_offset = EL_DRAW - _ik_c[1]
print(f"[CONFIG] IK centre: t1={_ik_c[0]:.2f}°  t2={_ik_c[1]:.2f}°")
print(f"[CONFIG] Offsets:   sh={sh_offset:.2f}°  el={el_offset:.2f}°")

time.sleep(2)

# ---------------------------------------------------------------------------
# 1. Home
# ---------------------------------------------------------------------------
print("\n[1] Homing...")
send('H')
while True:
    line = arduino.readline().decode('utf-8', errors='ignore').strip()
    if line:
        print(f"  [ARD] {line}")
    if "CALIBRATED" in line:
        print("[OK] Homed.")
        break

# The firmware fires MOVE_DONE immediately after CALIBRATED (all axes at rest).
# That stale MOVE_DONE would be consumed by the first wait_for_done() below,
# causing it to return before the shoulder has moved and cascading into the Z
# not reaching hover — which is what makes the first stroke's "pen up" time out.
time.sleep(0.3)            # let Arduino finish any residual serial output
arduino.reset_input_buffer()  # discard the stale MOVE_DONE

# ---------------------------------------------------------------------------
# 2. Path planning + IK pre-solve  (done while robot is idle at home)
# ---------------------------------------------------------------------------
print(f"\n[2] Planning path from {IMAGE_PATH} (canvas={CANVAS_MM}mm, density={DENSITY_PX}px)...")

# ---------------------------------------------------------------------------
# Visvalingam-Whyatt polyline simplification
# ---------------------------------------------------------------------------
def _visvalingam_whyatt(pts_in, threshold, closed=True):
    """
    Visvalingam-Whyatt polygon/polyline simplification.

    Iteratively removes the point whose removal creates the smallest effective
    triangle area (formed by the point and its two neighbours) until all
    remaining points have area >= threshold.

    Unlike Douglas-Peucker (which measures perpendicular distance to a chord),
    VW weights each point by its visual contribution to the overall shape.
    This preserves the character of organic shapes — faces, lettering, logos —
    better than DP while still removing noise and quantisation jitter.

    Parameters
    ----------
    pts_in    : list of (x, y) in pixel coordinates
    threshold : minimum triangle area to keep a point (pixels²).
                Set to 0.5 * epsilon² for equivalence with DP epsilon.
    closed    : True for closed polygons (first ↔ last point connected).

    Returns
    -------
    list of (x, y), minimum 3 points.
    """
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
        prev_idx[0]   = 0       # fixed endpoints for open polyline
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
        if removed[i] or a < areas[i]:   # stale heap entry
            continue
        eff = max(a, max_seen)   # enforce monotone increase (VW property)
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


def extract_segments(image_path, workspace_mm, density_px, line_thresh=0):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] Cannot open image: {os.path.abspath(image_path)}")
        print(f"        Place the file there and update IMAGE_PATH in the config.")
        return []

    # Add white padding so the outermost contour is never within 2 px of the
    # image edge.  Without this the border-filter strips the outer I on logos
    # that naturally fill the image all the way to the edge.
    PAD = 6
    img = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD,
                             cv2.BORDER_CONSTANT, value=255)
    h, w = img.shape
    orig_w, orig_h = w - 2 * PAD, h - 2 * PAD
    scale_x = workspace_mm / orig_w
    scale_y = workspace_mm / orig_h

    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    if line_thresh == 0:
        # Auto-detect the best threshold via Otsu's method.
        # Works for any image with a clear ink/background contrast.
        _, binary = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # Explicit threshold: pixels ≤ line_thresh are treated as ink.
        _, binary = cv2.threshold(blurred, line_thresh, 255, cv2.THRESH_BINARY_INV)

    # Morphological closing: merges thin JPEG-compression ringing bands
    # (the dark halos around high-contrast edges) into the main ink region.
    # A 3×3 kernel closes gaps up to ~1.5 px wide without blurring real edges.
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # RETR_CCOMP returns both the outer boundary and inner hole boundary of
    # every connected region — both are meaningful drawing paths.
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    min_area = (orig_w * orig_h) * 0.0003

    segments = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        # Skip contours that reach the padded border (image-frame artefacts).
        x_pts = cnt[:, 0, 0]
        y_pts = cnt[:, 0, 1]
        if x_pts.min() <= 2 or y_pts.min() <= 2 or x_pts.max() >= w - 2 or y_pts.max() >= h - 2:
            continue
        # Adaptive step: always keep at least 12 sample points so small
        # features (e.g. the ® R at ~30 contour pts) are not over-subsampled.
        step    = max(1, min(density_px, len(cnt) // 12))
        sampled = cnt[::step] if len(cnt) > step else cnt
        sampled = np.array(sampled, dtype=np.float32)
        if len(sampled) < 3:
            continue
        # Tighter epsilon for small contours so fine detail (® R) survives.
        arc     = cv2.arcLength(sampled, closed=True)
        epsilon = 0.0008 * arc if arc < 200 else 0.0015 * arc
        # OLD — Douglas-Peucker simplification (uncomment to revert to approxPolyDP):
        # approx = cv2.approxPolyDP(sampled, epsilon, closed=True)
        # if len(approx) < 3:
        #     continue
        # pts = [((float(p[0][0]) - PAD) * scale_x,
        #         (float(p[0][1]) - PAD) * scale_y) for p in approx]
        # NEW — Visvalingam-Whyatt: removes points by smallest effective triangle
        # area, preserving the visual character of organic shapes better than DP.
        vw_thresh = 0.5 * epsilon * epsilon  # area equivalent of DP epsilon
        pts_raw   = [(float(p[0][0]), float(p[0][1])) for p in sampled]
        pts_raw   = _visvalingam_whyatt(pts_raw, vw_thresh, closed=True)
        if len(pts_raw) < 3:
            continue
        pts = [((p[0] - PAD) * scale_x, (p[1] - PAD) * scale_y) for p in pts_raw]
        segments.append(pts)

    return segments


segments = extract_segments(IMAGE_PATH, CANVAS_MM, DENSITY_PX, LINE_THRESH)

if not segments:
    print("[ERROR] No contours found in image.")
    arduino.close()
    sys.exit(1)

half    = CANVAS_MM / 2.0
cos_r   = np.cos(np.deg2rad(CANVAS_ROTATION_DEG))
sin_r   = np.sin(np.deg2rad(CANVAS_ROTATION_DEG))

solved_segments = []
speed_segments  = []   # per-waypoint draw speeds (parallel to solved_segments)
dense_segments  = []   # densified+smoothed canvas-mm paths (for speed re-computation)
total_pts = total_skipped = 0

for seg in segments:
    dense = densify(seg, MAX_SEG_MM)
    dense = smooth_path(dense)              # remove JPEG/quantisation noise
    speeds = adaptive_speeds(dense, DRAW_SPEED,
                             min_speed=MIN_DRAW_SPEED,
                             sensitivity=CURVE_SENSITIVITY)
    dense_segments.append(dense)

    solved     = []
    seg_speeds = []
    for i, (px_mm, py_mm) in enumerate(dense):
        # Apply canvas rotation around the canvas centre before IK mapping.
        dx = px_mm - half
        dy = py_mm - half
        px_r = cos_r * dx - sin_r * dy + half
        py_r = sin_r * dx + cos_r * dy + half

        rx = DRAW_CENTER_MM + (py_r - half) * CANVAS_SCALE_Y  # Top of image = Robot Far
        ry = px_r - half                                       # Top of image = Robot Right

        angles = kin.inverse_kinematics(rx, ry)
        if angles is None:
            total_skipped += 1
            solved.append(None)
            seg_speeds.append(None)
            continue
        t1_send = angles[0] + sh_offset
        t2_send = angles[1] + el_offset
        if not (0.0 <= t1_send <= MAX_ROTATION and 0.0 <= t2_send <= MAX_ROTATION):
            total_skipped += 1
            solved.append(None)
            seg_speeds.append(None)
            continue
        solved.append((round(t1_send, 3), round(t2_send, 3)))
        seg_speeds.append(speeds[i])
        total_pts += 1
    solved_segments.append(solved)
    speed_segments.append(seg_speeds)

print(f"  {len(segments)} stroke(s), {total_pts} valid waypoints  ({total_skipped} skipped)")

# ---------------------------------------------------------------------------
# 2a. Stroke ordering — greedy nearest-neighbour + 2-opt, with per-stroke
#     direction selection.  For each unvisited stroke the algorithm also
#     considers traversing it in reverse, picking whichever endpoint is
#     closer to the current arm position.  A 2-opt improvement pass then
#     removes crossing detours that the greedy pass missed.
#     Together these typically cut pen-up travel by 30–60 %.
# ---------------------------------------------------------------------------
def _stroke_endpoints(seg):
    """Return (first_valid_pt, last_valid_pt) for a solved segment."""
    fwd = next((pt for pt in seg           if pt is not None), None)
    bwd = next((pt for pt in reversed(seg) if pt is not None), None)
    return fwd, bwd

def _jdist(a, b):
    """Euclidean distance in joint-space (degrees)."""
    if a is None or b is None:
        return float('inf')
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5

def _seg_start(idx, rev):
    fwd, bwd = _stroke_endpoints(solved_segments[idx])
    return bwd if rev else fwd

def _seg_end(idx, rev):
    fwd, bwd = _stroke_endpoints(solved_segments[idx])
    return fwd if rev else bwd

# --- Greedy nearest-neighbour with per-stroke direction selection ---
current_pos = (SH_DRAW, EL_DRAW)
remaining   = list(range(len(solved_segments)))
ordered_idx = []
ordered_rev = []   # True = traverse this stroke in reverse direction

while remaining:
    best_i, best_dist, best_r = None, float('inf'), False
    none_segs = []
    for i in remaining:
        fwd, bwd = _stroke_endpoints(solved_segments[i])
        if fwd is None and bwd is None:
            none_segs.append(i)
            continue
        for rev, start in [(False, fwd), (True, bwd)]:
            d = _jdist(current_pos, start)
            if d < best_dist:
                best_dist, best_i, best_r = d, i, rev
    if best_i is None:
        # remaining are all-None segments — append and finish
        ordered_idx.extend(none_segs)
        ordered_rev.extend([False] * len(none_segs))
        break
    ordered_idx.append(best_i)
    ordered_rev.append(best_r)
    end_pt = _seg_end(best_i, best_r)
    current_pos = end_pt if end_pt is not None else current_pos
    remaining.remove(best_i)

# --- 2-opt improvement pass (O(n²) per pass, converges in 2–4 passes) ---
# For each pair of edges (i-1→i) and (j→j+1), test whether reversing the
# sub-sequence [i..j] — and flipping each stroke's direction within it —
# reduces total pen-up travel.  This catches detours that greedy missed.
improved = True
while improved:
    improved = False
    n = len(ordered_idx)
    for i in range(n - 1):
        for j in range(i + 1, n):
            pre  = (SH_DRAW, EL_DRAW) if i == 0 else _seg_end(ordered_idx[i-1], ordered_rev[i-1])
            post = None                if j == n-1 else _seg_start(ordered_idx[j+1], ordered_rev[j+1])
            old = (_jdist(pre, _seg_start(ordered_idx[i], ordered_rev[i]))
                 + (0.0 if post is None else _jdist(_seg_end(ordered_idx[j], ordered_rev[j]), post)))
            # After reversing [i..j]: approach from end[j], depart from start[i]
            new_head = _seg_end(ordered_idx[j],   ordered_rev[j])
            new_tail = _seg_start(ordered_idx[i], ordered_rev[i])
            new = (_jdist(pre, new_head)
                 + (0.0 if post is None else _jdist(new_tail, post)))
            if new < old - 0.1:
                ordered_idx[i:j+1] = list(reversed(ordered_idx[i:j+1]))
                ordered_rev[i:j+1] = [not r for r in reversed(ordered_rev[i:j+1])]
                improved = True
                break
        if improved:
            break

# Apply ordering and direction to all four parallel lists
n_rev = sum(ordered_rev)
segments        = [list(reversed(segments[i]))        if r else segments[i]
                   for i, r in zip(ordered_idx, ordered_rev)]
dense_segments  = [list(reversed(dense_segments[i]))  if r else dense_segments[i]
                   for i, r in zip(ordered_idx, ordered_rev)]
solved_segments = [list(reversed(solved_segments[i])) if r else solved_segments[i]
                   for i, r in zip(ordered_idx, ordered_rev)]
speed_segments  = [list(reversed(speed_segments[i]))  if r else speed_segments[i]
                   for i, r in zip(ordered_idx, ordered_rev)]
print(f"  Stroke order: {[i+1 for i in ordered_idx]}  ({n_rev} reversed for shorter travel)")

def _recompute_speeds(draw_spd):
    """Rebuild speed_segments for a new draw speed without re-running IK."""
    result = []
    for dense, solved in zip(dense_segments, solved_segments):
        spds = adaptive_speeds(dense, draw_spd,
                               min_speed=MIN_DRAW_SPEED,
                               sensitivity=CURVE_SENSITIVITY)
        result.append([spds[i] if pt is not None else None
                        for i, pt in enumerate(solved)])
    return result

# ---------------------------------------------------------------------------
# 2b. Path preview — render and show before moving the arm.
#     Displayed at a fixed ~500 px size so it fits on a laptop screen.
# ---------------------------------------------------------------------------
_SCALE = 6        # preview pixels per mm
_PAD   = 30
_W     = int(CANVAS_MM * _SCALE + _PAD * 2)
_H     = int(CANVAS_MM * _SCALE + _PAD * 2)
_canvas = np.ones((_H, _W, 3), dtype=np.uint8) * 255
_colors = [(220,60,60),(60,160,60),(60,60,220),(180,120,40),(120,40,180),(40,180,180)]

# Re-densify segments in canvas-mm space (same points used for IK) for preview
_dense_segs = [densify(s, MAX_SEG_MM) for s in segments]

for _si, _dense in enumerate(_dense_segs):
    _col = _colors[_si % len(_colors)]
    for _i, (_px_mm, _py_mm) in enumerate(_dense):
        _px = int(_PAD + _px_mm * _SCALE)
        _py = int(_PAD + _py_mm * _SCALE)
        cv2.circle(_canvas, (_px, _py), 1, _col, -1)
        if _i > 0:
            _prev = _dense[_i - 1]
            cv2.line(_canvas,
                     (int(_PAD + _prev[0]*_SCALE), int(_PAD + _prev[1]*_SCALE)),
                     (_px, _py), _col, 1, cv2.LINE_AA)
    _sx = int(_PAD + _dense[0][0] * _SCALE)
    _sy = int(_PAD + _dense[0][1] * _SCALE)
    cv2.circle(_canvas, (_sx, _sy), 4, (0, 0, 0), -1)
    cv2.putText(_canvas, str(_si + 1), (_sx + 5, _sy - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

cv2.rectangle(_canvas, (_PAD, _PAD),
              (int(_PAD + CANVAS_MM*_SCALE), int(_PAD + CANVAS_MM*_SCALE)),
              (200, 200, 200), 1)

_preview_path = 'assets/outputs/path_preview.png'
cv2.imwrite(_preview_path, _canvas)
print(f"\n[2b] Path preview saved → {_preview_path}")

# Resize to ~500 px so it fits comfortably on a laptop screen.
_DISPLAY_SIZE = 500
_display = cv2.resize(_canvas, (_DISPLAY_SIZE, _DISPLAY_SIZE), interpolation=cv2.INTER_AREA)
try:
    cv2.imshow("Path preview (press any key to continue)", _display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception:
    print("      (no display — open the PNG to inspect)")

print("\n[2c] Preview confirmed. Press ENTER to move robot to drawing position.")
input()

# ---------------------------------------------------------------------------
# 3. Move to drawing position.
#
# [3a] Shoulder + Z in parallel: MOVE_DONE fires when BOTH are at rest,
#      so one wait covers both (~25 s instead of ~43 s sequential).
# [3b] Elbow only: shoulder is already at SH_DRAW so MultiStepper just
#      drives the elbow to EL_DRAW.
# ---------------------------------------------------------------------------
send(f"S {TRAVEL_SPEED}")

# Move to drawing position in two blended phases:
#   Phase 1 — shoulder moves to halfway point, Z lifts, elbow stays put.
#   Phase 2 — queued move fires at the halfway blend point: shoulder continues
#             to SH_DRAW while elbow simultaneously moves from 0 → EL_DRAW.
#             MultiStepper proportions their speeds so both arrive together.
# Net effect: shoulder runs at full speed the entire time; elbow only starts
# when shoulder is halfway, so it doesn't slow the shoulder down.
SH_HALF = SH_DRAW / 2.0
print(f"\n[3a] Shoulder + Z to halfway ({SH_HALF}°), elbow queued for second half ...")
send(f"M {SH_HALF} 0")           # phase 1: shoulder to halfway, elbow at 0
send('U')                         # Z lifts in parallel
send(f"M {SH_DRAW} {EL_DRAW}")   # phase 2: queued — fires when shoulder ~= SH_HALF
wait_for_done(35.0, "shoulder halfway blend")   # MOVE_DONE from halfway blend trigger
wait_for_done(35.0, "drawing position")         # MOVE_DONE from final arrival
print("[OK] At drawing position.")

# ---------------------------------------------------------------------------
# 3c. Drawing parameters — set while the arm is settling.
#     Press Enter to keep the current value shown in brackets.
# ---------------------------------------------------------------------------
def _prompt_int(label, current, lo, hi, description):
    print(f"\n  {label} [{current}]")
    print(f"    {description}")
    _raw = input(f"    Enter value ({lo}–{hi}), or Enter to keep: ").strip()
    if not _raw:
        return current
    if _raw.isdigit() and lo <= int(_raw) <= hi:
        return int(_raw)
    print(f"    Invalid — keeping {current}.")
    return current

print("\n[3c] Drawing parameters (press Enter to keep each current value):")

DRAW_SPEED = _prompt_int(
    "DRAW_SPEED (steps/s)", DRAW_SPEED, 300, 1500,
    "Pen speed on straight lines. Higher = faster drawing, lower = more accurate.\n"
    "    Below ~300 causes motor resonance (zigzag lines). Typical: 400 fine detail, 800 simple shapes.")

MIN_DRAW_SPEED = _prompt_int(
    "MIN_DRAW_SPEED (steps/s)", MIN_DRAW_SPEED, 300, DRAW_SPEED,
    "Speed floor on tight curves. Lower = smoother circles and corners.\n"
    "    Must be ≤ DRAW_SPEED and ≥ 300 to stay above resonance band.")

CURVE_SENSITIVITY = _prompt_int(
    "CURVE_SENSITIVITY", CURVE_SENSITIVITY, 0, 200,
    "How aggressively curves slow the pen. 0 = constant speed (no adaptation).\n"
    "    30 = gentle, 60 = moderate (default at 800 steps/s), 120 = aggressive for logos/text.")

TRAVEL_SPEED = _prompt_int(
    "TRAVEL_SPEED (steps/s)", TRAVEL_SPEED, 500, 2000,
    "Pen-up move speed between strokes. Higher = shorter total drawing time.\n"
    "    2000 is the recommended maximum.")

speed_segments = _recompute_speeds(DRAW_SPEED)
print(f"\n  Final: DRAW_SPEED={DRAW_SPEED}  MIN={MIN_DRAW_SPEED}  SENSITIVITY={CURVE_SENSITIVITY}  TRAVEL={TRAVEL_SPEED}")

# ---------------------------------------------------------------------------
# 3d. Confirm iPad placement before drawing begins.
# ---------------------------------------------------------------------------
print("\n[3d] Is the iPad centered under the pen?")
print("     Press ENTER to begin drawing, Ctrl+C to abort.")
input()

MOVE_TIMEOUT = 30.0

# Backlash compensator: detects joint direction reversals between strokes and
# sends a tiny pre-move (0.05° ≈ 0.1 mm at tip) to take up gear slack before
# arriving at each stroke-start position.  Only applied on travel moves — not
# during drawing — so it has no effect on in-stroke waypoint pipelining.
backlash_comp = BacklashCompensator(backlash_sh=0.05, backlash_el=0.05)

for seg_idx, (solved, seg_spds) in enumerate(zip(solved_segments, speed_segments)):
    # Zip points with their curvature-adaptive speeds; drop invalid IK entries.
    valid_pairs = [(pt, spd) for pt, spd in zip(solved, seg_spds) if pt is not None]
    if not valid_pairs:
        continue

    print(f"\n  Stroke {seg_idx+1}/{len(solved_segments)}  ({len(valid_pairs)} pts) ...")

    (t1, t2), _ = valid_pairs[0]
    print(f"    [MOVE] Travel -> SH: {t1} EL: {t2}")
    send(f"S {TRAVEL_SPEED}")
    # Inter-stroke lift: use near-hover (N) instead of full hover (U).
    # 'N' lifts to Z_NEAR_HOVER_POS = 153 mm — 8 mm below full hover but
    # still clear of the surface — saving ~1.3 s of Z travel each direction.
    # The first stroke also uses 'N'; full hover ('U') is only used for the
    # initial positioning move and the final lift after drawing completes.
    send('N')
    wait_for_done(15.0, "pen up (near hover)")
    # Backlash take-up: if travel reverses a joint direction, send a tiny
    # pre-move (BACKLASH_DEG past the target) so gear slack is taken up
    # before the arm arrives at the stroke-start position.
    _pre = backlash_comp.pre_move(t1, t2)
    if _pre:
        send(f"M {_pre[0]} {_pre[1]}")
        wait_for_done(8.0, "backlash take-up")
    send(f"M {t1} {t2}")
    wait_for_done(35.0, "travel to stroke start")
    # Allow arm inertia to settle before lowering the pen.
    # MultiStepper has no decel ramp — a short pause lets oscillations die out
    # so the pen contacts the surface at a stable position.
    time.sleep(0.3)

    send('W')
    # Z_NEAR_HOVER_POS→Z_PRESS_POS = 25 mm at Z_PRESS_SPEED=1050 → ~10 s. Allow 12 s.
    wait_for_done(12.0, "pen down Z settle")
    # Allow pen-touchdown vibration to damp before the first draw move.
    time.sleep(0.3)

    draw_pairs = valid_pairs[1:]
    stroke_ok  = True

    if draw_pairs:
        # Pipelined drawing with per-waypoint adaptive speed.
        # Send S[n] immediately before M[n] — firmware processes them in order,
        # so when M[n] is dispatched (blend) the new maxSpeed is already set.
        #
        # Protocol:
        #   send S[0] M[0]     → starts at speed[0]
        #   send S[1] M[1]     → queued at speed[1]
        #   MOVE_DONE          → M[0] done, M[1] running at speed[1]
        #   send S[2] M[2]     → queued at speed[2]  ...
        (t1_0, t2_0), spd_0 = draw_pairs[0]
        send(f"S {int(round(spd_0))}")
        send(f"M {t1_0} {t2_0}")          # kick off first draw move

        for idx in range(1, len(draw_pairs)):
            (t1_d, t2_d), spd_d = draw_pairs[idx]
            send(f"S {int(round(spd_d))}")
            send(f"M {t1_d} {t2_d}")
            if not wait_for_done(MOVE_TIMEOUT, f"pt {idx - 1}"):
                print(f"  [ERROR] Timeout on pt {idx - 1}, aborting stroke")
                stroke_ok = False
                break

        if stroke_ok:
            if not wait_for_done(MOVE_TIMEOUT, "final pt"):
                print(f"  [ERROR] Timeout on final pt, aborting stroke")
                stroke_ok = False

    print(f"  {'done' if stroke_ok else 'aborted'}")

send('U')
wait_for_done(30.0, "final pen up")

print("\n[DONE] Drawing complete. Homing.")
send(f"S {TRAVEL_SPEED}")  # restore full speed so homing is not clamped to draw speed
send('H')
home_deadline = time.time() + 60.0
while time.time() < home_deadline:
    line = arduino.readline().decode('utf-8', errors='ignore').strip()
    if line:
        print(f"  [ARD] {line}")
    if "CALIBRATED" in line:
        print("[OK] Homed.")
        break
else:
    print("[WARN] Homing timed out after 60s.")
arduino.close()