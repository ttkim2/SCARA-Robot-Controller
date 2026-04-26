"""
control.py — Drawing accuracy algorithms for the SCARA robot.

Functions
---------
smooth_path(pts, window, poly)
    Savitzky-Golay smoothing of a 2D waypoint list.
    Removes JPEG compression noise and pixel-quantisation jitter from
    contour edges without rounding corners appreciably.  Applied in
    canvas-mm space before inverse-kinematics solving.

path_curvature(pts)
    Menger curvature (mm⁻¹) at every waypoint using the three-point
    osculating-circle formula.  Higher value = tighter bend.

adaptive_speeds(pts, base_speed, min_speed, sensitivity)
    Per-waypoint draw speed that reduces at tight curves and restores at
    straight sections.  Passed to the 'S' firmware command so each
    segment runs at the optimal speed:
      - Straight lines  → base_speed   (full speed, minimal vibration)
      - Tight curves    → min_speed    (controlled pen placement)

Usage in main.py
----------------
    from control import smooth_path, adaptive_speeds

    dense = densify(seg, MAX_SEG_MM)
    dense = smooth_path(dense)                        # noise removal
    speeds = adaptive_speeds(dense, DRAW_SPEED)       # curvature-adaptive
"""

import numpy as np


# ---------------------------------------------------------------------------
# Path smoothing
# ---------------------------------------------------------------------------

def smooth_path(pts, window=5, poly=2):
    """
    Savitzky-Golay smoothing of a closed 2D waypoint sequence.

    Parameters
    ----------
    pts    : list of (x_mm, y_mm)
    window : filter window length (odd integer ≥ 3).
             Larger window = more smoothing but more corner rounding.
             Default 5 works well for MAX_SEG_MM = 2–5.
    poly   : polynomial order for the SG fit (default 2 = quadratic).

    Returns
    -------
    list of (x_mm, y_mm) — same length as input.

    Falls back to a Gaussian-weighted moving average if scipy is not
    installed (slightly softer corners, but still effective).
    """
    n = len(pts)
    if n < window + 2:
        return pts                  # too short to filter meaningfully

    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)

    try:
        from scipy.signal import savgol_filter
        # Wrap-pad for closed-path boundary continuity.
        pad  = window
        xs_s = savgol_filter(np.pad(xs, pad, mode='wrap'), window, poly)[pad:-pad]
        ys_s = savgol_filter(np.pad(ys, pad, mode='wrap'), window, poly)[pad:-pad]
    except ImportError:
        # Gaussian-weighted moving average fallback.
        hw   = window // 2
        k    = np.arange(-hw, hw + 1, dtype=float)
        w    = np.exp(-k ** 2 / (2 * (window / 4.0) ** 2))
        w   /= w.sum()
        xs_s = np.convolve(np.pad(xs, hw, mode='wrap'), w, mode='valid')[:n]
        ys_s = np.convolve(np.pad(ys, hw, mode='wrap'), w, mode='valid')[:n]

    return list(zip(xs_s.tolist(), ys_s.tolist()))


# ---------------------------------------------------------------------------
# Curvature
# ---------------------------------------------------------------------------

def path_curvature(pts):
    """
    Compute unsigned Menger curvature at each waypoint (mm⁻¹).

    Uses the circumradius of the triangle formed by three consecutive
    points: κ = 1/R = 4·Area / (a·b·c).

    Returns
    -------
    numpy array, shape (len(pts),).
    Value is 0 on perfectly straight sections, large on tight bends.
    """
    n   = len(pts)
    arr = np.asarray(pts, dtype=float)
    k   = np.zeros(n)

    for i in range(n):
        p0 = arr[(i - 1) % n]
        p1 = arr[i]
        p2 = arr[(i + 1) % n]

        a     = np.linalg.norm(p1 - p0)
        b     = np.linalg.norm(p2 - p1)
        c     = np.linalg.norm(p2 - p0)
        area  = abs((p1[0] - p0[0]) * (p2[1] - p0[1])
                  - (p2[0] - p0[0]) * (p1[1] - p0[1])) * 0.5
        denom = a * b * c
        k[i]  = 4.0 * area / denom if denom > 1e-9 else 0.0

    return k


# ---------------------------------------------------------------------------
# Adaptive speed profile
# ---------------------------------------------------------------------------

def adaptive_speeds(pts, base_speed, min_speed=None, sensitivity=90.0):
    """
    Per-waypoint draw speed based on local path curvature.

    Speed = base_speed / (1 + sensitivity * κ),  clamped to [min_speed, base_speed].

    Parameters
    ----------
    pts         : list of (x_mm, y_mm)
    base_speed  : maximum draw speed in steps/s (DRAW_SPEED config value).
    min_speed   : minimum allowed speed in steps/s (MIN_DRAW_SPEED config value).
                  Defaults to max(300, base_speed * 0.35) — allows up to 65 %
                  slowdown on tight curves while staying above stepper resonance.
    sensitivity : how strongly curvature reduces speed (CURVE_SENSITIVITY config).
                  0   → disable (constant speed)
                  30  → gentle   (designed for base_speed ≈ 400)
                  60  → moderate (default; designed for base_speed = 800)
                  120 → aggressive (tight logos / small text)

    Returns
    -------
    list of float, one speed per waypoint.

    Typical effect at sensitivity=60, base=800 steps/s, min=300 steps/s
    ---------------------------------------------------------------------
    Straight line  (κ ≈ 0)       → 800 steps/s
    Circle R=80mm  (κ = 0.0125)  → 800 / (1 + 0.75)  ≈ 457 steps/s
    Curve  R=20mm  (κ = 0.05)    → 800 / (1 + 3.0)   = 200 → clamped to 300
    Tight  R=10mm  (κ = 0.1)     → 800 / (1 + 6.0)   = 114 → clamped to 300
    """
    if min_speed is None:
        min_speed = max(300.0, base_speed * 0.35)

    curvatures = path_curvature(pts)
    speeds = []
    for kap in curvatures:
        s = base_speed / (1.0 + sensitivity * float(kap))
        speeds.append(float(np.clip(s, min_speed, base_speed)))
    return speeds
