"""
Trajectory planning for SCARA robot drawing paths.

Provides linear densification of closed 2D contour segments: inserts
intermediate waypoints so no single Cartesian move exceeds max_seg_mm.

Design choice — linear interpolation, not cubic splines:
  OpenCV's approxPolyDP already approximates curves as fine polygons
  (epsilon ≈ 0.4 mm for typical contours). Fitting a cubic spline through
  those corner points would curve through them and distort straight edges
  (Runge's phenomenon). Linear interpolation preserves straight lines
  exactly, and the chord error for a 3 mm segment on a typical 80 mm
  radius curve is only ~0.014 mm — well below any mechanical tolerance.
  This means the robot handles circles and curved logos correctly.

Usage:
    from trajectory import densify
    dense_pts = densify(segment_pts, max_seg_mm=3.0)
"""

import numpy as np


def densify(seg, max_seg_mm):
    """
    Resample a closed 2D contour segment so that no consecutive pair of
    waypoints is farther apart than *max_seg_mm* (Cartesian distance).

    Parameters
    ----------
    seg : list of (x_mm, y_mm) tuples
        Key-points of the closed contour (do NOT repeat the first point).
    max_seg_mm : float
        Maximum Cartesian distance between consecutive output points (mm).

    Returns
    -------
    list of (x_mm, y_mm) tuples
        Densely sampled points, including the closing segment back to seg[0].
    """
    closed = seg + [seg[0]]
    dense  = []
    for i in range(len(closed) - 1):
        p0, p1 = closed[i], closed[i + 1]
        dense.append(p0)
        d = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
        n_steps = max(1, int(np.ceil(d / max_seg_mm)))
        for k in range(1, n_steps):
            t = k / n_steps
            dense.append((p0[0] + t * (p1[0] - p0[0]),
                           p0[1] + t * (p1[1] - p0[1])))
    dense.append(closed[-1])
    return dense
