import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VERIFY_TOL_MM = 0.5     # mm — FK round-trip tolerance for IK verification
DEG           = np.pi / 180.0
RAD           = 180.0 / np.pi


class KinematicsEngine:

    def __init__(self, L1: float = 220.0, L2: float = 212.5,
                 max_rotation: float = 330.0):
        self.L1           = L1
        self.L2           = L2
        self.MAX_ROT      = max_rotation          # degrees, both joints

        # Reachability bounds (with small safety margins)
        self.MIN_REACH    = abs(L1 - L2) + 10.0  # avoid singularity at full extension inward
        self.MAX_REACH    = (L1 + L2) - 5.0      # avoid singularity at full extension outward

    # -----------------------------------------------------------------------
    # DH TRANSFORMATION MATRIX  (planar, α=0, d=0)
    # -----------------------------------------------------------------------
    @staticmethod
    def dh_matrix(theta_deg: float, a: float) -> np.ndarray:
        t   = theta_deg * DEG
        ct  = np.cos(t)
        st  = np.sin(t)
        return np.array([
            [ct, -st, a * ct],
            [st,  ct, a * st],
            [ 0,   0,      1],
        ])

    # -----------------------------------------------------------------------
    # FORWARD KINEMATICS
    # -----------------------------------------------------------------------
    def forward_kinematics(self, t1_deg: float, t2_deg: float
                           ) -> tuple[float, float, float]:
        T01 = self.dh_matrix(t1_deg,         self.L1)
        T12 = self.dh_matrix(t2_deg,         self.L2)
        T02 = T01 @ T12

        x   = T02[0, 2]
        y   = T02[1, 2]
        phi = (t1_deg + t2_deg) % 360.0
        return x, y, phi

    # -----------------------------------------------------------------------
    # INVERSE KINEMATICS  (elbow-up)
    # -----------------------------------------------------------------------
    def inverse_kinematics(self, x: float, y: float
                           ) -> tuple[float, float] | None:
        dist_sq = x**2 + y**2
        dist    = np.sqrt(dist_sq)

        # Reachability check
        if dist < self.MIN_REACH or dist > self.MAX_REACH:
            return None

        # Cosine rule for θ₂
        D = (dist_sq - self.L1**2 - self.L2**2) / (2.0 * self.L1 * self.L2)
        D = np.clip(D, -1.0, 1.0)           # guard against floating-point drift

        # Elbow-up: negative arccos gives the solution where the elbow
        # extends away from the robot body (not into it toward the limit switch)
        t2_raw = -np.arccos(D)

        # θ₁ via atan2 decomposition
        t1_raw = np.arctan2(y, x) - np.arctan2(
            self.L2 * np.sin(t2_raw),
            self.L1 + self.L2 * np.cos(t2_raw)
        )

        # Convert to degrees and normalise to [0, 360)
        t1 = (t1_raw * RAD) % 360.0
        t2 = (t2_raw * RAD) % 360.0

        # Safety / workspace check
        if not (0.0 <= t1 <= self.MAX_ROT and 0.0 <= t2 <= self.MAX_ROT):
            return None

        # FK verification — reject if round-trip error exceeds tolerance
        x_fk, y_fk, _ = self.forward_kinematics(t1, t2)
        err = np.sqrt((x_fk - x)**2 + (y_fk - y)**2)
        if err > VERIFY_TOL_MM:
            return None

        return round(t1, 3), round(t2, 3)

    # -----------------------------------------------------------------------
    # WORKSPACE VALIDATOR  (used before drawing phase begins)
    # -----------------------------------------------------------------------
    def validate_workspace(self, ipad_x: float, ipad_y: float,
                           size_mm: float = 120.0) -> tuple[bool, str]:
        """
        Checks whether the entire drawing workspace square (size_mm × size_mm)
        centred on (ipad_x, ipad_y) is reachable.

        Returns (ok: bool, message: str).
        If not ok, message gives directional advice to reposition the iPad.
        """
        half   = size_mm / 2.0
        corners = [
            (ipad_x - half, ipad_y - half),
            (ipad_x + half, ipad_y - half),
            (ipad_x - half, ipad_y + half),
            (ipad_x + half, ipad_y + half),
        ]
        # Also check a grid of interior points (not just corners)
        test_points = list(corners)
        for fx in np.linspace(ipad_x - half, ipad_x + half, 5):
            for fy in np.linspace(ipad_y - half, ipad_y + half, 5):
                test_points.append((fx, fy))

        failed = [p for p in test_points if self.inverse_kinematics(*p) is None]

        if not failed:
            dist = np.sqrt(ipad_x**2 + ipad_y**2)
            return True, f"READY — {len(test_points)} points checked, all reachable. " \
                         f"iPad at ({ipad_x:.0f}, {ipad_y:.0f}) mm, dist={dist:.0f} mm."

        # Directional advice
        dist = np.sqrt(ipad_x**2 + ipad_y**2)
        if dist < self.MIN_REACH + half:
            msg = "PUSH iPAD AWAY — workspace overlaps inner dead zone."
        elif dist > self.MAX_REACH - half:
            msg = "PULL iPAD CLOSER — workspace extends beyond max reach."
        elif any(p[1] < ipad_y - half * 0.8 for p in failed):
            msg = "SHIFT iPAD LEFT — left side of workspace unreachable."
        elif any(p[1] > ipad_y + half * 0.8 for p in failed):
            msg = "SHIFT iPAD RIGHT — right side of workspace unreachable."
        else:
            msg = f"ADJUST iPAD — {len(failed)}/{len(test_points)} points unreachable."

        return False, msg

    # -----------------------------------------------------------------------
    # SELF-TEST
    # -----------------------------------------------------------------------
    def run_tests(self) -> bool:
        """
        Runs a suite of FK and IK round-trip tests.
        Prints results. Returns True if all pass.
        """
        print("=" * 60)
        print("KinematicsEngine self-test")
        print(f"  L1={self.L1} mm  L2={self.L2} mm  max={self.MAX_ROT}°")
        print("=" * 60)
        all_pass = True

        # ---- FK tests: known angles → expected position ----
        fk_cases = [
            # (t1_deg, t2_deg, expected_x, expected_y, label)
            (0.0,    0.0,   self.L1 + self.L2, 0.0,         "Both at 0° — full extension along +x"),
            (90.0,   0.0,   0.0,               self.L1 + self.L2, "Shoulder 90° — full extension along +y"),
            (45.0,   0.0,   (self.L1+self.L2)*np.cos(45*DEG),
                            (self.L1+self.L2)*np.sin(45*DEG), "Shoulder 45°, elbow 0°"),
            (0.0,   90.0,   self.L1,            self.L2,     "Elbow 90° — L-shape"),
            (90.0,  270.0,  self.L2 * np.cos((90+270)*DEG),
                            self.L1 + self.L2 * np.sin((90+270)*DEG),
                            "Elbow 270° — arm folded"),
        ]

        print("\n  Forward kinematics:")
        for t1, t2, ex, ey, label in fk_cases:
            x, y, phi = self.forward_kinematics(t1, t2)
            err = np.sqrt((x - ex)**2 + (y - ey)**2)
            ok  = err < 0.01
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"    [{status}] {label}")
            if not ok:
                print(f"           expected ({ex:.2f}, {ey:.2f}), got ({x:.2f}, {y:.2f}), err={err:.4f} mm")

        # ---- IK round-trip tests: target → IK → FK → compare ----
        print("\n  IK round-trip (target → IK → FK → error):")
        ik_targets = [
            (300.0,   0.0,  "Straight ahead 300 mm"),
            (250.0,  50.0,  "Forward-right 250,50"),
            (250.0, -50.0,  "Forward-left 250,-50"),
            (200.0,   0.0,  "Close 200 mm"),
            (400.0,   0.0,  "Far 400 mm"),
            (280.0,  60.0,  "Diagonal"),
            (180.0,  30.0,  "Near edge"),
            (395.0, -20.0,  "Far edge within 330° limit"),
        ]

        for tx, ty, label in ik_targets:
            result = self.inverse_kinematics(tx, ty)
            if result is None:
                # Check if it's legitimately out of reach
                dist = np.sqrt(tx**2 + ty**2)
                if dist < self.MIN_REACH or dist > self.MAX_REACH:
                    print(f"    [SKIP] {label} — out of reach (dist={dist:.0f} mm)")
                else:
                    print(f"    [FAIL] {label} — IK returned None unexpectedly")
                    all_pass = False
                continue
            t1, t2 = result
            x_fk, y_fk, _ = self.forward_kinematics(t1, t2)
            err = np.sqrt((x_fk - tx)**2 + (y_fk - ty)**2)
            ok  = err < VERIFY_TOL_MM
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"    [{status}] {label}: θ₁={t1}° θ₂={t2}° → FK err={err:.4f} mm")

        # ---- Boundary / safety tests ----
        print("\n  Safety and boundary:")
        boundary_cases = [
            (0.0,    0.0,   True,  "Origin — inside dead zone, should fail"),
            (1000.0, 0.0,   True,  "Way out of reach, should fail"),
            (self.MIN_REACH - 1, 0.0, True,  "Just inside min reach, should fail"),
            (self.MAX_REACH + 1, 0.0, True,  "Just outside max reach, should fail"),
        ]
        for tx, ty, expect_none, label in boundary_cases:
            result = self.inverse_kinematics(tx, ty)
            ok = (result is None) == expect_none
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"    [{status}] {label}")

        # ---- Workspace validation test ----
        print("\n  Workspace validation:")
        wv_cases = [
            (280.0,  0.0,  "iPad 280mm ahead — should be READY"),
            (50.0,   0.0,  "iPad 50mm ahead — too close"),
            (450.0,  0.0,  "iPad 450mm ahead — too far"),
        ]
        for ix, iy, label in wv_cases:
            ok, msg = self.validate_workspace(ix, iy)
            print(f"    {'[OK]' if ok else '[--]'} {label}: {msg}")

        print("\n" + ("=" * 60))
        print(f"  Result: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
        print("=" * 60)
        return all_pass


# ---------------------------------------------------------------------------
# Backlash compensation
# ---------------------------------------------------------------------------

class BacklashCompensator:
    """
    Tracks per-joint motion direction and generates a small 'take-up' pre-move
    whenever a joint reverses direction.

    Mechanical backlash (gear play) means the first few steps after a reversal
    move the motor shaft without moving the output link — the output only starts
    moving once the slack is taken up.  Sending a tiny over-drive move in the
    new direction before the real move eliminates this dead-band.

    Parameters
    ----------
    backlash_sh : shoulder take-up in degrees (default 0.05° ≈ 0.1 mm at tip)
    backlash_el : elbow    take-up in degrees (default 0.05°)

    Usage (in the drawing loop — applied at stroke-start travel moves)
    ------------------------------------------------------------------
        comp = BacklashCompensator()
        pre  = comp.pre_move(t1, t2)   # None, or (sh, el) take-up position
        if pre:
            send(f"M {pre[0]} {pre[1]}")
            wait_for_done(5.0, "backlash take-up")
        send(f"M {t1} {t2}")
    """

    def __init__(self, backlash_sh: float = 0.05, backlash_el: float = 0.05):
        self.bl_sh     = backlash_sh
        self.bl_el     = backlash_el
        self._last_sh: float | None = None
        self._last_el: float | None = None
        self._dir_sh: int = 0   # +1 CCW, -1 CW, 0 unknown
        self._dir_el: int = 0

    def pre_move(self, sh: float, el: float) -> tuple[float, float] | None:
        """
        Given the next target (sh, el), return a backlash pre-move position if
        either joint is reversing direction, else None.

        The pre-move overshoots the real target by backlash_deg in the new
        direction so that when the real move runs, gear slack is already taken up.
        """
        if self._last_sh is None:
            self._last_sh, self._last_el = sh, el
            return None

        new_dir_sh = self._sign(sh - self._last_sh)
        new_dir_el = self._sign(el - self._last_el)
        rev_sh = (new_dir_sh != 0 and self._dir_sh != 0 and new_dir_sh != self._dir_sh)
        rev_el = (new_dir_el != 0 and self._dir_el != 0 and new_dir_el != self._dir_el)

        pre = None
        if rev_sh or rev_el:
            pre_sh = sh + (new_dir_sh * self.bl_sh if rev_sh else 0.0)
            pre_el = el + (new_dir_el * self.bl_el if rev_el else 0.0)
            pre = (round(pre_sh, 3), round(pre_el, 3))

        if new_dir_sh != 0:
            self._dir_sh = new_dir_sh
        if new_dir_el != 0:
            self._dir_el = new_dir_el
        self._last_sh, self._last_el = sh, el
        return pre

    def reset(self):
        """Call between strokes if the robot has been homed or repositioned."""
        self._last_sh = None
        self._last_el = None
        self._dir_sh  = 0
        self._dir_el  = 0

    @staticmethod
    def _sign(v: float) -> int:
        if v >  1e-4: return  1
        if v < -1e-4: return -1
        return 0

if __name__ == "__main__":
    kin = KinematicsEngine()
    kin.run_tests()