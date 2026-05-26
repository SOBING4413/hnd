"""
particle_engine.py
------------------
NexusFlow — Fully vectorised NumPy particle physics engine.

Developed by sobing4413 — Organization: Exter Interactive
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Physics constants ────────────────────────────────────────────────────────
TRAIL_LENGTH          = 10
HAND_INFLUENCE_RAD    = 260.0
HAND_REPEL_STRENGTH   = 2000.0
HAND_ATTRACT_STRENGTH = 1100.0
FORMATION_SPRING      = 0.10
DAMPING               = 0.80
TURBULENCE            = 0.55
MAX_SPEED             = 26.0
EXPLODE_STRENGTH      = 2800.0
GRAVITY_DRIFT         = 0.07      # ambient orbital curl
SHOCKWAVE_STRENGTH    = 5500.0
RIPPLE_SPEED          = 280.0     # px/s ring expansion
CHARGE_SCALE          = 2.5       # peak force multiplier at full charge
CHARGE_RAMP_SEC       = 2.0       # seconds to reach full charge
SHOCKWAVE_LIFETIME    = 1.2       # seconds before shockwave expires
RIPPLE_LIFETIME       = 2.0       # seconds before ripple ring expires
# ─────────────────────────────────────────────────────────────────────────────

# Typed dict for shockwave/ripple ring records
_ShockwaveEntry = Dict[str, float]
_RippleEntry    = Dict[str, float]


class ParticleEngine:
    """
    Fully vectorised particle system.

    State arrays (float32, shape [N]):
        px, py         – positions
        vx, vy         – velocities
        depth          – [0.3, 1.0] parallax/size multiplier
        hue_off        – per-particle hue shimmer offset  [0, 360)
        age            – per-particle normalised age [0, 1)
        colour_temp    – [0=cool, 1=warm] per particle

    trail_px/py        – circular buffer of TRAIL_LENGTH position snapshots
    """

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        n_particles: int = 4500,
    ) -> None:
        self.W = screen_w
        self.H = screen_h
        self._N = int(max(100, n_particles))
        N = self._N

        self._rng = np.random.default_rng()

        # ── Particle state arrays ─────────────────────────────────────────────
        self.px          = self._rng.uniform(0, self.W, N).astype(np.float32)
        self.py          = self._rng.uniform(0, self.H, N).astype(np.float32)
        self.vx          = self._rng.uniform(-1.5, 1.5, N).astype(np.float32)
        self.vy          = self._rng.uniform(-1.5, 1.5, N).astype(np.float32)
        self.depth       = self._rng.uniform(0.3, 1.0, N).astype(np.float32)
        self.hue_off     = self._rng.uniform(0, 360, N).astype(np.float32)
        self.age         = self._rng.uniform(0, 1,   N).astype(np.float32)
        self.colour_temp = np.zeros(N, dtype=np.float32)

        # ── Trail circular buffer ─────────────────────────────────────────────
        self.trail_px    = np.empty((TRAIL_LENGTH, N), dtype=np.float32)
        self.trail_py    = np.empty((TRAIL_LENGTH, N), dtype=np.float32)
        # Initialise all trail slots to starting positions
        self.trail_px[:] = self.px
        self.trail_py[:] = self.py
        self._trail_head = 0
        self._frames_run = 0

        # ── Formation state ───────────────────────────────────────────────────
        self._targets:          Optional[np.ndarray] = None   # (M, 2)
        self._t_assign:         Optional[np.ndarray] = None   # (N,) int indices
        self._formation_alpha:  float = 0.0

        self.mode:        str   = "free"
        self.vortex_angle: float = 0.0

        # ── Shockwave / ripple state ──────────────────────────────────────────
        self._shockwaves:   List[_ShockwaveEntry] = []
        self._ripple_rings: List[_RippleEntry]    = []

        # ── Charge system ─────────────────────────────────────────────────────
        self.charge_level:  float = 0.0
        self._charge_timer: float = 0.0

        # ── Simulation clock ──────────────────────────────────────────────────
        self._t: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def set_formation(self, targets: np.ndarray) -> None:
        """Assign particle targets from an (M, 2) array and enter 'form' mode."""
        if targets is None or len(targets) == 0:
            self.clear_formation()
            return
        M = len(targets)
        self._targets        = targets
        self._t_assign       = self._rng.integers(0, M, self._N)
        self._formation_alpha = 0.0
        self.mode            = "form"

    def set_formation_direct(
        self, targets: np.ndarray, mode: str = "form"
    ) -> None:
        """
        Directly set targets + mode without resetting formation_alpha.
        Useful for modes that manage alpha themselves (heart, galaxy…).
        """
        if targets is None or len(targets) == 0:
            self.clear_formation()
            return
        M = len(targets)
        self._targets         = targets
        self._t_assign        = self._rng.integers(0, M, self._N)
        self._formation_alpha = 0.0
        self.mode             = mode

    def clear_formation(self) -> None:
        self._targets         = None
        self._t_assign        = None
        self._formation_alpha = 0.0

    def set_mode(self, mode: str) -> None:
        if mode not in ("form", "heart", "explode", "galaxy", "ripple"):
            self.clear_formation()
        self.mode = mode

    def trigger_shockwave(
        self, cx: float, cy: float, strength: float = 1.0
    ) -> None:
        """Emit an expanding shockwave ring from (cx, cy)."""
        self._shockwaves.append({
            "cx": cx, "cy": cy,
            "r": 0.0, "strength": float(strength),
            "t": 0.0,
        })

    def add_ripple(
        self, cx: float, cy: float, max_r: float = 300.0
    ) -> None:
        """Add a visual ripple ring (also applies outward push force)."""
        self._ripple_rings.append({
            "cx": cx, "cy": cy,
            "r": 0.0, "max_r": float(max_r),
            "born": self._t,
        })

    def charge(self, dt: float) -> None:
        """Build charge while a gesture is held; call every frame."""
        self._charge_timer = min(self._charge_timer + dt, CHARGE_RAMP_SEC)
        self.charge_level  = self._charge_timer / CHARGE_RAMP_SEC

    def release_charge(self) -> float:
        """Return accumulated charge (0–1) and reset the counter."""
        lvl = self.charge_level
        self.charge_level  = 0.0
        self._charge_timer = 0.0
        return lvl

    def update(
        self,
        hand_pos:         Optional[Tuple[float, float]],
        gesture:          str,
        dt:               float = 1.0 / 60.0,
        gesture_hold_time: float = 0.0,
        palm_velocity:    Tuple[float, float] = (0.0, 0.0),
        finger_tips:      Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """Advance the full particle simulation by one frame."""
        self._t += dt
        N  = self._N
        px = self.px
        py = self.py
        vx = self.vx
        vy = self.vy

        # Charge strength multiplier from gesture hold time
        charge_mul = 1.0 + (
            min(gesture_hold_time, CHARGE_RAMP_SEC) / CHARGE_RAMP_SEC
        ) * (CHARGE_SCALE - 1.0)

        # Pre-allocate acceleration arrays
        ax = np.zeros(N, dtype=np.float64)
        ay = np.zeros(N, dtype=np.float64)

        # ── 1. Turbulence ─────────────────────────────────────────────────────
        noise = self._rng.standard_normal((2, N))
        ax   += noise[0] * TURBULENCE
        ay   += noise[1] * TURBULENCE

        # ── 2. Hand palm influence ────────────────────────────────────────────
        if hand_pos is not None:
            hx, hy = hand_pos
            dx  = px - hx
            dy  = py - hy
            r2  = dx * dx + dy * dy + 1.0
            r   = np.sqrt(r2)
            mask = r < HAND_INFLUENCE_RAD

            if np.any(mask):
                if gesture == "open" or self.mode == "explode":
                    s = EXPLODE_STRENGTH * charge_mul / (r2[mask] + 80.0)
                    ax[mask] += dx[mask] * s
                    ay[mask] += dy[mask] * s

                elif gesture == "fist":
                    s = HAND_ATTRACT_STRENGTH * charge_mul / (r2[mask] + 80.0)
                    ax[mask] -= dx[mask] * s
                    ay[mask] -= dy[mask] * s

                elif gesture == "pinch":
                    self.vortex_angle += dt * (2.5 + charge_mul * 1.5)
                    r_safe   = r[mask] + 1.0
                    tang_x   =  dy[mask] / r_safe
                    tang_y   = -dx[mask] / r_safe
                    s_mask   = (380.0 * charge_mul) / (r[mask] + 25.0)
                    ax[mask] += tang_x * s_mask
                    ay[mask] += tang_y * s_mask

                elif gesture == "thumbup":
                    ay[mask] -= 6.0 * charge_mul * (
                        1.0 - r[mask] / HAND_INFLUENCE_RAD
                    )
                    ax[mask] += (hx - px[mask]) * 0.002 * charge_mul

                elif gesture == "spider":
                    ang   = self._t * 1.8
                    arm_x = np.cos(ang) * 200.0
                    arm_y = np.sin(ang) * 200.0
                    pull_x = (hx + arm_x) - px
                    pull_y = (hy + arm_y) - py
                    ax += pull_x * 0.004 * charge_mul
                    ay += pull_y * 0.004 * charge_mul

                elif gesture == "wave":
                    svx, svy = palm_velocity
                    pspd = (svx ** 2 + svy ** 2) ** 0.5 + 1e-6
                    push = min(pspd * 0.8, 2000.0)
                    s    = push / (r2[mask] + 200.0)
                    ax[mask] += (svx / pspd) * s * 80.0
                    ay[mask] += (svy / pspd) * s * 80.0

                elif gesture == "three":
                    tri_r = 140.0
                    for k in range(3):
                        angle = 2.0 * np.pi * k / 3.0
                        tpx   = hx + tri_r * np.cos(angle)
                        tpy   = hy + tri_r * np.sin(angle)
                        ddx   = tpx - px
                        ddy   = tpy - py
                        dr2   = ddx * ddx + ddy * ddy + 1.0
                        dmsk  = np.sqrt(dr2) < HAND_INFLUENCE_RAD * 1.3
                        ax[dmsk] += ddx[dmsk] * 0.015 * charge_mul
                        ay[dmsk] += ddy[dmsk] * 0.015 * charge_mul

                else:
                    # Gentle deflection for unclassified poses
                    s = 280.0 / (r2[mask] + 350.0)
                    ax[mask] += dx[mask] * s
                    ay[mask] += dy[mask] * s

        # ── 2b. Per-fingertip influence ───────────────────────────────────────
        if finger_tips:
            for fi, (ftx, fty) in enumerate(finger_tips):
                fdx  = px - ftx
                fdy  = py - fty
                fr2  = fdx * fdx + fdy * fdy + 1.0
                fr   = np.sqrt(fr2)
                fmask = fr < 120.0
                if not np.any(fmask):
                    continue

                if gesture == "open":
                    s2 = 600.0 * charge_mul / (fr2[fmask] + 40.0)
                    ax[fmask] += fdx[fmask] * s2
                    ay[fmask] += fdy[fmask] * s2
                elif gesture == "fist":
                    s2 = 400.0 * charge_mul / (fr2[fmask] + 40.0)
                    ax[fmask] -= fdx[fmask] * s2
                    ay[fmask] -= fdy[fmask] * s2
                elif gesture == "peace" and fi in (1, 2):
                    fr_s   = fr[fmask] + 1.0
                    tang_x =  fdy[fmask] / fr_s
                    tang_y = -fdx[fmask] / fr_s
                    s2 = 300.0 * charge_mul / (fr[fmask] + 20.0)
                    ax[fmask] += tang_x * s2
                    ay[fmask] += tang_y * s2
                elif gesture == "point" and fi == 1:
                    s2 = 900.0 * charge_mul / (fr2[fmask] + 60.0)
                    ax[fmask] -= fdx[fmask] * s2
                    ay[fmask] -= fdy[fmask] * s2
                elif gesture == "pinch" and fi in (0, 1):
                    fr_s   = fr[fmask] + 1.0
                    tang_x =  fdy[fmask] / fr_s
                    tang_y = -fdx[fmask] / fr_s
                    s2 = 500.0 * charge_mul / (fr[fmask] + 15.0)
                    ax[fmask] += tang_x * s2
                    ay[fmask] += tang_y * s2
                elif gesture == "thumbup" and fi == 0:
                    ay[fmask] -= 12.0 * charge_mul * (
                        1.0 - fr[fmask] / 120.0
                    )
                elif gesture == "spider" and fi in (0, 1, 4):
                    s2 = 450.0 * charge_mul / (fr2[fmask] + 50.0)
                    ax[fmask] -= fdx[fmask] * s2
                    ay[fmask] -= fdy[fmask] * s2
                elif gesture == "three" and fi in (1, 2, 3):
                    s2 = 380.0 * charge_mul / (fr2[fmask] + 50.0)
                    ax[fmask] -= fdx[fmask] * s2
                    ay[fmask] -= fdy[fmask] * s2
                else:
                    s2 = 160.0 / (fr2[fmask] + 200.0)
                    ax[fmask] += fdx[fmask] * s2
                    ay[fmask] += fdy[fmask] * s2

                # Per-finger hue heat
                self.hue_off[fmask] = (
                    self.hue_off[fmask]
                    + fi * 18.0 * (1.0 - fr[fmask] / 120.0)
                ) % 360.0

        # ── 3. Shockwave rings ────────────────────────────────────────────────
        alive_sw: List[_ShockwaveEntry] = []
        for sw in self._shockwaves:
            sw["r"] += RIPPLE_SPEED * dt
            sw["t"] += dt
            if sw["t"] >= SHOCKWAVE_LIFETIME:
                continue
            alive_sw.append(sw)
            cx_s, cy_s = sw["cx"], sw["cy"]
            ddx  = px - cx_s
            ddy  = py - cy_s
            r_s  = np.sqrt(ddx * ddx + ddy * ddy) + 1e-6
            ring_r  = sw["r"]
            ring_w  = 60.0
            in_ring = np.abs(r_s - ring_r) < ring_w
            if np.any(in_ring):
                fade = 1.0 - sw["t"] / SHOCKWAVE_LIFETIME
                push = (
                    SHOCKWAVE_STRENGTH * sw["strength"] * fade / (ring_w ** 2)
                ) * np.maximum(0.0, ring_w - np.abs(r_s - ring_r))
                ax[in_ring] += (ddx[in_ring] / r_s[in_ring]) * push[in_ring]
                ay[in_ring] += (ddy[in_ring] / r_s[in_ring]) * push[in_ring]
        self._shockwaves = alive_sw

        # ── 4. Formation spring ───────────────────────────────────────────────
        if (
            self.mode in ("form", "explode", "heart", "galaxy")
            and self._targets is not None
            and self._t_assign is not None
        ):
            ramp = 0.9 if self.mode == "galaxy" else 0.75
            self._formation_alpha = min(
                1.0, self._formation_alpha + dt * ramp
            )
            alpha  = self._formation_alpha
            tx     = self._targets[self._t_assign, 0]
            ty     = self._targets[self._t_assign, 1]
            spring = FORMATION_SPRING * alpha
            if self.mode == "explode":
                spring *= 0.7   # looser — explode first, then slowly assemble
            ax += (tx - px) * spring
            ay += (ty - py) * spring

        elif self.mode == "gather":
            cx_g, cy_g = self.W * 0.5, self.H * 0.5
            ax += (cx_g - px) * 0.04
            ay += (cy_g - py) * 0.04

        # ── 5. Ambient orbital drift (galaxy curl) ────────────────────────────
        cx_c, cy_c = self.W * 0.5, self.H * 0.5
        dx2 = px - cx_c
        dy2 = py - cy_c
        r_  = np.sqrt(dx2 * dx2 + dy2 * dy2) + 1.0
        orbital_str = GRAVITY_DRIFT * (2.5 if self.mode == "galaxy" else 1.0)
        ax += (-dy2 / r_) * orbital_str
        ay += ( dx2 / r_) * orbital_str

        # ── 6. Ripple push + ring expiry ──────────────────────────────────────
        alive_rr: List[_RippleEntry] = []
        for rr in self._ripple_rings:
            age_r = self._t - rr["born"]
            if age_r >= RIPPLE_LIFETIME:
                continue
            alive_rr.append(rr)
            rr["r"] = min(age_r * RIPPLE_SPEED, rr["max_r"])
            if rr["r"] >= rr["max_r"]:
                continue
            ddx = px - rr["cx"]
            ddy = py - rr["cy"]
            r_r = np.sqrt(ddx * ddx + ddy * ddy) + 1e-6
            w   = 40.0
            in_ring = np.abs(r_r - rr["r"]) < w
            if np.any(in_ring):
                fade = 1.0 - age_r / RIPPLE_LIFETIME
                ax[in_ring] += (ddx[in_ring] / r_r[in_ring]) * 180.0 * fade
                ay[in_ring] += (ddy[in_ring] / r_r[in_ring]) * 180.0 * fade
        self._ripple_rings = alive_rr

        # ── 7. Integrate velocity ─────────────────────────────────────────────
        depth_scale = (0.45 + self.depth * 0.55).astype(np.float64)
        vx = (vx.astype(np.float64) + ax * depth_scale) * DAMPING
        vy = (vy.astype(np.float64) + ay * depth_scale) * DAMPING

        # Clamp to maximum speed
        spd  = np.sqrt(vx * vx + vy * vy) + 1e-9
        over = spd > MAX_SPEED
        if np.any(over):
            scale = MAX_SPEED / spd[over]
            vx[over] *= scale
            vy[over] *= scale

        px = (px.astype(np.float64) + vx)
        py = (py.astype(np.float64) + vy)

        # ── 8. Boundary wrap ──────────────────────────────────────────────────
        np.mod(px, self.W, out=px)
        np.mod(py, self.H, out=py)

        # ── 9. Trail snapshot ─────────────────────────────────────────────────
        head = self._trail_head % TRAIL_LENGTH
        self.trail_px[head] = px.astype(np.float32)
        self.trail_py[head] = py.astype(np.float32)
        self._trail_head += 1
        self._frames_run += 1

        # ── 10. Age, shimmer, colour temperature ──────────────────────────────
        self.age = (self.age + dt * 0.12) % 1.0

        # Hue shimmer driven by depth (no dead conditional branch)
        self.hue_off = (self.hue_off + self.depth * 0.6) % 360.0

        # Colour temperature: warm for fast/explode modes, cool otherwise
        target_temp = np.float32(0.8 if self.mode == "explode" else 0.1)
        speed_warm  = np.clip(spd / MAX_SPEED, 0, 1).astype(np.float32)
        self.colour_temp = (
            self.colour_temp * np.float32(0.95)
            + (speed_warm * target_temp) * np.float32(0.05)
        )

        # Commit back to float32 state
        self.px = px.astype(np.float32)
        self.py = py.astype(np.float32)
        self.vx = vx.astype(np.float32)
        self.vy = vy.astype(np.float32)

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.px, self.py

    def get_speeds(self) -> np.ndarray:
        return np.sqrt(self.vx ** 2 + self.vy ** 2)

    def get_trail(self, age: int) -> Tuple[np.ndarray, np.ndarray]:
        age = min(age, TRAIL_LENGTH - 1)
        if age >= self._frames_run:
            return self.px, self.py
        idx = (self._trail_head - 1 - age) % TRAIL_LENGTH
        return self.trail_px[idx], self.trail_py[idx]

    def get_ripple_rings(self) -> List[_RippleEntry]:
        return self._ripple_rings
