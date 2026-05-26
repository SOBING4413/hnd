"""
text_generator.py
-----------------
NexusFlow — Converts text / shapes into (x, y) target point clouds
for particle formation effects.

Developed by sobing4413 — Organization: Exter Interactive
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import pygame


class TextGenerator:
    """
    Pre-renders text and geometric shapes, returning Nx2 float32 arrays
    of screen-space target positions for the particle system.

    All methods cache their results keyed by parameters to avoid
    re-rasterising on repeated calls.
    """

    GESTURE_MESSAGES: Dict[str, str] = {
        "open":    "WELCOME",
        "fist":    "I LOVE YOU",
        "peace":   "HELLO",
        "point":   "HI THERE",
        "pinch":   "AMAZING",
        "thumbup": "AWESOME",
        "spider":  "GALAXY",
        "three":   "POWER",
        "wave":    "WHOOSH",
    }

    _FONT_CANDIDATES = [
        "couriernew", "courier", "lucidaconsole", "consolas",
        "monospace", "dejavusansmono", "ubuntumono",
        "sourcecodepro", "jetbrainsmono",
    ]

    def __init__(self, screen_w: int, screen_h: int) -> None:
        self.screen_w = screen_w
        self.screen_h = screen_h
        self._rasterise_cache: Dict[Tuple, np.ndarray] = {}

        if not pygame.font.get_init():
            pygame.font.init()

        self._fonts: Dict[str, pygame.font.Font] = {
            "xl":     self._load_font(int(screen_h * 0.18)),
            "large":  self._load_font(int(screen_h * 0.13)),
            "medium": self._load_font(int(screen_h * 0.09)),
            "small":  self._load_font(int(screen_h * 0.06)),
        }

    # ── Text targets ──────────────────────────────────────────────────────────

    def get_targets(
        self,
        text:    str,
        size:    str   = "large",
        density: float = 0.6,
        jitter:  float = 2.0,
    ) -> np.ndarray:
        """Return an Nx2 float32 array of target positions for the given text."""
        if not text:
            return np.empty((0, 2), dtype=np.float32)

        font      = self._fonts.get(size, self._fonts["large"])
        cache_key = (text, size)

        if cache_key not in self._rasterise_cache:
            self._rasterise_cache[cache_key] = self._rasterise(font, text)

        raw_pts = self._rasterise_cache[cache_key]
        if len(raw_pts) == 0:
            return np.empty((0, 2), dtype=np.float32)

        n    = max(1, int(len(raw_pts) * max(0.01, float(density))))
        n    = min(n, len(raw_pts))
        idxs = np.random.choice(len(raw_pts), size=n, replace=False)
        pts  = raw_pts[idxs].astype(np.float64)

        if jitter > 0:
            pts += np.random.randn(*pts.shape) * jitter

        return pts.astype(np.float32)

    # ── Geometric formation targets ────────────────────────────────────────────

    def get_heart_targets(self, n: int = 1200) -> np.ndarray:
        """Heart shape centred on screen."""
        n  = max(4, n)
        t  = np.linspace(0, 2 * np.pi, n)
        x  = 16 * np.sin(t) ** 3
        y  = -(13 * np.cos(t) - 5 * np.cos(2 * t)
               - 2 * np.cos(3 * t) - np.cos(4 * t))
        sc = min(self.screen_w, self.screen_h) * 0.023
        cx = self.screen_w  / 2
        cy = self.screen_h  / 2
        return np.column_stack(
            [cx + x * sc, cy + y * sc]
        ).astype(np.float32)

    def get_vortex_targets(self, n: int = 800, turns: int = 5) -> np.ndarray:
        """Double-arm galaxy spiral."""
        half = max(4, n // 2)
        t1   = np.linspace(0, turns * 2 * np.pi, half)
        t2   = t1 + np.pi
        scale = min(self.screen_w, self.screen_h) * 0.36
        r1    = t1 / (turns * 2 * np.pi) * scale
        r2    = r1 * 0.8
        cx    = self.screen_w  / 2
        cy    = self.screen_h  / 2
        x     = np.concatenate([cx + r1 * np.cos(t1), cx + r2 * np.cos(t2)])
        y     = np.concatenate([cy + r1 * np.sin(t1), cy + r2 * np.sin(t2)])
        return np.column_stack([x, y]).astype(np.float32)

    def get_triangle_targets(self, n: int = 600) -> np.ndarray:
        """Equilateral triangle formation."""
        cx  = self.screen_w  / 2
        cy  = self.screen_h  / 2
        R   = min(self.screen_w, self.screen_h) * 0.33
        verts = np.array([
            [cx + R * np.cos(np.pi / 2 + 2 * np.pi * k / 3),
             cy - R * np.sin(np.pi / 2 + 2 * np.pi * k / 3)]
            for k in range(3)
        ])
        pts: list = []
        per_edge = max(1, n // 3)
        for i in range(3):
            a = verts[i]
            b = verts[(i + 1) % 3]
            ts = np.linspace(0, 1, per_edge, endpoint=False)
            pts.append(a + np.outer(ts, b - a))
        return np.vstack(pts).astype(np.float32)

    def get_dna_targets(self, n: int = 900) -> np.ndarray:
        """Double-helix DNA strand with horizontal rungs."""
        half = max(4, n // 2)
        t    = np.linspace(0, 4 * np.pi, half)
        amp  = min(self.screen_w, self.screen_h) * 0.20
        cx   = self.screen_w  / 2
        cy   = self.screen_h  / 2
        y_arr = cy - self.screen_h * 0.4 + (t / (4 * np.pi)) * self.screen_h * 0.8
        x1    = cx + amp * np.cos(t)
        x2    = cx + amp * np.cos(t + np.pi)

        # Horizontal rungs connecting the two strands
        rung_n    = 20
        rung_pts: list = []
        for k in range(rung_n):
            frac = k / rung_n
            idx  = int(frac * (half - 1))
            for q in np.linspace(0.0, 1.0, 8):
                rung_pts.append([x1[idx] + q * (x2[idx] - x1[idx]), y_arr[idx]])

        helix = np.column_stack([
            np.concatenate([x1, x2]),
            np.concatenate([y_arr, y_arr]),
        ])
        rungs = np.array(rung_pts, dtype=np.float32)
        return np.vstack([helix, rungs]).astype(np.float32)

    def get_star_targets(self, n: int = 800, points: int = 5) -> np.ndarray:
        """N-pointed star outline."""
        cx  = self.screen_w  / 2
        cy  = self.screen_h  / 2
        R   = min(self.screen_w, self.screen_h) * 0.32
        r   = R * 0.42

        # Build star vertices (alternating outer/inner radii)
        num_verts = points * 2
        angles    = np.linspace(-np.pi / 2, -np.pi / 2 + 2 * np.pi,
                                num_verts, endpoint=False)
        radii     = np.tile([R, r], points)
        sx        = cx + radii * np.cos(angles)
        sy        = cy + radii * np.sin(angles)

        # Interpolate n evenly spaced points around the perimeter
        pts: list = []
        per_seg = max(1, n // num_verts)
        for i in range(num_verts):
            a  = np.array([sx[i], sy[i]])
            b  = np.array([sx[(i + 1) % num_verts], sy[(i + 1) % num_verts]])
            ts = np.linspace(0, 1, per_seg, endpoint=False)
            pts.append(a + np.outer(ts, b - a))
        return np.vstack(pts).astype(np.float32)

    def get_pentagon_targets(self, n: int = 700) -> np.ndarray:
        """Regular pentagon + inner ring."""
        cx  = self.screen_w  / 2
        cy  = self.screen_h  / 2
        R   = min(self.screen_w, self.screen_h) * 0.30
        r   = R * 0.55
        pts: list = []
        for radius, sides in [(R, 5), (r, 5)]:
            per_edge = max(1, n // (2 * sides))
            for i in range(sides):
                angle_a = -np.pi / 2 + 2 * np.pi * i       / sides
                angle_b = -np.pi / 2 + 2 * np.pi * (i + 1) / sides
                a = np.array([cx + radius * np.cos(angle_a),
                               cy + radius * np.sin(angle_a)])
                b = np.array([cx + radius * np.cos(angle_b),
                               cy + radius * np.sin(angle_b)])
                ts = np.linspace(0, 1, per_edge, endpoint=False)
                pts.append(a + np.outer(ts, b - a))
        return np.vstack(pts).astype(np.float32)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _load_font(self, size: int) -> pygame.font.Font:
        size = max(8, size)
        for name in self._FONT_CANDIDATES:
            try:
                f = pygame.font.SysFont(name, size, bold=True)
                if f is not None:
                    return f
            except Exception:
                continue
        return pygame.font.Font(None, size)

    def _rasterise(
        self,
        font: pygame.font.Font,
        text: str,
    ) -> np.ndarray:
        """Render text white-on-black and return the lit pixel coords as Nx2."""
        if not text:
            return np.empty((0, 2), dtype=np.float32)

        surf  = font.render(text, True, (255, 255, 255), (0, 0, 0))
        arr   = pygame.surfarray.array3d(surf)   # shape (W, H, 3)
        bright = arr[:, :, 0] > 128              # bool mask [W, H]

        xs, ys = np.where(bright)
        if len(xs) == 0:
            return np.empty((0, 2), dtype=np.float32)

        tw, th = surf.get_size()
        ox = (self.screen_w  - tw) // 2
        oy = (self.screen_h  - th) // 2

        return np.column_stack([xs + ox, ys + oy]).astype(np.float32)
