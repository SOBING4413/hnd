"""
renderer.py
-----------
NexusFlow — Pygame renderer, cyberpunk edition.

Developed by sobing4413 — Organization: Exter Interactive
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pygame


# ─── Colour palettes ──────────────────────────────────────────────────────────
PALETTES: Dict[str, Dict[str, Tuple[int, int, int]]] = {
    "CYBER_BLUE": {
        "core":  (0,   180, 255),
        "trail": (0,   80,  180),
        "glow":  (0,   100, 255),
        "text":  (0,   220, 255),
    },
    "NEON_PINK": {
        "core":  (255, 40,  180),
        "trail": (140, 0,   100),
        "glow":  (255, 0,   160),
        "text":  (255, 120, 220),
    },
    "ACID_GREEN": {
        "core":  (0,   255, 80),
        "trail": (0,   120, 30),
        "glow":  (60,  255, 0),
        "text":  (120, 255, 60),
    },
    "SOLAR_GOLD": {
        "core":  (255, 200, 0),
        "trail": (160, 80,  0),
        "glow":  (255, 140, 0),
        "text":  (255, 220, 80),
    },
    "VOID_PURPLE": {
        "core":  (180, 40,  255),
        "trail": (80,  0,   160),
        "glow":  (140, 0,   255),
        "text":  (200, 120, 255),
    },
}
PALETTE_NAMES: List[str] = list(PALETTES.keys())

# Finger metadata (colour, label, landmark id)
_FINGER_INFO: List[Tuple[Tuple[int,int,int], str]] = [
    ((255, 100, 40),  "T"),   # thumb
    ((0,   220, 255), "I"),   # index
    ((0,   255, 120), "M"),   # middle
    ((180, 60,  255), "R"),   # ring
    ((255, 210, 0),   "P"),   # pinky
]

_FINGER_CHAINS = [
    ([(0, 1), (1, 2), (2, 3), (3, 4)],            (255, 100, 40)),  # thumb
    ([(5, 6), (6, 7), (7, 8)],                     (0,   220, 255)),# index
    ([(9, 10), (10, 11), (11, 12)],                (0,   255, 120)),# middle
    ([(13, 14), (14, 15), (15, 16)],               (180, 60,  255)),# ring
    ([(17, 18), (18, 19), (19, 20)],               (255, 210, 0)),  # pinky
    ([(0, 5), (5, 9), (9, 13), (13, 17), (17, 0)],(80,  120, 200)),# palm
]

_GESTURE_COLOURS: Dict[str, Tuple[int,int,int]] = {
    "open":    (0,   255, 180),
    "fist":    (255, 80,  120),
    "peace":   (255, 200, 0),
    "pinch":   (150, 80,  255),
    "point":   (0,   220, 255),
    "thumbup": (255, 160, 0),
    "spider":  (100, 0,   255),
    "three":   (0,   255, 100),
    "wave":    (0,   180, 255),
}


def _best_mono_font(size: int, bold: bool = True) -> pygame.font.Font:
    candidates = [
        "couriernew", "courier", "lucidaconsole", "consolas",
        "monospace", "dejavusansmono", "ubuntumono",
    ]
    for name in candidates:
        try:
            f = pygame.font.SysFont(name, max(8, size), bold=bold)
            if f is not None:
                return f
        except Exception:
            continue
    return pygame.font.Font(None, max(8, size))


class Renderer:
    """
    Full-scene pygame renderer.
    Draw order each frame:
      begin_frame → draw_particles → draw_finger_auras
      → draw_webcam_thumb → draw_hud → draw_message
      → draw_scanlines → end_frame
    """

    TRAIL_STEPS = 8

    def __init__(
        self,
        screen_w:  int,
        screen_h:  int,
        cam_w:     int  = 640,
        cam_h:     int  = 480,
        fullscreen: bool = False,
    ) -> None:
        self.W     = screen_w
        self.H     = screen_h
        self.cam_w = cam_w
        self.cam_h = cam_h

        pygame.init()
        pygame.display.set_caption("★  NEXUSFLOW  |  Exter Interactive  ★")

        flags = pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE
        if fullscreen:
            flags |= pygame.FULLSCREEN
        self.screen = pygame.display.set_mode((screen_w, screen_h), flags)

        pygame.font.init()
        self._rebuild_fonts(screen_w, screen_h)

        self._frame = 0
        self._clock = pygame.time.Clock()

        # Palette
        self._palette_idx  = 0
        self._palette_name = PALETTE_NAMES[0]
        self.palette       = PALETTES[self._palette_name]

        # Glow sprite cache
        self._glow_cache: Dict[Tuple, pygame.Surface] = {}

        # CRT scanlines overlay
        self._scanlines_surf: Optional[pygame.Surface] = None
        self.scanlines_enabled = False

        # Webcam thumbnail dimensions (maintain 4:3 aspect ratio)
        self._thumb_w = max(200, screen_w // 7)
        self._thumb_h = int(self._thumb_w * (cam_h / max(1, cam_w)))

        # Message overlay
        self._msg_text:  str   = ""
        self._msg_sub:   str   = ""
        self._msg_alpha: float = 0.0
        self._msg_timer: float = 0.0

        self._build_scanlines()

    # ── Public API ────────────────────────────────────────────────────────────

    def handle_resize(self, new_w: int, new_h: int) -> None:
        self.W = max(320, new_w)
        self.H = max(240, new_h)
        self.screen = pygame.display.set_mode(
            (self.W, self.H),
            pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE,
        )
        self._rebuild_fonts(self.W, self.H)
        self._glow_cache.clear()
        self._build_scanlines()

    def cycle_palette(self) -> str:
        self._palette_idx  = (self._palette_idx + 1) % len(PALETTE_NAMES)
        self._palette_name = PALETTE_NAMES[self._palette_idx]
        self.palette       = PALETTES[self._palette_name]
        self._glow_cache.clear()
        return self._palette_name

    def toggle_scanlines(self) -> bool:
        self.scanlines_enabled = not self.scanlines_enabled
        return self.scanlines_enabled

    def begin_frame(self) -> None:
        """Motion-blur fade pass — dark nebula tint for persistence effect."""
        fade = pygame.Surface((self.W, self.H))
        fade.set_alpha(195)
        fade.fill((0, 0, 3))
        self.screen.blit(fade, (0, 0))

    def draw_particles(
        self,
        engine,
        gesture:      str,
        charge_level: float = 0.0,
    ) -> None:
        """Render trail passes + core particle layer + glow halos + ripples."""
        px, py = engine.get_positions()
        spd    = engine.get_speeds()
        depth  = engine.depth
        N      = len(px)
        pal    = self.palette

        cr, cg, cb = pal["core"]
        tr, tg, tb = pal["trail"]
        gr, gg, gb = pal["glow"]
        charge_boost = 1.0 + charge_level * 1.8

        # ── Trail passes (oldest → newest, lightest → brightest) ──────────────
        step = max(1, N // 1400)   # adaptive skip for performance
        for age in range(self.TRAIL_STEPS, 0, -1):
            tx, ty  = engine.get_trail(age)
            alpha_f = (1.0 - age / (self.TRAIL_STEPS + 1)) * 0.30
            base_a  = int(255 * alpha_f)

            for i in range(0, N, step * 3):
                a = min(base_a, int(base_a * depth[i]))
                if a < 4:
                    continue
                sp  = spd[i]
                col = (
                    min(255, int(tr + sp * 3)),
                    min(255, int(tg + sp * 2)),
                    min(255, int(tb + depth[i] * 60)),
                )
                pygame.draw.circle(
                    self.screen, col,
                    (int(tx[i]), int(ty[i])),
                    max(1, int(depth[i] * 1.4)),
                )

        # ── Core layer via surfarray (fast pixel-write) ────────────────────────
        ixs = np.clip(px.astype(np.int32), 0, self.W - 1)
        iys = np.clip(py.astype(np.int32), 0, self.H - 1)

        r_arr = np.clip(cr * 0.25 + spd * 3.0 + engine.hue_off * 0.04,
                        0, 255).astype(np.uint8)
        g_arr = np.clip(cg * 0.40 + spd * 2.5,
                        0, 255).astype(np.uint8)
        b_arr = np.clip(cb * 0.60 + depth * 40.0 * charge_boost,
                        0, 255).astype(np.uint8)

        try:
            pxa = pygame.surfarray.pixels3d(self.screen)
            pxa[ixs, iys, 0] = np.maximum(pxa[ixs, iys, 0], r_arr)
            pxa[ixs, iys, 1] = np.maximum(pxa[ixs, iys, 1], g_arr)
            pxa[ixs, iys, 2] = np.maximum(pxa[ixs, iys, 2], b_arr)
            del pxa  # release the surface lock immediately
        except Exception:
            # Fallback: individual draw calls (slower but always works)
            for i in range(0, N, step):
                sz  = max(1, int(depth[i] * 3))
                col = (int(r_arr[i]), int(g_arr[i]), int(b_arr[i]))
                pygame.draw.circle(self.screen, col, (ixs[i], iys[i]), sz)

        # ── Glow halos for fast particles ─────────────────────────────────────
        threshold = max(5.0, 10.0 - charge_level * 4.0)
        fast_idx  = np.where(spd > threshold)[0]
        glow_col  = (gr, gg, gb)
        sample    = fast_idx[::max(1, len(fast_idx) // 300)] if len(fast_idx) else fast_idx
        for i in sample:
            d   = float(depth[i])
            rad = max(3, int(d * 8 * charge_boost))
            glow = self._get_glow(rad, glow_col)
            self.screen.blit(
                glow,
                (int(px[i]) - rad, int(py[i]) - rad),
                special_flags=pygame.BLEND_ADD,
            )

        # ── Ripple ring overlays ───────────────────────────────────────────────
        now_t = getattr(engine, "_t", 0.0)
        for rr in engine.get_ripple_rings():
            age_r  = max(0.0, now_t - rr["born"])
            fade   = max(0.0, 1.0 - age_r / 2.0)
            radius = int(rr["r"])
            if radius < 2 or fade <= 0.0:
                continue
            ring_col = (
                min(255, int(gr * fade)),
                min(255, int(gg * fade)),
                min(255, int(gb * fade)),
            )
            pygame.draw.circle(
                self.screen, ring_col,
                (int(rr["cx"]), int(rr["cy"])), radius, 2,
            )

    def draw_finger_auras(
        self,
        landmarks:    Optional[list],
        screen_w:     int,
        screen_h:     int,
        gesture:      str,
        charge_level: float = 0.0,
    ) -> None:
        """Animated glow circles at each fingertip and the wrist."""
        if not landmarks:
            return

        pal   = self.palette
        boost = 1.0 + charge_level * 1.5
        t     = self._frame * 0.05

        # ── Wrist/palm glow ───────────────────────────────────────────────────
        px_s = int(landmarks[0][0] * screen_w)
        py_s = int(landmarks[0][1] * screen_h)
        palm_r  = int(22 * boost + 6 * math.sin(t))
        palm_col = pal["core"]
        gs_size  = palm_r * 4
        glow_surf = pygame.Surface((gs_size, gs_size), pygame.SRCALPHA)
        for ri in range(palm_r, 0, -3):
            alpha = int(60 * (ri / max(1, palm_r)) ** 2)
            pygame.draw.circle(
                glow_surf, (*palm_col, alpha),
                (palm_r * 2, palm_r * 2), ri,
            )
        self.screen.blit(
            glow_surf,
            (px_s - palm_r * 2, py_s - palm_r * 2),
            special_flags=pygame.BLEND_ADD,
        )

        # ── Fingertip auras ───────────────────────────────────────────────────
        tip_ids = [4, 8, 12, 16, 20]
        for i, tip_id in enumerate(tip_ids):
            fx  = int(landmarks[tip_id][0] * screen_w)
            fy  = int(landmarks[tip_id][1] * screen_h)
            col = _FINGER_INFO[i][0]
            r_base  = int(14 * boost)
            r_pulse = r_base + int(5 * math.sin(t + i * 1.2))
            ss      = max(4, r_pulse * 6)
            gs2     = pygame.Surface((ss, ss), pygame.SRCALPHA)
            for ri in range(r_pulse * 2, 0, -2):
                alpha = int(80 * (ri / max(1, r_pulse * 2)) ** 2)
                pygame.draw.circle(gs2, (*col, alpha), (ss // 2, ss // 2), ri)
            self.screen.blit(
                gs2,
                (fx - ss // 2, fy - ss // 2),
                special_flags=pygame.BLEND_ADD,
            )
            pygame.draw.circle(self.screen, col,           (fx, fy), max(3, r_pulse // 3))
            pygame.draw.circle(self.screen, (255, 255, 255),(fx, fy), max(1, r_pulse // 6))

        # ── Full-screen skeleton lines ─────────────────────────────────────────
        for chain, col in _FINGER_CHAINS:
            for a_id, b_id in chain:
                ax_s = int(landmarks[a_id][0] * screen_w)
                ay_s = int(landmarks[a_id][1] * screen_h)
                bx_s = int(landmarks[b_id][0] * screen_w)
                by_s = int(landmarks[b_id][1] * screen_h)
                dim  = (col[0] // 3, col[1] // 3, col[2] // 3)
                pygame.draw.line(self.screen, dim, (ax_s, ay_s), (bx_s, by_s), 1)

    def draw_webcam_thumb(
        self,
        bgr_frame: Optional[np.ndarray],
        landmarks: Optional[list] = None,
        cam_w:     int = 640,
        cam_h:     int = 480,
    ) -> None:
        """Render a mirrored webcam thumbnail with hand skeleton overlay."""
        if bgr_frame is None:
            return

        tw, th = self._thumb_w, self._thumb_h

        try:
            rgb_small = cv2.resize(
                cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB),
                (tw, th),
                interpolation=cv2.INTER_LINEAR,
            )
        except cv2.error:
            return

        if landmarks is not None and len(landmarks) == 21:
            sx = tw / max(1, cam_w)
            sy = th / max(1, cam_h)

            def lm_pt(idx: int) -> Tuple[int, int]:
                return (
                    int(landmarks[idx][0] * cam_w * sx),
                    int(landmarks[idx][1] * cam_h * sy),
                )

            FINGER_CHAINS_CAM = {
                "thumb":  ([(0,1),(1,2),(2,3),(3,4)],   (255,100,50)),
                "index":  ([(0,5),(5,6),(6,7),(7,8)],   (0,220,255)),
                "middle": ([(0,9),(9,10),(10,11),(11,12)],(0,255,120)),
                "ring":   ([(0,13),(13,14),(14,15),(15,16)],(200,80,255)),
                "pinky":  ([(0,17),(17,18),(18,19),(19,20)],(255,200,0)),
                "palm":   ([(5,9),(9,13),(13,17)],       (100,180,255)),
            }
            FINGERTIPS = {4, 8, 12, 16, 20}
            for _, (chain, col) in FINGER_CHAINS_CAM.items():
                for a, b in chain:
                    cv2.line(rgb_small, lm_pt(a), lm_pt(b), col, 1)
            for idx, lm in enumerate(landmarks):
                pt  = lm_pt(idx)
                col = _FINGER_INFO[min(idx // 4, 4)][0] if idx > 0 else (100,180,255)
                r   = 3 if idx in FINGERTIPS else 2
                cv2.circle(rgb_small, pt, r + 1, (0, 0, 0), -1)
                cv2.circle(rgb_small, pt, r, col, -1)

        thumb = pygame.surfarray.make_surface(
            np.transpose(rgb_small, (1, 0, 2))
        )
        thumb.set_alpha(200)

        cr, cg, cb = self.palette["core"]
        bx = self.W - tw - 14
        by = 14

        # Glowing border
        pygame.draw.rect(self.screen, (cr // 4, cg // 4, cb // 4),
                         (bx - 4, by - 4, tw + 8, th + 8))
        pygame.draw.rect(self.screen, (cr, cg, cb),
                         (bx - 2, by - 2, tw + 4, th + 4), 2)
        pygame.draw.rect(self.screen,
                         (min(255, cr + 80), min(255, cg + 80), min(255, cb + 80)),
                         (bx - 1, by - 1, tw + 2, th + 2), 1)
        self.screen.blit(thumb, (bx, by))

        lbl = self._font_small.render("▶ LIVE CAM", True, (cr, cg, cb))
        self.screen.blit(lbl, (bx, by + th + 4))

    def draw_hud(
        self,
        gesture:       str,
        fps:           float,
        mode:          str,
        charge_level:  float = 0.0,
        gesture_hold:  float = 0.0,
        palette_name:  str   = "",
        finger_states: Optional[List[bool]] = None,
    ) -> None:
        """Render the heads-up display: FPS, gesture, mode, charge bar, finger dots."""
        tc      = self.palette["text"]
        g_col   = _GESTURE_COLOURS.get(gesture, (80, 80, 120))
        fh      = self._font_hud.get_height()
        x_base  = 14

        # FPS counter
        fps_s = self._font_hud.render(f"FPS {fps:.0f}", True, (0, 200, 100))
        self.screen.blit(fps_s, (x_base, 14))

        # Gesture label
        g_s = self._font_hud.render(f"[{gesture.upper()}]", True, g_col)
        self.screen.blit(g_s, (x_base, 14 + fh + 4))

        # Mode label
        y2  = 14 + (fh + 4) * 2
        m_s = self._font_hud.render(
            f"MODE: {mode[:20].upper()}", True, (0, 120, 200)
        )
        self.screen.blit(m_s, (x_base, y2))

        # Hold timer
        if gesture != "none" and gesture_hold > 0.1:
            hold_s = self._font_small.render(
                f"HOLD {gesture_hold:.1f}s", True, g_col
            )
            self.screen.blit(hold_s, (x_base, y2 + fh + 4))

        # Palette name (bottom-left)
        if palette_name:
            p_s = self._font_small.render(
                f"PALETTE: {palette_name}", True, (60, 60, 100)
            )
            self.screen.blit(p_s, (x_base, self.H - p_s.get_height() - 10))

        # Charge bar
        if charge_level > 0.01:
            self._draw_charge_bar(charge_level, g_col)

        # Finger status dots
        states = finger_states if finger_states else [False] * 5
        dot_r  = max(6, self.H // 80)
        dot_y  = self.H - 60
        for fi, (col, label) in enumerate(_FINGER_INFO):
            active   = bool(states[fi]) if fi < len(states) else False
            draw_col = col if active else (col[0] // 5, col[1] // 5, col[2] // 5)
            dot_x    = x_base + fi * (dot_r * 2 + 5)
            pygame.draw.circle(self.screen, draw_col, (dot_x, dot_y), dot_r)
            if active:
                pygame.draw.circle(self.screen, (255, 255, 255),
                                   (dot_x, dot_y), dot_r, 1)
            lbl_f = self._font_small.render(
                label, True, draw_col if active else (40, 40, 60)
            )
            self.screen.blit(
                lbl_f,
                (dot_x - lbl_f.get_width() // 2, dot_y + dot_r + 2),
            )

        # Corner decorations and hint bar
        self._draw_corner_deco()

        hints = "Q=QUIT  R=RESET  F=FULL  P=PALETTE  S=SCAN  1-9=GESTURE"
        h_s   = self._font_small.render(hints, True, (40, 40, 70))
        self.screen.blit(h_s, (self.W - h_s.get_width() - 10,
                                self.H - h_s.get_height() - 10))

        # Credit
        credit_s = self._font_small.render(
            "NexusFlow  |  Exter Interactive", True, (30, 30, 55)
        )
        self.screen.blit(credit_s, (self.W // 2 - credit_s.get_width() // 2,
                                     self.H - credit_s.get_height() - 10))

    def show_message(
        self, text: str, duration: float = 3.5, sub: str = ""
    ) -> None:
        """Queue a large animated text overlay."""
        self._msg_text  = text
        self._msg_sub   = sub
        self._msg_alpha = 255.0
        self._msg_timer = max(0.1, float(duration))

    def draw_message(self, dt: float) -> None:
        """Tick and render the active message overlay (fade-out)."""
        if self._msg_timer <= 0.0:
            return
        self._msg_timer -= dt
        if self._msg_timer < 1.0:
            self._msg_alpha = max(0.0, self._msg_timer * 255.0)
        else:
            self._msg_alpha = 255.0

        alpha = int(self._msg_alpha)
        if alpha <= 0:
            self._msg_timer = 0.0
            return

        tc = self.palette["text"]

        # Glow shadow passes
        for spread in (28, 18, 9):
            gs = self._font_large.render(self._msg_text, True, (0, 50, 160))
            gw, gh = gs.get_size()
            gs.set_alpha(int(alpha * 0.25))
            self.screen.blit(gs, (
                self.W // 2 - gw // 2 + spread,
                self.H // 2 - gh // 2 + spread,
            ))

        # Main text
        ts = self._font_large.render(self._msg_text, True, tc)
        tw, th = ts.get_size()
        ts.set_alpha(alpha)
        self.screen.blit(ts, (self.W // 2 - tw // 2, self.H // 2 - th // 2))

        # Sub-text
        if self._msg_sub:
            ss = self._font_small.render(self._msg_sub, True, tc)
            sw, sh = ss.get_size()
            ss.set_alpha(int(alpha * 0.7))
            self.screen.blit(ss, (
                self.W // 2 - sw // 2,
                self.H // 2 + th // 2 + 10,
            ))

    def draw_scanlines(self) -> None:
        if self.scanlines_enabled and self._scanlines_surf is not None:
            self.screen.blit(
                self._scanlines_surf, (0, 0),
                special_flags=pygame.BLEND_MULT,
            )

    def end_frame(self) -> None:
        pygame.display.flip()
        self._clock.tick(60)
        self._frame += 1

    def quit(self) -> None:
        pygame.quit()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _rebuild_fonts(self, w: int, h: int) -> None:
        self._font_hud   = _best_mono_font(max(14, h // 44), bold=True)
        self._font_large = _best_mono_font(max(36, h // 14), bold=True)
        self._font_small = _best_mono_font(max(12, h // 58), bold=False)

    def _draw_charge_bar(
        self, level: float, colour: Tuple[int, int, int]
    ) -> None:
        bar_w  = max(80, self.W // 6)
        bar_h  = 10
        bx     = 14
        by     = self.H - 34
        filled = int(bar_w * min(1.0, max(0.0, level)))
        pygame.draw.rect(self.screen, (20, 20, 40), (bx, by, bar_w, bar_h))
        pygame.draw.rect(self.screen, colour,       (bx, by, filled, bar_h))
        pygame.draw.rect(self.screen, colour,       (bx, by, bar_w,  bar_h), 1)
        lbl = self._font_small.render(
            f"CHARGE {int(level * 100)}%", True, colour
        )
        self.screen.blit(lbl, (bx, by - lbl.get_height() - 2))

    def _get_glow(
        self, radius: int, colour: Tuple[int, int, int]
    ) -> pygame.Surface:
        """Return a cached radial glow sprite of the given radius/colour."""
        key = (radius, *colour)
        if key in self._glow_cache:
            return self._glow_cache[key]
        # Hard-limit cache size to prevent unbounded memory growth
        if len(self._glow_cache) > 512:
            self._glow_cache.clear()
        size = max(2, radius) * 2
        surf = pygame.Surface((size, size))
        surf.fill((0, 0, 0))
        cr, cg, cb = colour
        for r in range(radius, 0, -1):
            t   = 1.0 - r / max(1, radius)
            col = (
                min(255, int(cr * t * 0.4)),
                min(255, int(cg * t * 0.4)),
                min(255, int(cb * t * 0.6)),
            )
            pygame.draw.circle(surf, col, (radius, radius), r)
        self._glow_cache[key] = surf
        return surf

    def _draw_corner_deco(self) -> None:
        cr, cg, cb = self.palette["core"]
        col  = (cr // 3, cg // 3, cb // 3)
        size = max(16, self.W // 70)
        W, H = self.W, self.H
        for pts in [
            [(0, size), (0, 0), (size, 0)],
            [(W - size, 0), (W, 0), (W, size)],
            [(0, H - size), (0, H), (size, H)],
            [(W - size, H), (W, H), (W, H - size)],
        ]:
            pygame.draw.lines(self.screen, col, False, pts, 1)

    def _build_scanlines(self) -> None:
        """Pre-bake a half-alpha scanline overlay for the CRT effect."""
        surf = pygame.Surface((self.W, self.H))
        surf.fill((255, 255, 255))
        for y in range(0, self.H, 2):
            pygame.draw.line(surf, (180, 180, 180), (0, y), (self.W, y))
        self._scanlines_surf = surf
