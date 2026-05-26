"""
main.py
-------
NexusFlow — Real-time hand-gesture particle system.
Entry point: startup config, main loop, gesture routing.

Developed by sobing4413 — Organization: Exter Interactive

Keyboard controls:
  Q / ESC   – quit
  R         – reset particles
  1 – 9     – force gesture (bypass camera)
  F         – toggle fullscreen
  P         – cycle colour palette
  S         – toggle CRT scanlines
"""

from __future__ import annotations

import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import pygame

from hand_tracker import (
    HandTracker,
    GESTURE_OPEN, GESTURE_FIST, GESTURE_PEACE, GESTURE_PINCH,
    GESTURE_POINT, GESTURE_NONE, GESTURE_THUMB_UP, GESTURE_SPIDER,
    GESTURE_THREE, GESTURE_WAVE,
)
from particle_engine import ParticleEngine
from text_generator  import TextGenerator
from renderer        import Renderer


# ─── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_SCREEN_W    = 1280
DEFAULT_SCREEN_H    = 720
DEFAULT_CAM_W       = 640
DEFAULT_CAM_H       = 480
DEFAULT_N_PARTICLES = 4500
TARGET_FPS          = 60
GESTURE_COOLDOWN    = 2.5      # seconds before same gesture can re-trigger
CHARGE_HOLD_MIN     = 0.5      # minimum hold time before charge accumulates
MAX_DT              = 0.05     # clamp delta-time to avoid physics explosions
COOLDOWN_EPS        = 1e-6     # float tolerance for cooldown completion


# ─── Startup pickers ─────────────────────────────────────────────────────────

def _pick_resolution() -> Tuple[int, int]:
    presets = [
        (1920, 1080),
        (1280, 720),
        (1024, 576),
        (800,  600),
        (640,  480),
    ]
    print("\n╔══ SCREEN RESOLUTION ════════════════════════════════╗")
    for i, (w, h) in enumerate(presets, 1):
        marker = " ← default" if (w, h) == (1280, 720) else ""
        print(f"║  {i}. {w}×{h}{marker}")
    print("║  6. Custom")
    print("╚═════════════════════════════════════════════════════╝")
    choice = input("Choice [2]: ").strip() or "2"

    if choice == "6":
        try:
            w = int(input("  Width  [1280]: ").strip() or "1280")
            h = int(input("  Height  [720]: ").strip() or  "720")
            return max(400, w), max(300, h)
        except ValueError:
            pass

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(presets):
            return presets[idx]
    except ValueError:
        pass
    return DEFAULT_SCREEN_W, DEFAULT_SCREEN_H


def _pick_particles() -> int:
    print("\n╔══ PARTICLE COUNT ════════════════════════════════════╗")
    print("║  1. 2 000  (lite, any GPU)                          ║")
    print("║  2. 3 500  (balanced)                               ║")
    print("║  3. 4 500  (default, good GPU)        ← default     ║")
    print("║  4. 7 000  (epic, dedicated GPU)                    ║")
    print("║  5. Custom                                          ║")
    print("╚══════════════════════════════════════════════════════╝")
    choice = input("Choice [3]: ").strip() or "3"
    opts   = {1: 2000, 2: 3500, 3: 4500, 4: 7000}
    try:
        idx = int(choice)
        if idx in opts:
            return opts[idx]
        if idx == 5:
            return max(100, int(input("  Count [4500]: ").strip() or "4500"))
    except ValueError:
        pass
    return DEFAULT_N_PARTICLES


def _pick_camera() -> int:
    choice = input("\nCamera index [0]: ").strip() or "0"
    try:
        return max(0, int(choice))
    except ValueError:
        return 0


# ─── App ─────────────────────────────────────────────────────────────────────

class App:
    """
    Top-level application: wires HandTracker → ParticleEngine +
    TextGenerator → Renderer and runs the main loop.
    """

    def __init__(
        self,
        screen_w:     int  = DEFAULT_SCREEN_W,
        screen_h:     int  = DEFAULT_SCREEN_H,
        cam_w:        int  = DEFAULT_CAM_W,
        cam_h:        int  = DEFAULT_CAM_H,
        n_particles:  int  = DEFAULT_N_PARTICLES,
        camera_index: int  = 0,
        fullscreen:   bool = False,
    ) -> None:
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.cam_w    = cam_w
        self.cam_h    = cam_h

        # Renderer must come first (initialises pygame)
        self.renderer = Renderer(screen_w, screen_h, cam_w, cam_h, fullscreen)
        self.engine   = ParticleEngine(screen_w, screen_h, n_particles)
        self.text_gen = TextGenerator(screen_w, screen_h)

        # Camera — degrade gracefully if unavailable
        self.tracker: Optional[HandTracker] = None
        try:
            self.tracker = HandTracker(
                camera_index=camera_index,
                width=cam_w,
                height=cam_h,
            )
            print("[INFO] Camera initialised.")
        except RuntimeError as exc:
            print(f"[WARNING] Camera unavailable: {exc}")
            print("[INFO]    Running in keyboard-only demo mode.")

        # Pre-generate formation targets (expensive, done once at startup)
        self._heart_tgt  = self.text_gen.get_heart_targets(n=1400)
        self._vortex_tgt = self.text_gen.get_vortex_targets(n=900)
        self._tri_tgt    = self.text_gen.get_triangle_targets(n=700)
        self._dna_tgt    = self.text_gen.get_dna_targets(n=900)

        # App state
        self._prev_gesture:    str   = GESTURE_NONE
        self._gesture_cooldown: float = 0.0
        self._last_time:       float = time.perf_counter()
        self._fps_smooth:      float = float(TARGET_FPS)
        self._current_mode:    str   = "free"
        self._running:         bool  = True

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        while self._running:
            now = time.perf_counter()
            dt  = min(now - self._last_time, MAX_DT)
            self._last_time = now

            raw_fps          = 1.0 / max(dt, 1e-9)
            self._fps_smooth = self._fps_smooth * 0.92 + raw_fps * 0.08

            self._process_events()
            if not self._running:
                break

            # ── Hand tracking ─────────────────────────────────────────────────
            frame                = None
            gesture              = GESTURE_NONE
            hand_screen: Optional[Tuple[float, float]] = None
            landmarks            = None
            palm_vel: Tuple[float, float] = (0.0, 0.0)
            hold_time            = 0.0
            finger_states: List[bool] = [False] * 5
            finger_tips_screen: Optional[List[Tuple[float, float]]] = None

            if self.tracker is not None:
                try:
                    frame         = self.tracker.update()
                    gesture       = self.tracker.gesture
                    landmarks     = self.tracker.landmarks
                    palm_vel      = self.tracker.palm_velocity
                    hold_time     = self.tracker.gesture_hold_time
                    finger_states = list(self.tracker.finger_states)

                    hand_screen         = self.tracker.screen_palm_center(
                        self.screen_w, self.screen_h
                    )
                    finger_tips_screen  = self.tracker.screen_fingertips(
                        self.screen_w, self.screen_h
                    )
                except RuntimeError as exc:
                    print(f"[WARNING] Tracker error: {exc}")
                    self.tracker = None

            # ── Gesture routing ───────────────────────────────────────────────
            self._gesture_cooldown = max(0.0, self._gesture_cooldown - dt)
            gesture_changed = gesture != self._prev_gesture
            cooldown_clear  = self._gesture_cooldown <= COOLDOWN_EPS

            if gesture_changed and (gesture == GESTURE_NONE or cooldown_clear):
                self._on_gesture_change(gesture, hand_screen)

            self._prev_gesture = gesture

            # Charge build-up while gesture is held
            if gesture != GESTURE_NONE and hold_time > CHARGE_HOLD_MIN:
                self.engine.charge(dt)
            else:
                self.engine.charge_level = max(
                    0.0, self.engine.charge_level - dt * 0.5
                )

            # ── Physics step ──────────────────────────────────────────────────
            self.engine.update(
                hand_screen, gesture, dt,
                gesture_hold_time=hold_time,
                palm_velocity=palm_vel,
                finger_tips=finger_tips_screen,
            )

            # ── Render ────────────────────────────────────────────────────────
            charge = self.engine.charge_level

            self.renderer.begin_frame()
            self.renderer.draw_particles(self.engine, gesture, charge)

            if landmarks is not None:
                self.renderer.draw_finger_auras(
                    landmarks, self.screen_w, self.screen_h,
                    gesture, charge,
                )

            self.renderer.draw_webcam_thumb(
                frame, landmarks, self.cam_w, self.cam_h
            )
            self.renderer.draw_hud(
                gesture, self._fps_smooth, self._current_mode,
                charge_level=charge,
                gesture_hold=hold_time,
                palette_name=self.renderer._palette_name,
                finger_states=finger_states,
            )
            self.renderer.draw_message(dt)
            self.renderer.draw_scanlines()
            self.renderer.end_frame()

        self._shutdown()

    # ── Gesture logic ─────────────────────────────────────────────────────────

    def _on_gesture_change(
        self,
        gesture:  str,
        hand_pos: Optional[Tuple[float, float]],
    ) -> None:
        if gesture == GESTURE_NONE:
            self.engine.set_mode("free")
            self._current_mode = "free"
            return

        self._gesture_cooldown = GESTURE_COOLDOWN
        charge   = self.engine.release_charge()
        strength = 1.0 + charge * 1.5
        cx = hand_pos[0] if hand_pos else self.screen_w / 2
        cy = hand_pos[1] if hand_pos else self.screen_h / 2

        if gesture == GESTURE_OPEN:
            tgt = self.text_gen.get_targets("WELCOME", size="large", density=0.55)
            self.engine.set_formation(tgt)
            self.engine.set_mode("explode")
            self.engine.trigger_shockwave(cx, cy, strength)
            self.renderer.show_message(
                "WELCOME", sub=f"CHARGE {int(charge * 100)}%", duration=3.0
            )
            self._current_mode = "WELCOME"

        elif gesture == GESTURE_FIST:
            tgt = self.text_gen.get_targets("I LOVE YOU", size="medium", density=0.50)
            self.engine.set_formation(tgt)
            self.renderer.show_message("I LOVE YOU", duration=3.5)
            self._current_mode = "I LOVE YOU"

        elif gesture == GESTURE_PEACE:
            self.engine.set_formation_direct(self._heart_tgt, mode="heart")
            self.engine.add_ripple(cx, cy, max_r=400.0)
            self.renderer.show_message("HELLO ♡", duration=3.2)
            self._current_mode = "HEART"

        elif gesture == GESTURE_PINCH:
            self.engine.set_formation(self._vortex_tgt)
            self.engine.vortex_angle = 0.0
            self.engine.trigger_shockwave(cx, cy, strength * 0.6)
            self.renderer.show_message("AMAZING", duration=2.5)
            self._current_mode = "VORTEX"

        elif gesture == GESTURE_POINT:
            tgt = self.text_gen.get_targets("HELLO", size="large", density=0.60)
            self.engine.set_formation(tgt)
            self.renderer.show_message("HI THERE", duration=3.0)
            self._current_mode = "HELLO"

        elif gesture == GESTURE_THUMB_UP:
            self.engine.set_mode("free")
            for k in range(3):
                self.engine.add_ripple(cx, cy - k * 60, max_r=350.0)
            tgt = self.text_gen.get_targets("AWESOME", size="large", density=0.55)
            self.engine.set_formation(tgt)
            self.renderer.show_message("AWESOME ★", duration=3.0)
            self._current_mode = "AWESOME"

        elif gesture == GESTURE_SPIDER:
            self.engine.set_formation_direct(self._dna_tgt, mode="galaxy")
            self.engine.trigger_shockwave(cx, cy, strength)
            self.renderer.show_message("GALAXY", duration=3.0)
            self._current_mode = "GALAXY"

        elif gesture == GESTURE_THREE:
            self.engine.set_formation_direct(self._tri_tgt, mode="form")
            self.renderer.show_message("POWER ▲", duration=2.8)
            self._current_mode = "TRIANGLE"

        elif gesture == GESTURE_WAVE:
            self.engine.set_mode("free")
            for k in range(5):
                self.engine.add_ripple(cx, cy, max_r=float(220 + k * 80))
            self.engine.trigger_shockwave(cx, cy, strength * 1.2)
            self.renderer.show_message("WHOOSH ≋", duration=2.0)
            self._current_mode = "WAVE"

    # ── Event handling ────────────────────────────────────────────────────────

    def _process_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN:
                self._running = self._handle_key(event.key)
            elif event.type == pygame.VIDEORESIZE:
                self.renderer.handle_resize(event.w, event.h)
                self.screen_w = event.w
                self.screen_h = event.h

    def _handle_key(self, key: int) -> bool:
        """Return False to quit."""
        if key in (pygame.K_q, pygame.K_ESCAPE):
            return False

        if key == pygame.K_r:
            self.engine = ParticleEngine(
                self.screen_w, self.screen_h, self.engine._N
            )
            self._current_mode = "free"
            print("[INFO] Particles reset.")

        elif key == pygame.K_f:
            pygame.display.toggle_fullscreen()

        elif key == pygame.K_p:
            name = self.renderer.cycle_palette()
            print(f"[INFO] Palette → {name}")

        elif key == pygame.K_s:
            on = self.renderer.toggle_scanlines()
            print(f"[INFO] Scanlines {'ON' if on else 'OFF'}")

        # Manual gesture triggers (bypass cooldown, useful without camera)
        manual = {
            pygame.K_1: GESTURE_OPEN,
            pygame.K_2: GESTURE_FIST,
            pygame.K_3: GESTURE_PEACE,
            pygame.K_4: GESTURE_PINCH,
            pygame.K_5: GESTURE_POINT,
            pygame.K_6: GESTURE_THUMB_UP,
            pygame.K_7: GESTURE_SPIDER,
            pygame.K_8: GESTURE_THREE,
            pygame.K_9: GESTURE_WAVE,
        }
        if key in manual:
            g  = manual[key]
            cx = self.screen_w / 2.0
            cy = self.screen_h / 2.0
            self._gesture_cooldown = 0.0
            self._prev_gesture     = GESTURE_NONE
            self._on_gesture_change(g, (cx, cy))
            self._prev_gesture = g

        return True

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def _shutdown(self) -> None:
        if self.tracker is not None:
            try:
                self.tracker.release()
            except Exception:
                pass
        self.renderer.quit()


# ─── Entry point ─────────────────────────────────────────────────────────────

BANNER = r"""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║        ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗                  ║
║        ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝                  ║
║        ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗                  ║
║        ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║                  ║
║        ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║                  ║
║        ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝  F L O W        ║
║                                                                      ║
║              Developed by sobing4413                                 ║
║              Organization: Exter Interactive                         ║
╠══════════════════════════════════════════════════════════════════════╣
║  GESTURE → EFFECT                                                    ║
║   ✋ Open hand    → WELCOME    (explosion burst)                     ║
║   ✊ Fist         → I LOVE YOU (gravity gather)                      ║
║   ✌  Peace sign  → HELLO      (heart formation)                     ║
║   🤏 Pinch        → AMAZING    (galaxy vortex)                       ║
║   ☝  Point        → HI THERE  (text formation)                      ║
║   👍 Thumb up     → AWESOME    (ripple waves)                        ║
║   🤘 Spider/Rock  → GALAXY     (DNA helix)                           ║
║   3  Three fingers→ POWER      (triangle)                            ║
║   〜  Fast wave   → WHOOSH     (shockwave)                           ║
╠══════════════════════════════════════════════════════════════════════╣
║  KEYBOARD                                                            ║
║   1–9 = force gesture  |  R = reset  |  F = fullscreen              ║
║   P   = cycle palette  |  S = scanlines  |  Q/ESC = quit            ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def main() -> None:
    print(BANNER)

    do_custom = input("Customise settings? [y/N]: ").strip().lower()
    if do_custom == "y":
        sw, sh = _pick_resolution()
        n_part = _pick_particles()
        cam_i  = _pick_camera()
    else:
        sw, sh = DEFAULT_SCREEN_W, DEFAULT_SCREEN_H
        n_part = DEFAULT_N_PARTICLES
        cam_i  = 0

    print(f"\n[INFO] Window: {sw}×{sh}  |  Particles: {n_part}  |  Camera: {cam_i}")
    print("[INFO] Starting NexusFlow…\n")

    try:
        app = App(
            screen_w=sw, screen_h=sh,
            cam_w=DEFAULT_CAM_W, cam_h=DEFAULT_CAM_H,
            n_particles=n_part,
            camera_index=cam_i,
        )
        app.run()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as exc:
        print(f"\n[FATAL] {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
