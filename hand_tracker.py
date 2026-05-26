"""
hand_tracker.py
---------------
NexusFlow — Webcam capture + MediaPipe hand landmark detection.
Compatible with Python 3.8+ and MediaPipe 0.10+ (Tasks API).

Developed by sobing4413 — Organization: Exter Interactive
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request
from collections import Counter, deque
from typing import List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)


# ─── Model ────────────────────────────────────────────────────────────────────
MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def _ensure_model(model_path: str) -> str:
    """Download the MediaPipe model if not present or corrupt."""
    if os.path.exists(model_path) and os.path.getsize(model_path) > 10_000:
        return model_path

    print(f"[HandTracker] Downloading model → {model_path}")
    print("[HandTracker] (~4 MB, one-time download)")

    def _reporthook(count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        pct = min(100, int(count * block_size * 100 / total_size))
        bar = "#" * (pct // 4)
        sys.stdout.write(f"\r  [{bar:<25}] {pct}%")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(MODEL_URL, model_path, _reporthook)
        print("\n[HandTracker] Model downloaded successfully.")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download hand model: {exc}\n"
            f"Manual download: {MODEL_URL}\n"
            f"Place at: {model_path}"
        ) from exc
    return model_path


# ─── Gesture constants ────────────────────────────────────────────────────────
GESTURE_NONE     = "none"
GESTURE_OPEN     = "open"       # All fingers up      → explode burst
GESTURE_FIST     = "fist"       # All fingers down    → gather + text
GESTURE_PEACE    = "peace"      # Index + middle      → heart shape
GESTURE_PINCH    = "pinch"      # Thumb + index tip   → vortex
GESTURE_POINT    = "point"      # Only index up       → text formation
GESTURE_THUMB_UP = "thumbup"    # Thumb up, rest down → ripple wave
GESTURE_SPIDER   = "spider"     # Thumb + index + pinky up → galaxy
GESTURE_THREE    = "three"      # Index + middle + ring → triangle
GESTURE_WAVE     = "wave"       # Fast horizontal movement → shockwave

ALL_GESTURES = [
    GESTURE_NONE, GESTURE_OPEN, GESTURE_FIST, GESTURE_PEACE,
    GESTURE_PINCH, GESTURE_POINT, GESTURE_THUMB_UP, GESTURE_SPIDER,
    GESTURE_THREE, GESTURE_WAVE,
]

# Landmark indices (MediaPipe hand model)
_WRIST      = 0
_THUMB_TIP  = 4
_THUMB_IP   = 3
_THUMB_MCP  = 2
_INDEX_TIP  = 8
_INDEX_PIP  = 6
_INDEX_MCP  = 5
_MIDDLE_TIP = 12
_MIDDLE_PIP = 10
_MIDDLE_MCP = 9
_RING_TIP   = 16
_RING_MCP   = 13
_PINKY_TIP  = 20
_PINKY_MCP  = 17

FINGERTIP_IDS = [_THUMB_TIP, _INDEX_TIP, _MIDDLE_TIP, _RING_TIP, _PINKY_TIP]


class HandTracker:
    """
    MediaPipe HandLandmarker wrapper with:
    - 9 gesture classifiers with priority ordering
    - Gesture hold-time tracking for charge-up effects
    - Palm velocity (screen-space px/s) with exponential smoothing
    - EMA landmark smoothing
    - Robust camera error handling
    - Proper resource cleanup via context manager protocol
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        min_detection_confidence: float = 0.70,
        min_tracking_confidence:  float = 0.55,
        smooth_landmarks: bool = True,
        model_path: Optional[str] = None,
        smooth_n: int = 4,
        ema_alpha: float = 0.72,
    ):
        # ── Camera ────────────────────────────────────────────────────────────
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {camera_index}. "
                "Ensure a webcam is connected and not in use by another app."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # minimise latency
        self.cam_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ── Model ─────────────────────────────────────────────────────────────
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME
            )
        _ensure_model(model_path)

        # ── MediaPipe ─────────────────────────────────────────────────────────
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = HandLandmarker.create_from_options(options)

        # VIDEO mode requires a strictly increasing timestamp in milliseconds
        self._ts_ms: int = 0

        # ── Smoothing ─────────────────────────────────────────────────────────
        self._smooth_landmarks = smooth_landmarks
        self._lm_smooth: Optional[List[Tuple[float, float]]] = None
        self._ema_alpha = float(np.clip(ema_alpha, 0.01, 0.99))

        # ── Public state ──────────────────────────────────────────────────────
        self.landmarks:   Optional[List[Tuple[float, float]]] = None
        self.gesture:     str   = GESTURE_NONE
        self.wrist_pos:   Optional[Tuple[float, float]] = None
        self.palm_center: Optional[Tuple[float, float]] = None

        # Velocity & movement (screen-space px/s)
        self._palm_history: deque = deque(maxlen=8)
        self.palm_velocity: Tuple[float, float] = (0.0, 0.0)
        self.palm_speed:    float = 0.0
        self.movement_dir:  Tuple[float, float] = (0.0, 0.0)

        # Finger state (extended booleans: thumb, index, middle, ring, pinky)
        self.finger_states: List[bool] = [False] * 5

        # Gesture hold time
        self._gesture_start_time: float = time.perf_counter()
        self.gesture_hold_time:   float = 0.0

        # Gesture smoothing buffer (majority-vote)
        self._smooth_n       = max(1, smooth_n)
        self._gesture_buffer = [GESTURE_NONE] * self._smooth_n
        self._buf_idx        = 0
        self._last_frame_time: float = time.perf_counter()

        # Consecutive read-failure counter
        self._read_failures = 0
        self._MAX_FAILURES  = 10

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *_) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self) -> Optional[np.ndarray]:
        """
        Read one camera frame, run hand detection, update all public state.

        Returns:
            Mirrored BGR frame (numpy array) if capture succeeded, else None.
        Raises:
            RuntimeError if the camera fails too many consecutive times.
        """
        now = time.perf_counter()
        dt  = max(now - self._last_frame_time, 1e-6)
        self._last_frame_time = now

        ok, frame = self.cap.read()
        if not ok:
            self._read_failures += 1
            if self._read_failures >= self._MAX_FAILURES:
                raise RuntimeError(
                    "Camera stopped delivering frames after "
                    f"{self._MAX_FAILURES} consecutive failures."
                )
            # Propagate empty state
            self._clear_hand_state(now)
            return None

        self._read_failures = 0

        # Mirror horizontally so left/right feel natural
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # Strictly-increasing timestamp required by VIDEO mode
        self._ts_ms += max(1, int(dt * 1000))

        result: HandLandmarkerResult = self._detector.detect_for_video(
            mp_image, self._ts_ms
        )

        if result.hand_landmarks:
            raw_lm: List[Tuple[float, float]] = [
                (lm.x, lm.y) for lm in result.hand_landmarks[0]
            ]
            self.landmarks = self._apply_ema(raw_lm)
            self.wrist_pos   = self.landmarks[_WRIST]
            self.palm_center = self._compute_palm_center()
            self._update_velocity(dt)
            self._update_finger_states()

            raw_g = self._classify()

            # WAVE override: fast motion beats static classification
            if self.palm_speed > 340 and raw_g in (GESTURE_NONE, GESTURE_OPEN):
                raw_g = GESTURE_WAVE

            self._commit_gesture(self._smooth_gesture(raw_g), now)
        else:
            self._clear_hand_state(now)

        return frame

    def screen_palm_center(
        self, screen_w: int, screen_h: int
    ) -> Optional[Tuple[float, float]]:
        """Convert normalised palm centre to screen-space coordinates."""
        if self.palm_center is None:
            return None
        return (
            self.palm_center[0] * screen_w,
            self.palm_center[1] * screen_h,
        )

    def screen_fingertips(
        self, screen_w: int, screen_h: int
    ) -> Optional[List[Tuple[float, float]]]:
        """Return all 5 fingertip positions in screen-space, or None."""
        if self.landmarks is None:
            return None
        return [
            (self.landmarks[tip][0] * screen_w,
             self.landmarks[tip][1] * screen_h)
            for tip in FINGERTIP_IDS
        ]

    def release(self) -> None:
        """Release camera and MediaPipe detector resources."""
        try:
            if self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        try:
            self._detector.close()
        except Exception:
            pass

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _apply_ema(
        self, raw_lm: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Exponential moving average smoothing over landmark positions."""
        if not self._smooth_landmarks:
            return raw_lm
        if self._lm_smooth is None or len(self._lm_smooth) != len(raw_lm):
            self._lm_smooth = raw_lm
            return raw_lm
        a = self._ema_alpha
        b = 1.0 - a
        self._lm_smooth = [
            (a * r[0] + b * s[0], a * r[1] + b * s[1])
            for r, s in zip(raw_lm, self._lm_smooth)
        ]
        return self._lm_smooth

    def _compute_palm_center(self) -> Tuple[float, float]:
        idxs = [_WRIST, _INDEX_MCP, _MIDDLE_MCP, _RING_MCP, _PINKY_MCP]
        xs = [self.landmarks[i][0] for i in idxs]  # type: ignore[index]
        ys = [self.landmarks[i][1] for i in idxs]  # type: ignore[index]
        return (float(np.mean(xs)), float(np.mean(ys)))

    def _update_velocity(self, dt: float) -> None:
        if self.palm_center is None:
            return
        # Scale to stable pixel-magnitude units
        px = self.palm_center[0] * 1000.0
        py = self.palm_center[1] * 1000.0
        self._palm_history.append((px, py, time.perf_counter()))

        if len(self._palm_history) >= 2:
            x0, y0, t0 = self._palm_history[0]
            x1, y1, t1 = self._palm_history[-1]
            span = max(t1 - t0, 1e-6)
            vx = (x1 - x0) / span * 0.64
            vy = (y1 - y0) / span * 0.48
            self.palm_velocity = (vx, vy)
            spd = (vx ** 2 + vy ** 2) ** 0.5
            self.palm_speed = spd
            if spd > 1e-3:
                self.movement_dir = (vx / spd, vy / spd)

    def _update_finger_states(self) -> None:
        self.finger_states = [
            self._thumb_extended(),
            self._finger_extended(_INDEX_TIP,  _INDEX_MCP),
            self._finger_extended(_MIDDLE_TIP, _MIDDLE_MCP),
            self._finger_extended(_RING_TIP,   _RING_MCP),
            self._finger_extended(_PINKY_TIP,  _PINKY_MCP),
        ]

    def _finger_extended(self, tip: int, mcp: int) -> bool:
        """True if fingertip is further from wrist than MCP (scaled threshold)."""
        lm = self.landmarks
        assert lm is not None
        wx, wy = lm[_WRIST]
        tx, ty = lm[tip]
        mx, my = lm[mcp]
        d_tip = (tx - wx) ** 2 + (ty - wy) ** 2
        d_mcp = (mx - wx) ** 2 + (my - wy) ** 2
        return d_tip > d_mcp * 1.12

    def _thumb_extended(self) -> bool:
        lm = self.landmarks
        assert lm is not None
        tx, ty = lm[_THUMB_TIP]
        ix, iy = lm[_INDEX_MCP]
        wx, wy = lm[_WRIST]
        d_tip  = (tx - wx) ** 2 + (ty - wy) ** 2
        d_base = (ix - wx) ** 2 + (iy - wy) ** 2
        return d_tip > d_base * 0.88

    def _thumb_up_check(self) -> bool:
        """Thumb pointing up; all other four fingers curled."""
        lm = self.landmarks
        if lm is None:
            return False
        ty_tip = lm[_THUMB_TIP][1]
        ty_ip  = lm[_THUMB_IP][1]
        thumb_up = ty_tip < ty_ip - 0.04
        if not thumb_up:
            return False
        return (
            not self._finger_extended(_INDEX_TIP,  _INDEX_MCP)
            and not self._finger_extended(_MIDDLE_TIP, _MIDDLE_MCP)
            and not self._finger_extended(_RING_TIP,   _RING_MCP)
            and not self._finger_extended(_PINKY_TIP,  _PINKY_MCP)
        )

    def _pinch_distance(self) -> float:
        """Normalised thumb-index distance (< 0.32 → pinching)."""
        lm = self.landmarks
        if lm is None:
            return 1.0
        tx, ty = lm[_THUMB_TIP]
        ix, iy = lm[_INDEX_TIP]
        wx, wy = lm[_WRIST]
        mx, my = lm[_MIDDLE_MCP]
        palm   = max(((mx - wx) ** 2 + (my - wy) ** 2) ** 0.5, 1e-6)
        dist   = ((tx - ix) ** 2 + (ty - iy) ** 2) ** 0.5
        return dist / palm

    def _classify(self) -> str:
        """Priority-ordered gesture classifier returning a GESTURE_* constant."""
        lm = self.landmarks
        if lm is None:
            return GESTURE_NONE

        index  = self._finger_extended(_INDEX_TIP,  _INDEX_MCP)
        middle = self._finger_extended(_MIDDLE_TIP, _MIDDLE_MCP)
        ring   = self._finger_extended(_RING_TIP,   _RING_MCP)
        pinky  = self._finger_extended(_PINKY_TIP,  _PINKY_MCP)
        thumb  = self._thumb_extended()
        pinch  = self._pinch_distance() < 0.32

        fingers_up = int(index) + int(middle) + int(ring) + int(pinky)

        # Priority order — most specific first
        if self._thumb_up_check():
            return GESTURE_THUMB_UP
        if pinch and not middle:
            return GESTURE_PINCH
        if thumb and index and not middle and not ring and pinky:
            return GESTURE_SPIDER          # spider-man / metal sign
        if fingers_up == 4 and thumb:
            return GESTURE_OPEN
        if fingers_up == 0 and not thumb:
            return GESTURE_FIST
        if index and middle and ring and not pinky:
            return GESTURE_THREE
        if index and middle and not ring and not pinky:
            return GESTURE_PEACE
        if index and not middle and not ring and not pinky:
            return GESTURE_POINT
        return GESTURE_NONE

    def _smooth_gesture(self, raw: str) -> str:
        """Majority-vote over a rolling buffer to reduce flicker."""
        self._gesture_buffer[self._buf_idx % self._smooth_n] = raw
        self._buf_idx += 1
        return Counter(self._gesture_buffer).most_common(1)[0][0]

    def _commit_gesture(self, smoothed: str, now: float) -> None:
        if smoothed != self.gesture:
            self.gesture            = smoothed
            self._gesture_start_time = now
            self.gesture_hold_time  = 0.0
        else:
            self.gesture_hold_time = now - self._gesture_start_time

    def _clear_hand_state(self, now: float) -> None:
        """Reset all hand-dependent state when no hand is detected."""
        self.landmarks    = None
        self._lm_smooth   = None
        self.wrist_pos    = None
        self.palm_center  = None
        self.palm_velocity = (0.0, 0.0)
        self.palm_speed    = 0.0
        self.movement_dir  = (0.0, 0.0)
        self.finger_states = [False] * 5
        self._palm_history.clear()
        smoothed = self._smooth_gesture(GESTURE_NONE)
        self._commit_gesture(smoothed, now)
