# ★ NexusFlow — Real-time Hand Gesture Particle System

**Developed by sobing4413 — Organization: Exter Interactive**

Real-time hand-gesture particle system using MediaPipe + Pygame.
Production-grade refactor with optimised performance, enhanced stability, and modern code architecture.

---

## Features

- Real-time hand tracking via MediaPipe Tasks API
- Cyberpunk neon particle visual effects with motion trails
- 9 distinct gesture modes with charge-up amplification
- Per-fingertip physics influence and aura rendering
- Shockwave + ripple ring visual effects
- 5 swappable colour palettes (P key)
- CRT scanline overlay (S key)
- Webcam thumbnail with full hand skeleton overlay
- Keyboard-only fallback mode (no webcam required)
- Fullscreen support with dynamic resize handling
- Graceful camera error recovery

---

## Requirements

### Hardware
- Webcam (optional — keyboard-only mode works without one)

### Software
- Windows 10/11 (also works on macOS and Linux)
- Python 3.11 recommended (3.9–3.12 supported)

> ⚠ Do NOT use Python 3.14 — `pygame` may fail to build.

---

## Installation

### 1. Install Python 3.11
[https://www.python.org/downloads/](https://www.python.org/downloads/)

Enable **Add Python to PATH** during installation.

### 2. Navigate to project folder
```bash
cd path/to/NexusFlow
```

### 3. Create & activate virtual environment
```bash
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
```

### 4. Install dependencies
```bash
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

---

## Run

```bash
python main.py
```

On first launch you may customise resolution, particle count, and camera index.
Press **Enter** to accept all defaults.

---

## Gesture Reference

| Gesture              | Effect            | Mode      |
|----------------------|-------------------|-----------|
| ✋ Open hand          | Explosion burst   | EXPLODE   |
| ✊ Fist               | Gravity gather    | GATHER    |
| ✌ Peace sign         | Heart formation   | HEART     |
| 🤏 Pinch             | Galaxy vortex     | VORTEX    |
| ☝ Point              | Text formation    | FORM      |
| 👍 Thumb up          | Ripple waves      | RIPPLE    |
| 🤘 Spider/Rock       | DNA helix         | GALAXY    |
| 3️⃣ Three fingers    | Triangle          | TRIANGLE  |
| ≋ Fast wave          | Shockwave         | WAVE      |

> **Tip:** Hold a gesture to build charge — release for a stronger effect.

---

## Keyboard Controls

| Key       | Action                 |
|-----------|------------------------|
| `1` – `9` | Force-trigger gesture  |
| `R`       | Reset all particles    |
| `F`       | Toggle fullscreen      |
| `P`       | Cycle colour palette   |
| `S`       | Toggle CRT scanlines   |
| `Q` / ESC | Quit                   |

---

## Colour Palettes (cycle with P)

1. CYBER BLUE (default)
2. NEON PINK
3. ACID GREEN
4. SOLAR GOLD
5. VOID PURPLE

---

## Troubleshooting

### pygame install error
You are likely on Python 3.14. Use Python 3.11 instead.

### Camera not found
The app runs in keyboard-only mode automatically. Use keys `1`–`9` to trigger gestures.

### Slow frame rate
Choose a lower particle count at startup (Lite or Balanced).

### MediaPipe model download fails
Manual download:
[https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

Place `hand_landmarker.task` in the same folder as `hand_tracker.py`.

### macOS Apple Silicon
```bash
pip install mediapipe-silicon
```
instead of standard mediapipe.

---

## Project Structure

```
NexusFlow/
├── main.py            — App entry point, main loop, gesture routing
├── hand_tracker.py    — MediaPipe webcam + gesture classification
├── particle_engine.py — NumPy vectorised physics engine
├── renderer.py        — Pygame renderer, palettes, HUD, effects
├── text_generator.py  — Text/shape → particle target point clouds
├── hand_landmarker.task — MediaPipe model (auto-downloaded)
├── requirements.txt
└── README.md
```

---

*NexusFlow — Developed by sobing4413 — Organization: Exter Interactive*
