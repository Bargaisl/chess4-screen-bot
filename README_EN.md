# ♟️ Chess4 Screen Bot

[🇬🇧 English](README_EN.md) | [🇷🇺 Русский](README.md)

**Chess4 Screen Bot** is a next-generation autonomous screen-based chess assistant. It runs transparently on top of any web browser (Chess.com, Lichess, Chess24, etc.) or desktop chess application using a lightweight click-through HUD overlay.

Unlike earlier iterations (`chess2`, `chess3`), architecture v4 completely solves common computer-vision artifacts (1-square grid phase shifts, pawn-vs-bishop confusion, Stockfish pipe crashes) and introduces a biomechanical mouse humanization engine, offline voice commentary, and multi-line evaluation HUD.

---

## 🌟 Key Features

### 1. 🥷 Stealth & Anti-Cheat Humanization
* **Biomechanical Mouse Kinematics** ([`chessvision/mouse.py`](file:///chessvision/mouse.py)):
  * Cursor trajectories are computed using cubic Bézier splines governed by the physiological **minimum-jerk velocity profile (Flash & Hogan)**:
    $$s(t) = 10t^3 - 15t^4 + 6t^5$$
  * Hand micro-tremor (sub-pixel jitter), subtle natural trajectory overshoots, and post-target corrective micro-movements.
  * Gaussian target dispersion within each square and human click-hold intervals (45–95 ms).
* **Dynamic Time Management**:
  * Fast intuitive moves in opening book and forced exchanges (0.3–0.7 s).
  * Gaussian-distributed thinking intervals in complex middlegame positions (1.5–3.5 s), accurately replicating human grandmaster decision-making.

### 2. 🖱️ Dual-Mode Auto-Move: Drag-and-Drop & Click-to-Move
* Toggle button **«Drag» / «2 Clicks»** in the GUI:
  * **«Drag» (Drag-and-Drop, default)**: smoothly grips the piece $\to$ moves across the Bézier trajectory with left button held $\to$ releases on target square. Compatible with **all** chess platforms (including sites where click-to-move is disabled).
  * **«2 Clicks» (Click-to-Move)**: clicks the source piece $\to$ pauses $\to$ clicks the destination square.

### 3. 🗣️ Offline Voice Move Announcer (Windows SAPI TTS)
* The [`chessvision/voice.py`](file:///chessvision/voice.py) module runs via an isolated COM background thread:
  * **Zero latency, zero external libraries, zero internet required** — runs completely offline using Windows native Speech API.
  * Automatically binds to the high-quality Russian voice (**Microsoft Irina Desktop - Russian**) when available.
  * Pronounces chess moves naturally: *«e4»*, *«Nf3»*, *«kingside castling»*, *«Bxd7 check»*.
  * Queue auto-purging: when the opponent plays rapidly, outdated speech is canceled so announcements remain real-time.

### 4. 🏹 MultiPV Analysis & "No Arrows" Audio-Only Mode
* Multi-line depth evaluation selectable via GUI:
  * 🎧 **«No arrows (audio only / window)»** — maximum stealth. The overlay is 100% invisible and transparent (0 arrows, 0 highlights); advice is delivered purely via voice into headphones and logged in the desktop window.
  * 🟢 **«1 Best Arrow»** — primary tactical choice in emerald green (`#00e676`).
  * 🟡 **«2 Arrows»** — primary + golden alternative (`#ffd600`).
  * 🔵 **«3 Arrows»** — primary + secondary + tertiary cyan line (`#00b0ff`).
* Live evaluations and lines displayed in the status bar (e.g. `1. c4 (+0.25) | 2. d4 (+0.18)`).

### 5. 🎚️ Elo Skill Level Selector
* Engine strength presets:
  * `Max (3500+)` — full Stockfish 19 Universal calculation depth.
  * `2500 (GM)` — Grandmaster strength.
  * `2100 (Master)`
  * `1800 (Club player / Candidate Master)`
  * `1500 (Intermediate / Casual)`
* Controls Stockfish `UCI_LimitStrength`, `UCI_Elo`, and `Skill Level` parameters to prevent suspicious 100% engine accuracy profiles.

### 6. 🛡️ 100% Accurate Vision & Auto-Grid Snapping
* **Corner Parity Invariant**: Enforces chess color topology ($a1/h8$ must be dark, $a8/h1$ must be light). Completely eliminates 1-square edge phantom detections.
* **Smart Grid Snapping (`snap_selection`)**: When manually outlining the chessboard with your cursor, the algorithm automatically locates internal 8x8 cell line edges and snaps pixel-perfectly to the board boundary.
* **Silhouette & Geometry Classification**: Re-scaled aspect ratio and crown width checks (`ratio_top`, `rel_h`) prevent pawn vs bishop and queen vs king mix-ups on custom piece sets (Chess.com Neo, Staunton, etc.).
* **Placement Sanity Validator**: Rejects corrupt frame reads (must contain exactly two kings, valid king spacing, legal pawn ranks, and realistic piece counts).

### 7. 🖥️ Multi-Monitor Support & Side Detection
* Seamless multi-display support: select any monitor from the dropdown; the click-through HUD aligns automatically.
* Automatic board orientation detection: accurately determines whether White or Black is on the bottom side, and locks orientation during the game to avoid endgame flip flicker.

---

## ⌨️ Global Hotkeys (Optimized for Laptops)

On modern laptops, top-row function keys (F1–F12) are frequently bound to system controls (volume, brightness) and require pressing `Fn`. **Chess4 Screen Bot** provides ergonomic shortcuts on `Alt + 1..6` and the tilde `~` key that can be triggered comfortably with one hand from any active browser window:

| Laptop Shortcut | Desktop Key | Action | Description |
| :---: | :---: | :--- | :--- |
| **`Alt + 1`** (or **`~`** / **`Alt + Z`**) | `F4` / `F1` | **Start / Pause** | Toggle board tracking and engine calculation. |
| **`Alt + 2`** (or **`Alt + X`**) | `F5` | **Reset** | Clear board state and prepare for a new match. |
| **`Alt + 3`** (or **`Alt + C`**) | `F6` | **Toggle Turn** | Manually switch side to move (White ↔ Black). |
| **`Alt + 4`** (or **`Alt + V`**) | `F7` | **Toggle Side** | Switch player color (White ↔ Black). |
| **`Alt + 5`** (or **`Alt + B`**) | `F2` | **Toggle Overlay** | Hide / show on-screen arrows and squares. |
| **`Alt + 6`** (or **`Alt + S`**) | `F3` | **Force Recalculate** | Immediately re-scan screen and re-evaluate. |
| **`Esc`** | `Esc` | **Panic Hide** | Instantly hides the on-screen overlay. |

---

## 🚀 Installation & Running

### 1. Requirements
* Windows 10 / 11 (64-bit)
* Python 3.10+
* Stockfish binary (e.g. Stockfish 16/17/19 x86-64 executable placed in the project folder as `stockfish.exe`)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

*Dependencies listed in `requirements.txt`:*
```text
numpy>=1.24.0
opencv-python>=4.8.0
Pillow>=9.5.0
mss>=9.0.1
pyautogui>=0.9.54
pywin32>=306
```

### 3. Launch
```bat
start.bat
```
*(or run `python run.py` in your terminal)*

---

## 🎮 Quick Start Guide

1. Open your chess game on any website (Chess.com, Lichess, etc.).
2. Launch the bot using `start.bat`.
3. If playing on a secondary monitor, select it in the **«Monitor»** dropdown.
4. Click **«Select Area»** and drag a rectangular bounding box around the 8x8 board. The bot will automatically snap to the exact grid borders.
5. Click **«Grid»** for a visual check to confirm cell boundaries match your board.
6. Choose your desired mode:
   * **Visual assistance**: play manually by following the on-screen arrows.
   * **Stealth / audio mode**: select **«No arrows»** and turn **«Voice: ON»** to listen to moves through your headphones.
   * **Automove mode**: click **«Automove: ON»** to let the bot naturally drag and drop pieces with humanized Bézier mouse kinematics.

---

## 📂 Project Architecture

```text
chess4-screen-bot/
├── chessvision/
│   ├── app.py            # Tkinter GUI, background threads, and click-through HUD overlay
│   ├── board.py          # 0x88 board representation, move generator, FEN, sanity validator
│   ├── engine.py         # Stockfish UCI driver (MultiPV, Elo strength limiter, auto-recovery)
│   ├── mouse.py          # Biomechanical mouse controller (Bézier splines, Flash-Hogan kinematics, Drag & Drop)
│   ├── templates.py      # Piece silhouette classifier with geometric aspect ratio validation
│   ├── vision.py         # Screen capture (mss), grid search (Canny/Hough), cell extraction
│   └── voice.py          # Windows SAPI offline speech synthesizer with queue auto-purging
├── stockfish/            # Stockfish source files and documentation
├── tests/                # Automated test suite
│   ├── test_app.py
│   ├── test_board.py
│   ├── test_hotkeys_and_hidden_arrows.py
│   ├── test_stealth_multipv_voice.py
│   └── test_user_scenarios.py
├── config.json           # User configuration file
├── requirements.txt      # Python dependencies
├── run.py                # Python entry point
└── start.bat             # One-click Windows launcher
```

---

## 🧪 Testing

To run the complete automated test suite:
```bash
python -m unittest discover -s tests -p "test_*.py"
```
Or execute individual test scripts:
```bash
python tests/test_hotkeys_and_hidden_arrows.py
python tests/test_user_scenarios.py
python tests/test_stealth_multipv_voice.py
python tests/test_app.py
python tests/test_board.py
```

---

## 📄 License
MIT License. Created strictly for educational and research purposes.
