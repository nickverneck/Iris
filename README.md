# Iris Eye Tracker

A macOS eye-tracking application built with Rust and Python (MediaPipe).

## Prerequisites
- Rust (Cargo)
- Python 3
- Webcam

## Setup

1. **Initialize Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt # (Or manually: pip install mediapipe opencv-python)
   ```
   *(Note: The setup step has already been performed automatically)*

2. **Build**:
   ```bash
   cargo build --release
   ```

## Running

```bash
cargo run --release
```

## Permissions (macOS)
On the first run, you may be prompted for:
- **Camera Access**: Required for eye tracking.
- **Accessibility Access**: Required for controlling the mouse cursor.
    - If the mouse doesn't move, go to `System Settings > Privacy & Security > Accessibility` and add your Terminal (e.g., iTerm, Terminal, VSCode).

## Customization
- Adjust smoothing in `src/main.rs` (variable `alpha`).
- Adjust gaze sensitivity in `tracking/eye_tracker.py` (`x_min`, `x_max`, etc.).
