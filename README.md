# Reality is Not What It Seems

A live art installation for **London Tech Week 2025** (9–11 June, Olympia London), in collaboration with UAL Creative Computing Institute.

Visitors stand in front of a camera. Their faces are detected, encoded through a real quantum image compression algorithm (QPIXL), and displayed back — pixelated, greyscale, seen through the lens of quantum information. A panel beside each face shows the actual quantum gate parameters that encoded *their specific face* into a quantum state.

The pixelation isn't a glitch. It's what a quantum computer would "see" if it stored your face.

## How It Works

The installation runs a live pipeline from camera to display:

### 1. Capture & Detect
A webcam captures the scene. YuNet, a small neural face detector run through OpenCV, finds faces on a downscaled frame for speed, including tilted, turned and partly covered ones. Coordinates are then mapped back to full resolution.

### 2. Quantum Encoding (QPIXL)
Each detected face is resized to 32×32 grayscale and encoded using **QPIXL** — a quantum image compression algorithm based on [this Nature paper](https://www.nature.com/articles/s41598-022-11024-y) (compressed FRQI). The steps:

1. **Pixel → angle**: Each pixel brightness (rescaled to 1–254) maps to a rotation angle (up to π/2)
2. **Walsh-Hadamard Transform**: Transforms pixel angles into frequency-domain coefficients (similar to how JPEG uses DCT). Natural images concentrate energy in low frequencies, so many coefficients are near zero
3. **Compression**: The smallest 95% of coefficients are zeroed out — fewer non-zero coefficients means fewer quantum gates
4. **Gray code reordering**: Consecutive pixel addresses differ by only 1 bit, minimising CNOT gates needed
5. **Circuit construction**: Builds a quantum circuit with **k+1 qubits** (10 address qubits for 1024 pixel positions + 1 colour qubit encoding brightness via RY rotations). The address qubits start in superposition (H gates), and RY + CX gate sequences "paint" each pixel's brightness onto the colour qubit

For a 32×32 face at 95% compression, this produces an 11-qubit circuit with 52 RY gates at depth ~170.

### 3. Simulate & Decode
The circuit runs on Qiskit Aer's statevector simulator, producing 2048 complex amplitudes. The decoder takes the square root of the 2048 probabilities (the amplitudes), applies `arctan2` to each pair to recover the pixel angle, and rescales the result to 0–255.

### 4. Display
The quantum-reconstructed greyscale face is stamped onto an edge-detected background. A HUD panel next to each face shows the live quantum circuit parameters — the actual `ry(angle, q0)` rotation gates that encoded that person's face.

## Install

Needs macOS on Apple Silicon and [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 and the pinned packages.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
git clone https://github.com/artreadcode/LondonTechWeek2025.git
cd LondonTechWeek2025
uv sync --locked
```

Without uv: `pip install -r requirements.txt` in a Python 3.12 environment, then use `python` in place of `uv run --locked`.

## Running

```bash
# Find the camera: prints the opened index and resolution
uv run --locked reality.py --camera 0 --verbose

# Exhibition mode (fullscreen, no sleep)
caffeinate -di uv run --locked reality.py --camera 0 --fullscreen

# Static pipeline test on a photo with a face (no camera needed)
uv run --locked reality.py --test photo.jpg
```

### Options

| Flag | Description |
|------|-------------|
| `--bg edges-white` | White background with dark edges (default) |
| `--bg edges-black` | Black background with white edges |
| `--bg passthrough` | Raw camera feed as background |
| `--compression N` | QPIXL compression 0–100 (default: 95, lite: 70) |
| `--camera N` | Camera index (default: 0) |
| `--fullscreen` | Start in fullscreen |
| `--lite` | Lightweight mode: 16×16 faces, 640×480 camera |
| `--verbose` | Print FPS and timing info |

### Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `f` | Toggle fullscreen |
| `esc` | Leave fullscreen |
| `o` | Toggle quantum parameter panels |
| `v` | Toggle verbose output |

## Dependencies

Pinned in `pyproject.toml`, locked in `uv.lock`, exported to `requirements.txt`:

- Python 3.12
- opencv-python 4.11.0.86 (the window code relies on its Cocoa backend)
- numpy 2.4.4
- qiskit 2.5.2
- qiskit-aer 0.17.2
- pyobjc-core 12.1
- pyobjc-framework-Cocoa 12.1

After changing a dependency: `uv lock`, then `uv export --format requirements.txt --no-hashes --no-emit-project --locked -o requirements.txt`.

## Project Structure

- **`reality.py`** — Main installation application
- **`lib/qpixl.py`** — QPIXL algorithm: builds compressed FRQI quantum circuits from image data
- **`models/face_detection_yunet_2023mar.onnx`** — YuNet face detection model from [opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (MIT licence)
- **`lib/helper.py`** — Support functions: Walsh-Hadamard transform, Gray code, angle conversion, statevector decoding, image reconstruction
