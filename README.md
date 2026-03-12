# Reality is Not What It Seems

A live art installation for **London Tech Week 2025** (9–11 June, Olympia London), in collaboration with UAL Creative Computing Institute.

Visitors stand in front of a camera. Their faces are detected, encoded through a real quantum image compression algorithm (QPIXL), and displayed back — pixelated, colour-shifted, seen through the lens of quantum information. A panel beside each face shows the actual quantum gate parameters that encoded *their specific face* into a quantum state.

The pixelation isn't a glitch. It's what a quantum computer would "see" if it stored your face.

## How It Works

The installation runs a live pipeline from camera to 4K display:

### 1. Capture & Detect
A webcam captures the scene. OpenCV's Haar cascade detects faces on a downscaled frame for speed, then coordinates are mapped back to full resolution.

### 2. Quantum Encoding (QPIXL)
Each detected face is resized to 32×32 grayscale and encoded using **QPIXL** — a quantum image compression algorithm based on [this Nature paper](https://www.nature.com/articles/s41598-022-11024-y) (compressed FRQI). The steps:

1. **Pixel → angle**: Each pixel brightness (0–254) maps to a rotation angle (0 to π/2)
2. **Walsh-Hadamard Transform**: Transforms pixel angles into frequency-domain coefficients (similar to how JPEG uses DCT). Natural images concentrate energy in low frequencies, so many coefficients are near zero
3. **Compression**: The smallest 40% of coefficients are zeroed out — fewer non-zero coefficients means fewer quantum gates
4. **Gray code reordering**: Consecutive pixel addresses differ by only 1 bit, minimising CNOT gates needed
5. **Circuit construction**: Builds a quantum circuit with **k+1 qubits** (10 address qubits for 1024 pixel positions + 1 colour qubit encoding brightness via RY rotations). The address qubits start in superposition (H gates), and RY + CX gate sequences "paint" each pixel's brightness onto the colour qubit

For a 32×32 face at 40% compression, this produces an 11-qubit circuit with ~600 RY gates and ~1700 CX gates at depth ~1500.

### 3. Simulate & Decode
The circuit runs on Qiskit Aer's statevector simulator, producing 2048 complex amplitudes. The decoder extracts pixel angles from amplitude pairs via `arctan2` and scales them back to grayscale values.

### 4. Display
The quantum-reconstructed face (with artistic hue shift) is stamped onto an edge-detected background. A HUD panel next to each face shows the live quantum circuit parameters — the actual `ry(angle, q0)` rotation gates that encoded that person's face.

## Running

Requires conda base Python (dependencies are installed in miniconda3):

```bash
# Exhibition mode (4K, fullscreen)
python reality.py --fullscreen

# Development (windowed, debug info)
python reality.py --verbose --resolution 1280 720

# Static pipeline test (no camera needed)
python reality.py --test

# Test with a specific image
python reality.py --test --test-image photo.jpg
```

### Options

| Flag | Description |
|------|-------------|
| `--bg edges-black` | Black background with white edges (default) |
| `--bg edges-white` | White background with dark edges |
| `--bg passthrough` | Raw camera feed as background |
| `--resolution W H` | Display resolution (default: 3840 2160) |
| `--compression N` | QPIXL compression 0–100 (default: 40) |
| `--fullscreen` | Start in fullscreen |
| `--verbose` | Print FPS and timing info |

### Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `f` | Toggle fullscreen |
| `o` | Toggle quantum parameter panels |
| `v` | Toggle verbose output |

## Dependencies

- Python 3.12 (conda base)
- opencv-python
- numpy
- qiskit 2.0
- qiskit-aer

## Project Structure

- **`reality.py`** — Main installation application
- **`qpixl.py`** — QPIXL algorithm: builds compressed FRQI quantum circuits from image data
- **`helper.py`** — Support functions: Walsh-Hadamard transform, Gray code, angle conversion, statevector decoding, image reconstruction
- **`qpixl_angs.py`** — Variant QPIXL with angle-based encoding (not used by the installation)
- **`dummy/`** — (Suspended) Alternative approach using OSC to send face coordinates to TouchDesigner
