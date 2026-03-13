# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

"Reality is Not What It Seems" — a live art installation for London Tech Week 2025 (9-11 June, Olympia London) in collaboration with UAL Creative Computing Institute. Captures camera feed, detects faces, encodes them through the QPIXL quantum image compression algorithm, and displays the result on a 4K screen with edge-detected artistic backgrounds.

## Running

Requires **conda base** Python (packages are in miniconda3, not brew Python):

```bash
# Main exhibition file (4K, fullscreen, edges-black background)
/Users/astrydpark/miniconda3/bin/python reality.py --fullscreen

# Development / debug
/Users/astrydpark/miniconda3/bin/python reality.py --verbose --resolution 1280 720

# Static pipeline test (no camera needed)
/Users/astrydpark/miniconda3/bin/python reality.py --test

# Test with custom image
/Users/astrydpark/miniconda3/bin/python reality.py --test --test-image photo.jpg

# Background modes: edges-black (default), edges-white, passthrough
/Users/astrydpark/miniconda3/bin/python reality.py --bg edges-white
```

Controls: `q` quit, `f` fullscreen toggle, `o` toggle quantum panels, `v` toggle verbose.

## Architecture

**`reality.py`** — Main entry point. CLI with argparse. Contains all installation logic:
- `LatestFrame` — Thread-safe single-slot frame holder (replaces Queue for frame dropping)
- `QuantumOperationTracker` — Tracks per-face quantum circuit parameters for HUD
- `EdgeDetector` — Canny edge detection with caching (native resolution, not 4K)
- `FaceDetector` — Haar cascade on downscaled frame (640px wide), coords scaled back
- `QPIXLProcessor` — Prepares faces (32x32), runs QPIXL circuit, decodes statevector
- `InstallationProcessor` — Orchestrates camera thread, processing thread, display loop
- `run_test()` — Static pipeline test mode (no camera)

**Performance pipeline**: Camera native res → face detect on 640px downscale → edge detect at native → compose at native → single upscale to display res for imshow.

**`qpixl.py`** — Core QPIXL algorithm. `cFRQI()` builds quantum circuit from image data using compressed FRQI encoding (based on [Nature paper](https://www.nature.com/articles/s41598-022-11024-y)). Do not modify.

**`helper.py`** — QPIXL support functions: Walsh-Hadamard transform, Gray code, angle conversion, statevector decoding (`decodeQPIXL`), image reconstruction. Do not modify.

**`qpixl_angs.py`** — Variant QPIXL with angle-based encoding. Not used by reality.py.

**`dummy/`** — Alternative approach using OSC to send face coordinates to TouchDesigner.

## Dependencies

opencv-python, numpy, qiskit 2.0, qiskit-aer (all in conda base environment).

## Key Constraints

- QPIXL processing is CPU-intensive (~1-3s per face at 32x32). Face caching (2s timeout) prevents reprocessing every frame.
- macOS requires camera authorization — on fresh install, run once from a terminal that has camera permission.
- The old variant files (`reality_black.py`, `reality_white.py`, `reality_start.py`) are superseded by `reality.py --bg` flag.
