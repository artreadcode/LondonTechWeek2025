"""
Reality is Not What It Seems
============================
Live art installation for London Tech Week 2025 (9-11 June, Olympia London).
Collaborative project with UAL Creative Computing Institute.

Captures camera feed, detects faces, encodes them through QPIXL quantum image
compression, and displays the result with edge-detected artistic backgrounds.

Usage:
    python reality.py                          # Default: 4K, edges-black
    python reality.py --lite                   # Lightweight mode for low-spec hardware
    python reality.py --lite --fullscreen      # Lite + fullscreen
    python reality.py --bg edges-white         # White background with dark edges
    python reality.py --bg passthrough         # No edge detection, raw camera bg
    python reality.py --resolution 1920 1080   # Custom display resolution
    python reality.py --fullscreen             # Start in fullscreen
    python reality.py --test                   # Static test mode (no camera)
    python reality.py --verbose                # Print FPS and timing info
"""

import argparse
import cv2
import numpy as np
import threading
import time

import lib.helper as hlp
import lib.qpixl as qpixl

from qiskit import transpile
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector

WINDOW_NAME = "Reality?"


# ---------------------------------------------------------------------------
# Latest-frame holder (replaces Queue — always keep only the newest frame)
# ---------------------------------------------------------------------------
class LatestFrame:
    """Thread-safe single-slot frame holder. Writers always overwrite; readers
    always get the most recent frame (or None)."""

    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()

    def put(self, frame):
        with self._lock:
            self._frame = frame

    def get(self):
        with self._lock:
            frame = self._frame
            self._frame = None
            return frame


# ---------------------------------------------------------------------------
# Quantum operation tracker (per-face, bounded)
# ---------------------------------------------------------------------------
class QuantumOperationTracker:
    """Track quantum circuit parameters for HUD display, per face."""

    def __init__(self, max_ops=6):
        self.max_ops = max_ops
        self._face_ops = {}
        self._lock = threading.Lock()

    def clear_stale(self, active_face_ids):
        """Remove data for faces no longer on screen."""
        with self._lock:
            stale = [fid for fid in self._face_ops if fid not in active_face_ids]
            for fid in stale:
                del self._face_ops[fid]

    def add(self, face_id, text, has_params=False):
        with self._lock:
            ops = self._face_ops.setdefault(face_id, [])
            ops.append({"text": text, "has_params": has_params})
            if len(ops) > self.max_ops:
                self._face_ops[face_id] = ops[-self.max_ops :]

    def get(self, face_id):
        with self._lock:
            ops = list(self._face_ops.get(face_id, []))
        # Prioritise parameterised gates
        param = [o for o in ops if o["has_params"]]
        other = [o for o in ops if not o["has_params"]]
        return (param + other)[: self.max_ops]

    def analyze_circuit(self, circuit, face_id):
        try:
            n_qubits = circuit.num_qubits
            depth = circuit.depth()
            self.add(face_id, f"Qubits: {n_qubits}  Depth: {depth}")

            # Collect rotation gates grouped by qubit
            qubit_gates = {}
            for inst in circuit.data:
                name = inst.operation.name.upper()
                if name not in ("RY", "RZ", "RX") or not inst.operation.params:
                    continue
                angle = float(inst.operation.params[0])
                qubit = circuit.find_bit(inst.qubits[0]).index
                qubit_gates.setdefault(qubit, []).append((name, angle))

            # Sort each qubit's gates by magnitude, take top ones
            entries = []
            for q, gates in qubit_gates.items():
                gates.sort(key=lambda g: abs(g[1]), reverse=True)
                for name, angle in gates[:2]:
                    entries.append((abs(angle), f"{name.lower()}({angle:.3f}, q{q})"))
            entries.sort(reverse=True)
            for _, text in entries[:5]:
                self.add(face_id, text, has_params=True)
        except Exception:
            self.add(face_id, "Analysis Error")


# ---------------------------------------------------------------------------
# Edge detection (operates at native resolution, cached)
# ---------------------------------------------------------------------------
class EdgeDetector:
    def __init__(self, cache_interval=0.1):
        self._cache = None
        self._cache_time = 0.0
        self._interval = cache_interval
        self._kernel = np.ones((3, 3), np.uint8)

    def process(self, frame, mode="edges-black"):
        """Return edge-detected background. mode: edges-black | edges-white | passthrough"""
        if mode == "passthrough":
            return frame.copy()

        now = time.time()
        if self._cache is not None and now - self._cache_time < self._interval:
            return self._cache.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        edges = cv2.dilate(edges, self._kernel, iterations=2)
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        if mode == "edges-white":
            result = cv2.bitwise_not(result)

        self._cache = result
        self._cache_time = now
        return result.copy()


# ---------------------------------------------------------------------------
# Face detector (runs on downscaled frame for speed)
# ---------------------------------------------------------------------------
class FaceDetector:
    def __init__(self, min_face=(60, 60), detect_width=640):
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._min_face = min_face
        self._detect_width = detect_width

    def detect(self, frame):
        """Detect faces on a downscaled copy; return coords in original frame space."""
        h, w = frame.shape[:2]
        scale = self._detect_width / w
        small = cv2.resize(frame, (self._detect_width, int(h * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=self._min_face,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        results = []
        inv_scale = 1.0 / scale
        for fx, fy, fw, fh in faces:
            # Scale back to original resolution
            ox = int(fx * inv_scale)
            oy = int(fy * inv_scale)
            ow = int(fw * inv_scale)
            oh = int(fh * inv_scale)

            # Make square
            size = max(ow, oh)
            cx, cy = ox + ow // 2, oy + oh // 2
            nx = max(0, cx - size // 2)
            ny = max(0, cy - size // 2)

            # Bounds check
            if nx + size > w or ny + size > h:
                continue
            region = frame[ny : ny + size, nx : nx + size]
            if region.size == 0:
                continue
            results.append({"region": region, "coords": (nx, ny, size, size)})
        return results


# ---------------------------------------------------------------------------
# Face tracker (stable IDs across frames via spatial proximity matching)
# ---------------------------------------------------------------------------
class FaceTracker:
    """Assign stable IDs to detected faces using nearest-neighbor matching.
    IDs are recycled from a small pool (1..max_faces) so numbers stay low."""

    def __init__(self, max_distance=150, expire_time=5.0, max_faces=6):
        self._tracks = {}   # id -> {"center": (cx, cy), "time": t}
        self._max_dist = max_distance
        self._expire = expire_time
        self._pool = list(range(max_faces, 0, -1))  # [6,5,4,3,2,1] — pop() gives lowest

    def _alloc_id(self):
        if self._pool:
            return self._pool.pop()  # lowest available
        # fallback: pick first int not in tracks
        i = 1
        while i in self._tracks:
            i += 1
        return i

    def update(self, faces):
        """Match detected faces to existing tracks. Returns faces with stable 'id' key."""
        now = time.time()

        # Expire old tracks, recycle their IDs
        expired = [tid for tid, t in self._tracks.items()
                   if now - t["time"] > self._expire]
        for tid in expired:
            del self._tracks[tid]
            if tid not in self._pool:
                self._pool.append(tid)
        self._pool.sort(reverse=True)  # keep pop() giving lowest

        if not faces:
            return []

        # Compute centres
        centres = []
        for face in faces:
            x, y, w, h = face["coords"]
            centres.append((x + w // 2, y + h // 2))

        # Build (distance, face_index, track_id) pairs
        pairs = []
        for fi, (cx, cy) in enumerate(centres):
            for tid, track in self._tracks.items():
                tcx, tcy = track["center"]
                dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
                if dist < self._max_dist:
                    pairs.append((dist, fi, tid))
        pairs.sort()

        # Greedy assignment
        assigned = {}
        used_tracks = set()
        for _, fi, tid in pairs:
            if fi in assigned or tid in used_tracks:
                continue
            assigned[fi] = tid
            used_tracks.add(tid)

        # Allocate recycled IDs for unmatched faces
        for fi in range(len(faces)):
            if fi not in assigned:
                assigned[fi] = self._alloc_id()

        # Update tracks and build result
        result = []
        for fi, face in enumerate(faces):
            tid = assigned[fi]
            self._tracks[tid] = {"center": centres[fi], "time": now}
            result.append({"region": face["region"], "coords": face["coords"], "id": tid})
        return result


# ---------------------------------------------------------------------------
# QPIXL quantum face processor
# ---------------------------------------------------------------------------
class QPIXLProcessor:
    def __init__(self, compression=40, verbose=False, face_size=32, lite=False):
        self.compression = compression
        self.verbose = verbose
        self.face_size = face_size
        self.lite = lite
        self.backend = Aer.get_backend("statevector_simulator")
        self.tracker = QuantumOperationTracker()

    def prepare(self, face_bgr):
        """Convert face ROI to padded 1-D array for QPIXL."""
        sz = self.face_size
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if face_bgr.ndim == 3 else face_bgr.copy()
        resized = cv2.resize(gray, (sz, sz)).astype(np.float64)
        resized = np.clip(resized, 0, 255)

        # Guard against degenerate images
        lo, hi = resized.min(), resized.max()
        if hi == lo:
            resized += np.random.uniform(-2, 2, resized.shape)
            resized = np.clip(resized, 1, 254)
            lo, hi = resized.min(), resized.max()
        # Normalise to [1, 254]
        resized = 1 + (resized - lo) * 253 / (hi - lo)
        flat = resized.astype(np.uint8).flatten().astype(np.float64)
        return hlp.pad_0(flat), (sz, sz)

    def process(self, padded, face_id=1):
        """Run QPIXL encoding → simulation → decode. Returns 1-D pixel array."""
        try:
            padded = np.asarray(padded, dtype=np.float64)
            if np.any(np.isnan(padded)) or padded.max() == padded.min():
                raise ValueError("degenerate input")

            circuit = qpixl.cFRQI(padded, self.compression)
            self.tracker.analyze_circuit(circuit, face_id)

            sv = self._run_circuit(circuit, face_id)
            return hlp.decodeQPIXL(sv.probabilities(), min_pixel_val=0, max_pixel_val=255)
        except Exception as e:
            if self.verbose:
                print(f"  QPIXL error face {face_id}: {e}")
            self.tracker.add(face_id, "QPIXL Error")
            return self._fallback(padded)

    def _run_circuit(self, circuit, face_id):
        """Try multiple execution strategies. Lite mode uses Statevector directly."""
        if self.lite:
            # Skip transpile + Aer overhead entirely
            try:
                return Statevector.from_instruction(circuit)
            except Exception:
                pass
            try:
                return self.backend.run(circuit).result().get_statevector()
            except Exception:
                pass
            tc = transpile(circuit, self.backend, optimization_level=0)
            return self.backend.run(tc).result().get_statevector()

        # Normal mode: transpile + run first
        try:
            tc = transpile(circuit, self.backend, optimization_level=0)
            return self.backend.run(tc).result().get_statevector()
        except Exception:
            pass
        try:
            return self.backend.run(circuit).result().get_statevector()
        except Exception:
            pass
        return Statevector.from_instruction(circuit)

    def _fallback(self, padded):
        n = self.face_size * self.face_size
        arr = np.asarray(padded[:n], dtype=np.float64)
        return np.clip(arr * 0.8 + np.random.normal(0, 10, arr.shape), 0, 255)

    def reconstruct(self, decoded, shape=None):
        """Decoded 1-D → 2-D uint8 image."""
        if shape is None:
            shape = (self.face_size, self.face_size)
        n = shape[0] * shape[1]
        if len(decoded) > n:
            decoded = decoded[:n]
        elif len(decoded) < n:
            decoded = np.concatenate([decoded, np.zeros(n - len(decoded))])
        img = hlp.reconstruct_img(decoded, shape).T
        return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# HUD / overlay drawing
# ---------------------------------------------------------------------------
FACE_COLORS = [
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
]


def _draw_panel(frame, face_id, coords, ops, scale):
    """Draw quantum parameter panel next to a face. Uses ROI-only blending."""
    x, y, w, h = coords
    color = FACE_COLORS[face_id % len(FACE_COLORS)]

    # Panel dimensions scaled to frame size
    pw = int(400 * scale)
    ph = int(300 * scale)

    # Position: right of face if room, else left
    fwidth = frame.shape[1]
    fheight = frame.shape[0]
    gap = int(25 * scale)
    if x + w + pw + gap < fwidth:
        px = x + w + gap
    else:
        px = max(gap, x - pw - gap)
    py = max(gap, min(y, fheight - ph - gap))

    # Clamp to frame bounds
    px2 = min(px + pw, fwidth)
    py2 = min(py + ph, fheight)
    if px2 - px < 50 or py2 - py < 50:
        return  # not enough room

    # ROI-only semi-transparent background (no full-frame copy)
    roi = frame[py:py2, px:px2]
    dark = np.zeros_like(roi)
    cv2.addWeighted(roi, 0.25, dark, 0.75, 0, roi)
    frame[py:py2, px:px2] = roi

    # Border
    thickness = max(2, int(3 * scale))
    cv2.rectangle(frame, (px, py), (px2, py2), color, thickness)

    # Title
    fs_title = 1.2 * scale
    fs_body = 0.8 * scale
    lh = int(30 * scale)
    tx, ty = px + int(15 * scale), py + int(35 * scale)

    cv2.putText(frame, f"FACE {face_id}", (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, fs_title, color, max(1, int(3 * scale)))
    ty += lh
    cv2.putText(frame, "QUANTUM PARAMETERS", (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, fs_body * 0.8, (255, 255, 255), max(1, int(2 * scale)))
    ty += int(10 * scale)
    cv2.line(frame, (tx, ty), (px2 - int(15 * scale), ty), color, max(1, int(2 * scale)))
    ty += lh

    for op in ops:
        if ty + lh > py2 - int(10 * scale):
            break
        text = op["text"]
        if "ry(" in text.lower():
            c = (0, 255, 255)
        elif "rz(" in text.lower():
            c = (255, 100, 255)
        elif "rx(" in text.lower():
            c = (100, 255, 255)
        elif "qubits:" in text.lower():
            c = (0, 255, 0)
        else:
            c = (200, 200, 200)

        fs = fs_body * (1.2 if op["has_params"] else 1.0)
        cv2.putText(frame, text, (tx + int(10 * scale), ty),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, c, max(1, int(2 * scale)))
        ty += lh

    # Connection line from face centre to panel edge
    fcx, fcy = x + w // 2, y + h // 2
    pcx = px if px > x else px2
    pcy = py + int(50 * scale)
    cv2.line(frame, (fcx, fcy), (pcx, pcy), color, max(1, int(2 * scale)))


def _draw_title(frame, scale):
    """Draw centred title and subtitle."""
    fh, fw = frame.shape[:2]
    title = "REALITY IS NOT WHAT IT SEEMS"
    fs = 1.8 * scale
    th = max(2, int(4 * scale))
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    tx = (fw - tw) // 2
    ty = int(70 * scale)
    # Shadow
    cv2.putText(frame, title, (tx + 2, ty + 2),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2)
    cv2.putText(frame, title, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th)

    sub = "View reality through quantum algorithm."
    fs2 = 0.9 * scale
    th2 = max(1, int(2 * scale))
    (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, fs2, th2)
    sx = (fw - sw) // 2
    sy = ty + int(45 * scale)
    cv2.putText(frame, sub, (sx + 1, sy + 1),
                cv2.FONT_HERSHEY_SIMPLEX, fs2, (0, 0, 0), th2 + 1)
    cv2.putText(frame, sub, (sx, sy),
                cv2.FONT_HERSHEY_SIMPLEX, fs2, (255, 255, 0), th2)


def _draw_face_count(frame, count, scale):
    if count > 0:
        fw = frame.shape[1]
        fs = 1.2 * scale
        cv2.putText(frame, f"Faces: {count}", (fw - int(220 * scale), int(50 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 0), max(1, int(3 * scale)))


# ---------------------------------------------------------------------------
# Main installation processor
# ---------------------------------------------------------------------------
class InstallationProcessor:
    def __init__(self, args):
        self.bg_mode = args.bg
        self.display_res = tuple(args.resolution)
        self.compression = args.compression
        self.verbose = args.verbose
        self.start_fullscreen = args.fullscreen
        self.show_panels = True
        self.lite = getattr(args, "lite", False)

        if self.lite:
            self.face_detector = FaceDetector(detect_width=320, min_face=(30, 30))
            self.qpixl = QPIXLProcessor(self.compression, self.verbose,
                                         face_size=16, lite=True)
            self.edge_detector = EdgeDetector(cache_interval=0.5)
            self._cache_timeout = 5.0
            self._cam_width = 640
            self._cam_height = 480
        else:
            self.face_detector = FaceDetector()
            self.qpixl = QPIXLProcessor(self.compression, self.verbose)
            self.edge_detector = EdgeDetector()
            self._cache_timeout = 2.0
            self._cam_width = 1920
            self._cam_height = 1080

        self.face_tracker = FaceTracker()
        self._latest_input = LatestFrame()
        self._latest_output = LatestFrame()
        self._running = False

        self._face_cache = {}
        self._stats_lock = threading.Lock()
        self._success = 0
        self._errors = 0
        self._last_tracked = []
        self._frame_count = 0

    # --- threads -----------------------------------------------------------

    def _camera_thread(self):
        cap = None
        fail_count = 0
        max_consecutive_fails = 30

        while self._running:
            # (Re-)open camera if needed
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cam_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cam_height)
                cap.set(cv2.CAP_PROP_FPS, 30)
                fail_count = 0
                if self.verbose:
                    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    print(f"Camera opened: {w:.0f}x{h:.0f}")

            ret, frame = cap.read()
            if ret:
                self._latest_input.put(frame)
                fail_count = 0
            else:
                fail_count += 1
                if fail_count >= max_consecutive_fails:
                    if self.verbose:
                        print("Camera: too many read failures, reopening...")
                    cap.release()
                    cap = None
                    time.sleep(1.0)
                else:
                    time.sleep(0.03)  # avoid tight loop on transient failure

        if cap is not None:
            cap.release()

    def _processing_thread(self):
        fps_time = time.time()
        fps_count = 0

        while self._running:
            frame = self._latest_input.get()
            if frame is None:
                time.sleep(0.005)
                continue

            t0 = time.time()

            # 1. Edge detection at native camera resolution
            background = self.edge_detector.process(frame, self.bg_mode)

            # 2. Face detection on downscaled copy (lite: skip every other frame)
            self._frame_count += 1
            if self.lite and self._frame_count % 2 == 0 and self._last_tracked:
                tracked = self._last_tracked
                # Re-extract regions from current frame
                for t in tracked:
                    x, y, w, h = t["coords"]
                    if y + h <= frame.shape[0] and x + w <= frame.shape[1]:
                        t["region"] = frame[y:y+h, x:x+w]
            else:
                raw_faces = self.face_detector.detect(frame)
                tracked = self.face_tracker.update(raw_faces)
                self._last_tracked = tracked

            # 3. Compose output at native resolution
            output = background  # already a copy from EdgeDetector

            now = time.time()
            active_ids = set()

            for face in tracked:
                fid = face["id"]
                active_ids.add(fid)
                try:
                    region = face["region"]
                    x, y, w, h = face["coords"]
                    color = FACE_COLORS[fid % len(FACE_COLORS)]

                    # Face outline
                    cv2.rectangle(output, (x, y), (x + w, y + h), color, max(2, w // 40))

                    # Cache lookup — invalidate on timeout OR significant movement
                    ckey = str(fid)
                    cached = self._face_cache.get(ckey)
                    cache_hit = False
                    if cached and now - cached["time"] < self._cache_timeout:
                        ox, oy = cached.get("pos", (x, y))
                        if abs(x - ox) < 40 and abs(y - oy) < 40:
                            cache_hit = True
                    if cache_hit:
                        proc_bgr = cached["img"]
                    else:
                        padded, shape = self.qpixl.prepare(region)
                        decoded = self.qpixl.process(padded, fid)
                        gray_face = self.qpixl.reconstruct(decoded, shape)
                        proc_bgr = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2BGR)

                        # Colour shift
                        hsv = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2HSV)
                        hue_shift = (x + y) % 180
                        hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + hue_shift) % 180
                        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255).astype(np.uint8)
                        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255).astype(np.uint8)
                        proc_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                        self._face_cache[ckey] = {"img": proc_bgr, "time": now, "pos": (x, y)}
                        with self._stats_lock:
                            self._success += 1

                    # Stamp processed face into output
                    target = output[y : y + h, x : x + w]
                    resized = cv2.resize(proc_bgr, (target.shape[1], target.shape[0]))
                    if resized.shape == target.shape:
                        output[y : y + h, x : x + w] = resized

                except Exception as e:
                    if self.verbose:
                        print(f"  Face {fid} error: {e}")
                    with self._stats_lock:
                        self._errors += 1

            # Clean stale caches
            self._cleanup_cache(now)
            self.qpixl.tracker.clear_stale(active_ids)

            # HUD scale factor (relative to 1080p)
            scale = output.shape[0] / 1080.0

            # Draw panels
            if self.show_panels:
                for face in tracked:
                    fid = face["id"]
                    ops = self.qpixl.tracker.get(fid)
                    _draw_panel(output, fid, face["coords"], ops, scale)

            _draw_title(output, scale)
            _draw_face_count(output, len(tracked), scale)

            # 4. Upscale to display resolution ONCE
            if (output.shape[1], output.shape[0]) != self.display_res:
                output = cv2.resize(output, self.display_res)

            self._latest_output.put(output)

            # FPS counter
            fps_count += 1
            elapsed = time.time() - fps_time
            if self.verbose and elapsed >= 2.0:
                print(f"Processing FPS: {fps_count / elapsed:.1f}  "
                      f"Frame time: {(time.time() - t0) * 1000:.0f}ms  "
                      f"Faces: {len(tracked)}")
                fps_count = 0
                fps_time = time.time()

    def _cleanup_cache(self, now):
        stale = [k for k, v in self._face_cache.items()
                 if now - v["time"] > self._cache_timeout * 2]
        for k in stale:
            del self._face_cache[k]

    # --- main loop ---------------------------------------------------------

    def run(self):
        if self.lite:
            print("=" * 50)
            print("LITE MODE — optimised for low-spec hardware")
            print(f"  Face: {self.qpixl.face_size}x{self.qpixl.face_size}  "
                  f"Camera: {self._cam_width}x{self._cam_height}  "
                  f"Compression: {self.compression}%")
            print("=" * 50)
        if self.verbose:
            print("=" * 50)
            print("Reality is Not What It Seems")
            print("=" * 50)
            print(f"Display: {self.display_res[0]}x{self.display_res[1]}")
            print(f"Background: {self.bg_mode}")
            print(f"Compression: {self.compression}%")
            print("Controls: q=quit  f=fullscreen  o=toggle panels  v=verbose")
            print("=" * 50)

        self._running = True

        cam = threading.Thread(target=self._camera_thread, daemon=True)
        proc = threading.Thread(target=self._processing_thread, daemon=True)
        cam.start()
        proc.start()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if self.start_fullscreen:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        last_display = None
        try:
            while self._running:
                output = self._latest_output.get()
                if output is not None:
                    last_display = output
                if last_display is not None:
                    cv2.imshow(WINDOW_NAME, last_display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("f"):
                    prop = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
                    new = cv2.WINDOW_NORMAL if prop == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, new)
                elif key == ord("o"):
                    self.show_panels = not self.show_panels
                elif key == ord("v"):
                    self.verbose = not self.verbose
                    self.qpixl.verbose = self.verbose

        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            if self.verbose:
                with self._stats_lock:
                    print(f"\nDone. Success: {self._success}  Errors: {self._errors}")
            cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Static test mode (no camera required)
# ---------------------------------------------------------------------------
def run_test(args):
    """Run the full pipeline on a static image or generated test pattern."""
    print("=== Static Pipeline Test ===\n")

    # Load or generate test image
    if args.test_image:
        img = cv2.imread(args.test_image)
        if img is None:
            print(f"Error: cannot read {args.test_image}")
            return
        print(f"Loaded: {args.test_image}  ({img.shape[1]}x{img.shape[0]})")
    else:
        # Generate 640x480 test image with a synthetic "face-like" pattern
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (80, 80, 80)
        # Draw an ellipse as a rough face
        cv2.ellipse(img, (320, 240), (80, 100), 0, 0, 360, (200, 180, 160), -1)
        cv2.circle(img, (295, 220), 10, (50, 50, 50), -1)  # left eye
        cv2.circle(img, (345, 220), 10, (50, 50, 50), -1)  # right eye
        cv2.ellipse(img, (320, 270), (25, 10), 0, 0, 360, (100, 80, 80), -1)  # mouth
        print("Using generated test pattern (640x480)")

    # 1. Edge detection
    edge_det = EdgeDetector()
    t0 = time.time()
    bg = edge_det.process(img, args.bg)
    t_edge = time.time() - t0
    print(f"[1] Edge detection ({args.bg}): {t_edge * 1000:.1f} ms")

    # 2. Face detection
    face_det = FaceDetector()
    t0 = time.time()
    faces = face_det.detect(img)
    t_face = time.time() - t0
    print(f"[2] Face detection: {t_face * 1000:.1f} ms  ({len(faces)} faces found)")

    if not faces:
        print("\nNo faces detected. Saving edge-detected background only.")
        cv2.imwrite("test_output.png", bg)
        print("Saved: test_output.png")
        return

    # 3. QPIXL processing
    lite = getattr(args, "lite", False)
    face_size = 16 if lite else 32
    proc = QPIXLProcessor(args.compression, verbose=True, face_size=face_size, lite=lite)
    if lite:
        print(f"  [LITE] Face size: {face_size}x{face_size}")
    output = bg.copy()

    for i, face in enumerate(faces):
        fid = i + 1
        x, y, w, h = face["coords"]
        color = FACE_COLORS[i % len(FACE_COLORS)]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

        t0 = time.time()
        padded, shape = proc.prepare(face["region"])
        decoded = proc.process(padded, fid)
        gray_face = proc.reconstruct(decoded, shape)
        t_qpixl = time.time() - t0
        print(f"[3] QPIXL face {fid}: {t_qpixl * 1000:.1f} ms")

        proc_bgr = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2BGR)
        target = output[y : y + h, x : x + w]
        resized = cv2.resize(proc_bgr, (target.shape[1], target.shape[0]))
        if resized.shape == target.shape:
            output[y : y + h, x : x + w] = resized

        # Draw panel
        scale = output.shape[0] / 1080.0
        ops = proc.tracker.get(fid)
        _draw_panel(output, fid, face["coords"], ops, scale)

    _draw_title(output, output.shape[0] / 1080.0)
    _draw_face_count(output, len(faces), output.shape[0] / 1080.0)

    cv2.imwrite("test_output.png", output)
    print(f"\nSaved: test_output.png ({output.shape[1]}x{output.shape[0]})")
    print("Pipeline test complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Reality is Not What It Seems — QPIXL installation")
    p.add_argument("--bg", choices=["edges-black", "edges-white", "passthrough"],
                    default="edges-black", help="Background mode (default: edges-black)")
    p.add_argument("--resolution", type=int, nargs=2, default=None,
                    metavar=("W", "H"), help="Display resolution (default: 3840x2160, lite: 1280x720)")
    p.add_argument("--compression", type=int, default=None,
                    help="QPIXL compression 0-100 (default: 40, lite: 70)")
    p.add_argument("--verbose", action="store_true", help="Print FPS and debug info")
    p.add_argument("--fullscreen", action="store_true", help="Start in fullscreen")
    p.add_argument("--test", action="store_true", help="Run static pipeline test (no camera)")
    p.add_argument("--test-image", type=str, default=None,
                    help="Image file for --test mode (optional)")
    p.add_argument("--lite", action="store_true",
                    help="Lightweight mode for low-spec hardware (16x16 faces, 720p, lower camera res)")
    args = p.parse_args()

    # Apply lite defaults (user-specified values take priority)
    if args.lite:
        if args.resolution is None:
            args.resolution = [1280, 720]
        if args.compression is None:
            args.compression = 70
    else:
        if args.resolution is None:
            args.resolution = [3840, 2160]
        if args.compression is None:
            args.compression = 40

    if args.test:
        run_test(args)
    else:
        print("Initializing QPIXL Installation...")
        processor = InstallationProcessor(args)
        processor.run()


if __name__ == "__main__":
    main()
