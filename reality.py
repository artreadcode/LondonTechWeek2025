import cv2
import numpy as np
import threading
import time
from queue import Queue
import sys
import os

# Import the actual QPIXL modules
import helper as hlp
import qpixl

# Updated imports for Qiskit 2.0
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector


class QuantumOperationTracker:
    """Track and display quantum operations in real-time per face"""
    
    def __init__(self, max_operations=6):  # Max 6 entries as requested
        self.max_operations = max_operations
        self.face_operations = {}  # Store operations per face
        self.lock = threading.Lock()
        
    def add_operation(self, face_id, operation_type, details=""):
        """Add a quantum operation to the display for specific face"""
        with self.lock:
            if face_id not in self.face_operations:
                self.face_operations[face_id] = []
            
            timestamp = time.strftime("%H:%M:%S")
            operation = {
                'operation': operation_type,
                'details': details,
                'timestamp': timestamp,
                'time': time.time(),
                'has_params': 'ry(' in operation_type.lower() or 'rz(' in operation_type.lower() or 'rx(' in operation_type.lower()
            }
            
            self.face_operations[face_id].append(operation)
            
            # Keep only the most recent operations (max 6)
            if len(self.face_operations[face_id]) > self.max_operations:
                self.face_operations[face_id] = self.face_operations[face_id][-self.max_operations:]
    
    def get_face_operations(self, face_id):
        """Get recent operations for specific face, prioritizing gates with parameters"""
        with self.lock:
            if face_id not in self.face_operations:
                return []
            
            # Get all operations for this face
            all_ops = self.face_operations[face_id].copy()
            
            # Prioritize operations with parameters (like ry, rz)
            param_ops = [op for op in all_ops if op['has_params']]
            other_ops = [op for op in all_ops if not op['has_params']]
            
            # Return parameterized gates first, then others if space remains
            return param_ops + other_ops
    
    def analyze_circuit(self, circuit, face_id):
        """Analyze a quantum circuit and extract key operations for specific face"""
        try:
            # Get quantum circuit metrics
            num_qubits = circuit.num_qubits
            circuit_depth = circuit.depth()
            
            # Add quantum metrics
            self.add_operation(face_id, f"Qubits: {num_qubits}", f"Depth: {circuit_depth}")
            
            # Focus on collecting rotation gates with parameters
            rotation_gates = []
            
            for instruction in circuit.data:
                gate_name = instruction.operation.name.upper()
                params = instruction.operation.params
                qubits = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
                
                # Collect specific gate details with parameters
                if gate_name == 'RY' and len(params) > 0 and len(qubits) > 0:
                    angle = float(params[0])
                    qubit = qubits[0]
                    rotation_gates.append((f"ry({angle:.3f}, q{qubit})", angle))
                elif gate_name == 'RZ' and len(params) > 0 and len(qubits) > 0:
                    angle = float(params[0])
                    qubit = qubits[0]
                    rotation_gates.append((f"rz({angle:.3f}, q{qubit})", angle))
                elif gate_name == 'RX' and len(params) > 0 and len(qubits) > 0:
                    angle = float(params[0])
                    qubit = qubits[0]
                    rotation_gates.append((f"rx({angle:.3f}, q{qubit})", angle))
            
            # Sort rotation gates by angle to show the most interesting ones
            rotation_gates.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Add the most interesting rotation gates (those with largest angles)
            for gate_detail, _ in rotation_gates[:5]:  # Show up to 5 rotation gates
                self.add_operation(face_id, gate_detail, "")
        
        except Exception as e:
            self.add_operation(face_id, "Analysis Error", str(e)[:20])


class EdgeDetectionProcessor:
    """Edge detection processor for dramatic background effects"""
    
    def __init__(self):
        self.edge_cache = None
        self.edge_cache_time = 0
        self.edge_cache_interval = 0.1  # Update edges frequently for dynamic effect
        
    def apply_edge_detection(self, frame):
        """Apply dramatic edge detection to create artistic background"""
        current_time = time.time()
        
        # Check cache to avoid recomputing every frame
        if (self.edge_cache is not None and 
            current_time - self.edge_cache_time < self.edge_cache_interval):
            return self.edge_cache
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Canny edge detection with strong parameters for dramatic effect
            edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
            
            # Dilate edges to make them MUCH thicker and more dramatic
            kernel = np.ones((4, 4), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=3)
            
            # Convert edges to 3-channel for display
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # Invert colors for dramatic effect (white edges on black background)
            edges_bgr = cv2.bitwise_not(edges_bgr)
            
            # Cache the result
            self.edge_cache = edges_bgr
            self.edge_cache_time = current_time
            
            return edges_bgr
            
        except Exception as e:
            print(f"Edge detection error: {e}")
            # Fallback to simple black background
            return np.zeros_like(frame)


class RealQPIXLProcessor:
    """
    Real QPIXL Quantum Pixel Processing with per-face operation tracking
    """
    
    def __init__(self, compression=40, verbose=False):
        self.compression = compression
        self.backend = Aer.get_backend('statevector_simulator')
        self.operation_tracker = QuantumOperationTracker()
        self.verbose = verbose  # Control debug output
        
    def prepare_face_for_qpixl(self, face_image):
        """Prepare face image for QPIXL processing"""
        # Convert to grayscale if needed
        if len(face_image.shape) == 3:
            gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_image.copy()
        
        # Use 32x32 for faster processing in real-time installation
        target_size = 32
        resized_face = cv2.resize(gray_face, (target_size, target_size))
        
        # Flatten and pad to next power of 2 if needed
        flattened = resized_face.flatten()
        padded = hlp.pad_0(flattened)
        
        return padded, (target_size, target_size)
    
    def process_face_with_qpixl(self, face_data, face_id=1):
        """Process face using the real QPIXL algorithm with operation tracking"""
        try:
            # Use the standard version for better performance
            circuit = qpixl.cFRQI(face_data, self.compression)
        
            # Analyze and track only the quantum gate operations for this specific face
            self.operation_tracker.analyze_circuit(circuit, face_id)
        
            # Try multiple transpilation strategies
            statevector = None
        
            # Strategy 1: Try with minimal optimization
            try:
                transpiled_circuit = transpile(circuit, self.backend, optimization_level=0)
                job = self.backend.run(transpiled_circuit)
                result = job.result()
                statevector = result.get_statevector()
                if self.verbose:
                    print(f"Face {face_id}: Transpilation successful with optimization_level=0")
            except Exception as e:
                if self.verbose:
                    print(f"Face {face_id}: Transpilation failed with optimization_level=0: {e}")
            
            # Strategy 2: Try without transpilation (direct execution)
            if statevector is None:
                try:
                    job = self.backend.run(circuit)
                    result = job.result()
                    statevector = result.get_statevector()
                    if self.verbose:
                        print(f"Face {face_id}: Direct execution successful (no transpilation)")
                except Exception as e2:
                    if self.verbose:
                        print(f"Face {face_id}: Direct execution failed: {e2}")
            
            # Strategy 3: Use Statevector directly
            if statevector is None:
                try:
                    statevector = Statevector.from_instruction(circuit)
                    if self.verbose:
                        print(f"Face {face_id}: Statevector creation successful")
                except Exception as e3:
                    if self.verbose:
                        print(f"Face {face_id}: Statevector creation failed: {e3}")
                    raise e3
        
            if statevector is None:
                raise Exception("All quantum execution strategies failed")
        
            # Decode the quantum state back to image
            decoded = hlp.decodeQPIXL(
                statevector.probabilities(),
                min_pixel_val=0,
                max_pixel_val=255
            )
        
            return decoded
        
        except Exception as e:
            if self.verbose:
                print(f"QPIXL processing error for face {face_id}: {e}")
            # Add error to operations
            self.operation_tracker.add_operation(face_id, "QPIXL Error", str(e)[:20])
            # Return a simple processed version of the original data
            return self.fallback_processing(face_data)
    
    def fallback_processing(self, face_data):
        """Fallback processing when quantum processing fails"""
        # Simple classical processing as fallback
        processed = np.array(face_data[:1024])  # Take first 1024 elements for 32x32
        
        # Apply some classical "quantum-like" effects
        processed = processed * 0.8 + np.random.normal(0, 10, processed.shape)
        processed = np.clip(processed, 0, 255)
        
        return processed
    
    def reconstruct_face_image(self, decoded_data, original_shape):
        """Reconstruct the face image from decoded QPIXL data"""
        try:
            # Trim padding if necessary
            expected_size = original_shape[0] * original_shape[1]
            if len(decoded_data) > expected_size:
                decoded_data = decoded_data[:expected_size]
            elif len(decoded_data) < expected_size:
                # Pad if too small
                padding = np.zeros(expected_size - len(decoded_data))
                decoded_data = np.concatenate([decoded_data, padding])
            
            # Reshape back to image and fix rotation
            reconstructed = hlp.reconstruct_img(decoded_data, original_shape)
            reconstructed = reconstructed.T  # Fix rotation issue
            
            # Ensure values are in valid range
            reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
            
            return reconstructed
            
        except Exception as e:
            if self.verbose:
                print(f"Reconstruction error: {e}")
            # Return a simple pattern if reconstruction fails
            pattern = np.random.randint(0, 256, original_shape, dtype=np.uint8)
            return pattern


class FaceDetector:
    """Face detection optimized for installation"""
    
    def __init__(self, verbose=False):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # For installation, detect larger faces (people standing further back)
        self.min_face_size = (80, 80)  # Minimum face size for detection
        self.verbose = verbose
    
    def detect_faces(self, frame):
        """Detect faces and return square regions"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Optimized parameters for installation environment
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5,
                minSize=self.min_face_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            face_regions = []
            for (x, y, w, h) in faces:
                # Make it square by using the larger dimension
                size = max(w, h)
                # Center the square
                center_x = x + w // 2
                center_y = y + h // 2
                new_x = max(0, center_x - size // 2)
                new_y = max(0, center_y - size // 2)
                
                # Ensure we don't go out of bounds
                if new_x + size <= frame.shape[1] and new_y + size <= frame.shape[0]:
                    face_region = frame[new_y:new_y+size, new_x:new_x+size]
                    if face_region.size > 0:
                        face_regions.append({
                            'region': face_region,
                            'coords': (new_x, new_y, size, size)
                        })
            
            return face_regions
        except Exception as e:
            if self.verbose:
                print(f"Face detection error: {e}")
            return []


class InstallationProcessor:
    """Main processor for the 4K installation display with edge detection background"""
    
    def __init__(self, compression=40, target_resolution=(3840, 2160), verbose=False):
        self.face_detector = FaceDetector(verbose)
        self.qpixl_processor = RealQPIXLProcessor(compression, verbose)
        self.edge_processor = EdgeDetectionProcessor()  # New edge detection processor
        self.input_queue = Queue(maxsize=3)  # Smaller queue for real-time performance
        self.output_queue = Queue(maxsize=3)
        self.processing = False
        self.compression = compression
        self.target_resolution = target_resolution
        self.face_cache = {}  # Cache processed faces to reduce computation
        self.cache_timeout = 2.0  # Reduced cache timeout to show parameter changes more frequently
        self.error_count = 0
        self.success_count = 0
        self.show_operations = True  # Toggle for operation display
        self.verbose = verbose
        
    def camera_capture_thread(self):
        """Thread for capturing camera input"""
        OPENCV_AVFOUNDATION_SKIP_AUTH=1 
        
        cap = cv2.VideoCapture(0)
        
        # Set camera to highest available resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if self.verbose:
            print(f"Camera initialized at {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        
        while self.processing:
            ret, frame = cap.read()
            if ret:
                # Resize to target resolution for display
                frame_resized = cv2.resize(frame, self.target_resolution)
                
                if not self.input_queue.full():
                    self.input_queue.put(frame_resized)
            time.sleep(0.033)  # ~30 FPS capture
        
        cap.release()
    
    def draw_face_quantum_panel(self, frame, face_id, coords, operations):
        """Draw individual quantum operations panel for each face with MUCH bigger text"""
        x, y, w, h = coords
        
        # Panel positioning - place next to face
        panel_width = 800  # Bigger panel
        panel_height = 600  # Bigger panel
        
        # Position panel to the right of face if possible, otherwise to the left
        if x + w + panel_width + 50 < frame.shape[1]:
            panel_x = x + w + 50
        else:
            panel_x = max(50, x - panel_width - 50)
        
        panel_y = max(50, min(y, frame.shape[0] - panel_height - 50))
        
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Draw panel border with face-specific color
        colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0), (0, 0, 255)]
        panel_color = colors[face_id % len(colors)]
        
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     panel_color, 8)  # Thick border
        
        # DRAMATICALLY BIGGER title
        cv2.putText(frame, f"FACE {face_id}", (panel_x + 30, panel_y + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 3.0, panel_color, 8)  # MUCH bigger
        
        cv2.putText(frame, "QUANTUM PARAMETERS", (panel_x + 30, panel_y + 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)  # Bigger
        
        # Draw line separator
        cv2.line(frame, (panel_x + 30, panel_y + 170), 
                (panel_x + panel_width - 30, panel_y + 170), panel_color, 4)
        
        # Display quantum operations with MUCH bigger text
        y_offset = panel_y + 220
        line_height = 60  # Much bigger line spacing
        
        # Show most recent operations (max 6)
        # Prioritize operations with parameters
        param_ops = [op for op in operations if op['has_params']]
        other_ops = [op for op in operations if not op['has_params']]
        
        # Show parameterized gates first
        display_ops = param_ops + other_ops
        display_ops = display_ops[:6]  # Limit to 6 total
        
        for i, op in enumerate(display_ops):
            if y_offset + line_height > panel_y + panel_height - 50:
                break
            
            # Color code by operation type
            if 'ry(' in op['operation'].lower():
                color = (0, 255, 255)  # Cyan for RY gates
            elif 'rz(' in op['operation'].lower():
                color = (255, 100, 255)  # Pink for RZ gates
            elif 'rx(' in op['operation'].lower():
                color = (100, 255, 255)  # Light blue for RX gates
            elif 'qubits:' in op['operation'].lower():
                color = (0, 255, 0)    # Green for qubit info
            elif 'error' in op['operation'].lower():
                color = (0, 0, 255)    # Red for errors
            else:
                color = (200, 200, 200)  # Gray for other operations
            
            # Draw operation with DRAMATICALLY bigger text
            operation_text = op['operation']
            
            # EXTRA LARGE text for parameterized gates
            if op['has_params']:
                cv2.putText(frame, operation_text, (panel_x + 50, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 5)  # EXTRA big for parameters
            else:
                cv2.putText(frame, operation_text, (panel_x + 50, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)  # Normal big for others
            
            y_offset += line_height
        
        # Draw connection line from face to panel
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        panel_connect_x = panel_x if panel_x > x else panel_x + panel_width
        panel_connect_y = panel_y + 100
        
        cv2.line(frame, (face_center_x, face_center_y), 
                (panel_connect_x, panel_connect_y), panel_color, 6)  # Thick connection line
    
    def face_processing_thread(self):
        """Thread for processing detected faces with real QPIXL and edge detection background"""
        while self.processing:
            try:
                if not self.input_queue.empty():
                    frame = self.input_queue.get()
                    current_time = time.time()
                    
                    # Apply dramatic edge detection to background
                    edge_background = self.edge_processor.apply_edge_detection(frame)
                    
                    # Detect faces
                    face_regions = self.face_detector.detect_faces(frame)
                    
                    # Start with the edge-detected background
                    output_frame = edge_background.copy()
                    
                    # Process each detected face
                    for i, face_data in enumerate(face_regions):
                        try:
                            face_region = face_data['region']
                            coords = face_data['coords']
                            x, y, w, h = coords
                            
                            # Draw square outline around face with thick border
                            colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0), (0, 0, 255)]
                            face_color = colors[i % len(colors)]
                            cv2.rectangle(output_frame, (x, y), (x+w, y+h), face_color, 8)  # Thick square outline
                            
                            # Create a cache key based on face position and size
                            cache_key = f"{x//30}_{y//30}_{w//30}_{h//30}"  # Larger quantization for more cache hits
                            
                            # Check if we have a recent processed version of this face
                            if (cache_key in self.face_cache and 
                                current_time - self.face_cache[cache_key]['timestamp'] < self.cache_timeout):
                                # Use cached processed face
                                processed_face_bgr = self.face_cache[cache_key]['processed']
                                if self.verbose:
                                    print(f"Using cached face {i+1}")
                            else:
                                # Process new face with QPIXL
                                if self.verbose:
                                    print(f"Processing new face {i+1}/{len(face_regions)}...")
                                
                                # Prepare face for QPIXL
                                face_array, original_shape = self.qpixl_processor.prepare_face_for_qpixl(face_region)
                                
                                # Apply QPIXL quantum processing with face ID for operation tracking
                                processed_data = self.qpixl_processor.process_face_with_qpixl(face_array, face_id=i+1)
                                self.success_count += 1
                                
                                # Reconstruct the processed face
                                processed_face = self.qpixl_processor.reconstruct_face_image(processed_data, original_shape)
                                
                                # Convert to BGR for display
                                processed_face_bgr = cv2.cvtColor(processed_face, cv2.COLOR_GRAY2BGR)
                                
                                # Apply quantum-inspired color effects
                                processed_face_hsv = cv2.cvtColor(processed_face_bgr, cv2.COLOR_BGR2HSV)
                                
                                # Create quantum color shift based on face position
                                hue_shift = (x + y) % 180
                                processed_face_hsv[:,:,0] = (processed_face_hsv[:,:,0] + hue_shift) % 180
                                processed_face_hsv[:,:,1] = np.clip(processed_face_hsv[:,:,1] * 1.3, 0, 255)  # Increase saturation
                                processed_face_hsv[:,:,2] = np.clip(processed_face_hsv[:,:,2] * 1.1, 0, 255)  # Increase brightness
                                
                                processed_face_bgr = cv2.cvtColor(processed_face_hsv, cv2.COLOR_HSV2BGR)
                                
                                # Cache the processed face
                                self.face_cache[cache_key] = {
                                    'processed': processed_face_bgr,
                                    'timestamp': current_time
                                }
                                
                                if self.verbose:
                                    print(f"Face {i+1} processed and cached successfully")
                            
                            # Get the exact dimensions of the target region
                            target_region = output_frame[y:y+h, x:x+w]
                            target_height, target_width = target_region.shape[:2]
                            
                            # Resize processed face to EXACTLY match the target region dimensions
                            processed_face_resized = cv2.resize(processed_face_bgr, (target_width, target_height))
                            
                            # Verify dimensions match before assignment
                            if processed_face_resized.shape == target_region.shape:
                                # Replace the face region with the quantum-processed square
                                output_frame[y:y+h, x:x+w] = processed_face_resized
                            else:
                                if self.verbose:
                                    print(f"Dimension mismatch: processed {processed_face_resized.shape} vs target {target_region.shape}")
                            
                            # Draw individual quantum operations panel for this face
                            if self.show_operations:
                                operations = self.qpixl_processor.operation_tracker.get_face_operations(i+1)
                                self.draw_face_quantum_panel(output_frame, i+1, coords, operations)
                        
                        except Exception as face_error:
                            if self.verbose:
                                print(f"Error processing face {i+1}: {face_error}")
                            self.error_count += 1
                            # Draw error indicator
                            cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 0, 255), 8)  # Red border for error
                    
                    # Clean up old cache entries
                    self.cleanup_cache(current_time)
                    
                    # Add installation info overlay with BIGGER text
                    self.add_info_overlay(output_frame, len(face_regions))
                    
                    if not self.output_queue.full():
                        self.output_queue.put(output_frame)
                
                else:
                    time.sleep(0.01)
            
            except Exception as thread_error:
                if self.verbose:
                    print(f"Face processing thread error: {thread_error}")
                self.error_count += 1
                time.sleep(0.1)  # Brief pause before retrying
    
    def cleanup_cache(self, current_time):
        """Remove old entries from face cache"""
        keys_to_remove = []
        for key, data in self.face_cache.items():
            if current_time - data['timestamp'] > self.cache_timeout * 2:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.face_cache[key]
    
    def add_info_overlay(self, frame, face_count):
        """Add installation information overlay with philosophical title"""
        height, width = frame.shape[:2]
        
        # Add main philosophical title at the top center
        title_text = "REALITY IS NOT WHAT IT SEEMS"
        title_font_scale = 3.5
        title_thickness = 8
        
        # Calculate text size to center it
        (title_width, title_height), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, title_thickness)
        title_x = (width - title_width) // 2
        title_y = 120
        
        # Add title with dramatic white text and black outline
        cv2.putText(frame, title_text, (title_x + 4, title_y + 4), 
                   cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, (0, 0, 0), title_thickness + 4)  # Black outline
        cv2.putText(frame, title_text, (title_x, title_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, (255, 255, 255), title_thickness)  # White text
        
        # Add subtitle below the main title
        subtitle_text = "View reality through quantum algorithm."
        subtitle_font_scale = 1.8
        subtitle_thickness = 4
        
        # Calculate subtitle position
        (subtitle_width, subtitle_height), _ = cv2.getTextSize(subtitle_text, cv2.FONT_HERSHEY_SIMPLEX, subtitle_font_scale, subtitle_thickness)
        subtitle_x = (width - subtitle_width) // 2
        subtitle_y = title_y + 80
        
        # Add subtitle with cyan color and black outline
        cv2.putText(frame, subtitle_text, (subtitle_x + 2, subtitle_y + 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, subtitle_font_scale, (0, 0, 0), subtitle_thickness + 2)  # Black outline
        cv2.putText(frame, subtitle_text, (subtitle_x, subtitle_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, subtitle_font_scale, (255, 255, 0), subtitle_thickness)  # Cyan text
        
        # Add face count in top-right corner
        if face_count > 0:
            cv2.putText(frame, f"Faces: {face_count}", (width - 400, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 6)
    
    def run(self, verbose=False):
        """Main execution method for installation"""
        self.verbose = verbose
        
        if verbose:
            print("=" * 60)
            print("Reality is Not What It Seems.")
            print("=" * 60)
            print(f"Target Resolution: {self.target_resolution[0]}x{self.target_resolution[1]}")
            print(f"Quantum Compression: {self.compression}%")
            print("Features:")
            print("- EDGE DETECTION BACKGROUND")
            print("- Individual quantum panels per face")
            print("- Live updating gate parameters (max 6)")
            print("- PRIORITIZED PARAMETER DISPLAY")
            print("- Square face outlines")
            print("- Dramatic artistic background")
            print("- DRAMATICALLY bigger text")
            print()
            print("Controls:")
            print("- 'q' - quit")
            print("- 'f' - fullscreen")
            print("- 'o' - toggle quantum panels")
            print("=" * 60)
        
        self.processing = True
        
        # Start background threads
        camera_thread = threading.Thread(target=self.camera_capture_thread)
        processing_thread = threading.Thread(target=self.face_processing_thread)
        
        camera_thread.start()
        processing_thread.start()
        
        # Create fullscreen window for installation
        cv2.namedWindow('Reality?', cv2.WINDOW_NORMAL)
        
        # Main display loop
        try:
            while self.processing:
                if not self.output_queue.empty():
                    output_frame = self.output_queue.get()
                    
                    # Display the frame
                    cv2.imshow('Reality?', output_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.processing = False
                        break
                    elif key == ord('f'):
                        # Toggle fullscreen
                        current_prop = cv2.getWindowProperty('Reality?', cv2.WND_PROP_FULLSCREEN)
                        if current_prop == cv2.WINDOW_FULLSCREEN:
                            cv2.setWindowProperty('Reality?', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        else:
                            cv2.setWindowProperty('Reality?', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    elif key == ord('o'):
                        # Toggle operations display
                        self.show_operations = not self.show_operations
                        if self.verbose:
                            print(f"Quantum operations display: {'ON' if self.show_operations else 'OFF'}")
                    elif key == ord('v'):
                        # Toggle verbose mode
                        self.verbose = not self.verbose
                        self.face_detector.verbose = self.verbose
                        self.qpixl_processor.verbose = self.verbose
                        if self.verbose:
                            print("Verbose mode: ON")
                        else:
                            print("Verbose mode: OFF")
                else:
                    time.sleep(0.01)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.processing = False
                        break
        
        except KeyboardInterrupt:
            if verbose:
                print("\nInstallation stopped by user")
            self.processing = False
        finally:
            # Ensure we clean up properly
            self.processing = False
            if verbose:
                print("Waiting for threads to finish...")
        
        # Cleanup
        camera_thread.join()
        processing_thread.join()
        cv2.destroyAllWindows()
        
        if verbose:
            print(f"Installation stopped. Final stats - Success: {self.success_count}, Errors: {self.error_count}")


if __name__ == "__main__":
    # Installation configuration
    compression_level = 40  # Balanced compression for performance
    display_resolution = (3840, 2160)  # 4K resolution
    verbose_mode = False  # Set to True for debug output
    
    print("Initializing QPIXL Installation...")
    processor = InstallationProcessor(
        compression=compression_level, 
        target_resolution=display_resolution,
        verbose=verbose_mode
    )
    processor.run(verbose=verbose_mode)
