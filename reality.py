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
import qpixl_angs

# Updated imports for Qiskit 2.0
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector


class QuantumOperationTracker:
    """Track and display quantum operations in real-time"""
    
    def __init__(self, max_operations=10):
        self.max_operations = max_operations
        self.operations = []
        self.lock = threading.Lock()
        
    def add_operation(self, face_id, operation_type, details=""):
        """Add a quantum operation to the display"""
        with self.lock:
            timestamp = time.strftime("%H:%M:%S")
            operation = {
                'face_id': face_id,
                'operation': operation_type,
                'details': details,
                'timestamp': timestamp,
                'time': time.time()
            }
            
            self.operations.append(operation)
            
            # Keep only the most recent operations
            if len(self.operations) > self.max_operations:
                self.operations = self.operations[-self.max_operations:]
    
    def get_recent_operations(self):
        """Get recent operations for display"""
        with self.lock:
            return self.operations.copy()
    
    def analyze_circuit(self, circuit, face_id):
        """Analyze a quantum circuit and extract key operations"""
        try:
            # Analyze the circuit instructions for quantum gates only
            gate_counts = {}
            total_gates = 0
            for instruction in circuit.data:
                gate_name = instruction.operation.name.upper()
                gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
                total_gates += 1
    
            # Get quantum circuit metrics
            num_qubits = circuit.num_qubits
            circuit_depth = circuit.depth()
        
            # Add quantum metrics first
            self.add_operation(face_id, f"QUBITS: {num_qubits}", f"depth: {circuit_depth}")
        
            # Add only the quantum gate operations with enhanced info
            for gate, count in gate_counts.items():
                if count > 0:
                    percentage = (count / total_gates * 100) if total_gates > 0 else 0
                    if gate == 'RY':
                        self.add_operation(face_id, f"qc.ry()", f"×{count} ({percentage:.0f}%)")
                    elif gate == 'H':
                        self.add_operation(face_id, f"qc.h()", f"×{count} superposition")
                    elif gate == 'CX' or gate == 'CNOT':
                        self.add_operation(face_id, f"qc.cx()", f"×{count} entanglement")
                    elif gate == 'RZ':
                        self.add_operation(face_id, f"qc.rz()", f"×{count} phase shift")
                    elif gate == 'X':
                        self.add_operation(face_id, f"qc.x()", f"×{count} bit flip")
                    elif gate == 'Y':
                        self.add_operation(face_id, f"qc.y()", f"×{count} Y-rotation")
                    elif gate == 'Z':
                        self.add_operation(face_id, f"qc.z()", f"×{count} phase flip")
                    else:
                        self.add_operation(face_id, f"qc.{gate.lower()}()", f"×{count}")
        
            # Add quantum complexity metrics
            if total_gates > 50:
                self.add_operation(face_id, "COMPLEXITY", "HIGH quantum load")
            elif total_gates > 20:
                self.add_operation(face_id, "COMPLEXITY", "MEDIUM quantum load")
            else:
                self.add_operation(face_id, "COMPLEXITY", "LOW quantum load")
    
        except Exception as e:
            # Silent error handling for cleaner output
            pass


class RealQPIXLProcessor:
    """
    Real QPIXL Quantum Pixel Processing with lightweight operation tracking
    """
    
    def __init__(self, compression=40, use_angles_version=False, verbose=False):
        self.compression = compression
        self.use_angles_version = use_angles_version
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
        
            # Analyze and track only the quantum gate operations
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
    """Main processor for the 3m x 8m installation display with quantum operation display"""
    
    def __init__(self, compression=40, target_resolution=(3840, 2160), verbose=False):
        self.face_detector = FaceDetector(verbose)
        self.qpixl_processor = RealQPIXLProcessor(compression, False, verbose)  # Use standard version
        self.input_queue = Queue(maxsize=3)  # Smaller queue for real-time performance
        self.output_queue = Queue(maxsize=3)
        self.processing = False
        self.compression = compression
        self.target_resolution = target_resolution
        self.face_cache = {}  # Cache processed faces to reduce computation
        self.cache_timeout = 3.0  # Increased cache timeout to reduce quantum processing
        self.error_count = 0
        self.success_count = 0
        self.show_operations = True  # Toggle for operation display
        self.verbose = verbose
        
    def camera_capture_thread(self):
        """Thread for capturing camera input"""
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
    
    def face_processing_thread(self):
        """Thread for processing detected faces with real QPIXL"""
        while self.processing:
            try:
                if not self.input_queue.empty():
                    frame = self.input_queue.get()
                    current_time = time.time()
                    
                    # Detect faces
                    face_regions = self.face_detector.detect_faces(frame)
                    
                    # Start with the original frame as background
                    output_frame = frame.copy()
                    
                    # Process each detected face
                    for i, face_data in enumerate(face_regions):
                        try:
                            face_region = face_data['region']
                            coords = face_data['coords']
                            x, y, w, h = coords
                            
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
                                # Simply replace the face region with the quantum-processed square
                                output_frame[y:y+h, x:x+w] = processed_face_resized
                                # NO OUTLINE - clean quantum faces
                            else:
                                if self.verbose:
                                    print(f"Dimension mismatch: processed {processed_face_resized.shape} vs target {target_region.shape}")
                                # Fallback: just draw a rectangle to indicate face detection
                                cv2.rectangle(output_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue border for error
                        
                        except Exception as face_error:
                            if self.verbose:
                                print(f"Error processing face {i+1}: {face_error}")
                            self.error_count += 1
                            # Draw error indicator
                            cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red border for error
                    
                    # Clean up old cache entries
                    self.cleanup_cache(current_time)
                    
                    # Add installation info overlay
                    self.add_info_overlay(output_frame, len(face_regions))
                    
                    # Add quantum operations display
                    if self.show_operations:
                        output_frame = self.add_quantum_operations_display(output_frame)
                    
                    if not self.output_queue.full():
                        self.output_queue.put(output_frame)
                
                else:
                    time.sleep(0.01)
            
            except Exception as thread_error:
                if self.verbose:
                    print(f"Face processing thread error: {thread_error}")
                self.error_count += 1
                time.sleep(0.1)  # Brief pause before retrying
    
    def add_quantum_operations_display(self, frame):
        """Add lightweight quantum operations display to the frame"""
        try:
            # Get recent operations
            operations = self.qpixl_processor.operation_tracker.get_recent_operations()
            
            if operations:
                frame_height, frame_width = frame.shape[:2]
                
                # Create a semi-transparent overlay area in the bottom-right
                overlay_width = 600
                overlay_height = 400
                overlay_x = frame_width - overlay_width - 20
                overlay_y = frame_height - overlay_height - 20
                
                # Create overlay background
                overlay = frame[overlay_y:overlay_y+overlay_height, overlay_x:overlay_x+overlay_width].copy()
                overlay = cv2.addWeighted(overlay, 0.3, np.zeros_like(overlay), 0.7, 0)  # Semi-transparent
                
                # Add title
                cv2.putText(overlay, "QUANTUM OPERATIONS", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.line(overlay, (10, 40), (overlay_width-10, 40), (255, 255, 255), 1)
                
                # Display recent operations
                y_offset = 60
                line_height = 25
                
                # Show most recent operations first
                recent_ops = operations[-12:]  # Show last 12 operations
                recent_ops.reverse()
                
                for i, op in enumerate(recent_ops):
                    if y_offset + line_height > overlay_height - 10:
                        break
                    
                    # Color code by operation type
                    if op['operation'].startswith('qc.'):
                        color = (0, 255, 255)  # Yellow for quantum gates
                    elif op['operation'].startswith('QUBITS'):
                        color = (255, 100, 255)  # Magenta for qubit info
                    elif op['operation'] == 'COMPLEXITY':
                        color = (100, 255, 100)  # Light green for complexity
                    elif op['operation'] in ['SUCCESS', 'COMPLETE']:
                        color = (0, 255, 0)    # Green for success
                    elif op['operation'] in ['ERROR', 'FAILED']:
                        color = (0, 0, 255)    # Red for errors
                    else:
                        color = (200, 200, 200)  # Gray for other operations
                    
                    # Format the operation text
                    face_text = f"F{op['face_id']}"
                    op_text = f"{op['operation']}"
                    detail_text = f"{op['details']}" if op['details'] else ""
                    
                    # Draw face ID
                    cv2.putText(overlay, face_text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Draw operation
                    cv2.putText(overlay, op_text, (50, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Draw details if they fit
                    if detail_text and len(detail_text) < 30:
                        cv2.putText(overlay, detail_text, (250, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                    
                    y_offset += line_height
                
                # Apply the overlay to the frame
                frame[overlay_y:overlay_y+overlay_height, overlay_x:overlay_x+overlay_width] = overlay
            
            return frame
            
        except Exception as e:
            if self.verbose:
                print(f"Quantum operations display error: {e}")
            return frame
    
    def cleanup_cache(self, current_time):
        """Remove old entries from face cache"""
        keys_to_remove = []
        for key, data in self.face_cache.items():
            if current_time - data['timestamp'] > self.cache_timeout * 2:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.face_cache[key]
    
    def add_info_overlay(self, frame, face_count):
        """Add installation information overlay"""
        height, width = frame.shape[:2]
        
        # Add title in top-left
        cv2.putText(frame, "Reality is Not What It Seems", (50, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4)
        cv2.putText(frame, "What Quantum Computing would encode.", (50, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
        
        # Add face count in top-right
        if face_count > 0:
            cv2.putText(frame, f"Faces: {face_count}", (width - 300, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # Add compression info and stats in bottom-left
        cv2.putText(frame, f"Compression: {self.compression}%", (50, height - 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 255), 2)
        cv2.putText(frame, f"Success: {self.success_count} | Errors: {self.error_count}", (50, height - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    def run(self, verbose=False):
        """Main execution method for installation"""
        self.verbose = verbose
        
        if verbose:
            print("=" * 80)
            print("QC INSTALLATION WITH LIVE OPERATION DISPLAY")
            print("=" * 80)
            print(f"Target Resolution: {self.target_resolution[0]}x{self.target_resolution[1]}")
            print(f"Quantum Compression: {self.compression}%")
            print("Real-time QPIXL processing with live quantum operation display")
            print("No outlines around quantum faces")
            print("Live quantum operations shown in bottom-right corner")
            print()
            print("Press 'q' to quit, 'f' for fullscreen, 'o' to toggle operations, 'v' to toggle verbose")
            print("=" * 80)
        
        self.processing = True
        
        # Start background threads
        camera_thread = threading.Thread(target=self.camera_capture_thread)
        processing_thread = threading.Thread(target=self.face_processing_thread)
        
        camera_thread.start()
        processing_thread.start()
        
        # Create fullscreen window for installation
        cv2.namedWindow('Reality is Not What It Seems', cv2.WINDOW_NORMAL)
        
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