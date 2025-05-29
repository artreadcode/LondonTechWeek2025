# Python script for sending face coordinates via OSC
import cv2
import numpy as np
from pythonosc import udp_client
import time

def setup_osc_client(ip="127.0.0.1", port=7000):
    """Setup OSC client for sending data to TouchDesigner"""
    return udp_client.SimpleUDPClient(ip, port)

def detect_faces_and_send(client):
    """Main face detection and OSC sending loop"""
    # Initialize camera
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Previous positions for motion calculation
    prev_positions = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Initialize all face positions to 0
        face_data = {}
        for i in range(10):
            face_data[f'face{i}_x'] = 0.0
            face_data[f'face{i}_y'] = 0.0
            face_data[f'face{i}_motion'] = 0.0
        
        # Process detected faces
        for i, (x, y, w, h) in enumerate(faces[:10]):  # Limit to 10 faces
            # Calculate face center
            center_x = (x + w/2) / width   # Normalize to 0-1
            center_y = (y + h/2) / height  # Normalize to 0-1
            
            # Store current position
            current_pos = (center_x, center_y)
            
            # Calculate motion if we have previous position
            motion = 0.0
            if i in prev_positions:
                prev_x, prev_y = prev_positions[i]
                motion = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
            
            prev_positions[i] = current_pos
            
            # Store in face data
            face_data[f'face{i}_x'] = center_x
            face_data[f'face{i}_y'] = center_y
            face_data[f'face{i}_motion'] = motion
            
            # Draw rectangle on frame for visualization
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.circle(frame, (int(x + w/2), int(y + h/2)), 5, (0, 255, 0), -1)
        
        # Send all face data via OSC
        for key, value in face_data.items():
            client.send_message(f"/{key}", value)
        
        # Display frame
        cv2.imshow('Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(0.033)  # ~30 FPS
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Setup OSC client
    osc_client = setup_osc_client()
    
    # Start face detection and sending
    detect_faces_and_send(osc_client)
