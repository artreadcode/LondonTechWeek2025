import cv2
import numpy as np
from pythonosc.udp_client import SimpleUDPClient

client = SimpleUDPClient("127.0.0.1", 8000)
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for i in range(10):
        if i < len(faces):
            (x, y, w, h) = faces[i]
            cx = x + w // 2
            cy = y + h // 2
            client.send_message(f"/face{i}_x", cx / frame.shape[1])
            client.send_message(f"/face{i}_y", 1.0 - (cy / frame.shape[0]))
        else:
            client.send_message(f"/face{i}_x", 0.0)
            client.send_message(f"/face{i}_y", 0.0)