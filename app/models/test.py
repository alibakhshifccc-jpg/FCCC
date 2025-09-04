import cv2
import numpy as np
import time
import os
import threading

# RTSP stream
url = "rtsp://it:123456aA@192.168.2.66:554/cam/realmonitor?channel=1&subtype=1" 
cap = cv2.VideoCapture(url)

# Define green color range in HSV
green_lower = np.array([40, 50, 50], dtype="uint8")
green_upper = np.array([80, 255, 255], dtype="uint8")

# Load Haar Cascade face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Directories to save frames
green_save_dir = "saved_green_frames"
face_save_dir = "saved_face_frames"

for directory in [green_save_dir, face_save_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

last_green_save_time = 0
last_face_save_time = 0

frame_count = 0

# Function to save images in a separate thread
def save_frame(path, img):
    cv2.imwrite(path, img)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1
    if frame_count % 2 != 0:
        continue  # Process every other frame to increase speed

    # Resize for faster processing
    frame_resized = cv2.resize(frame, (640, 360))
    hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, green_lower, green_upper)

    # Fast morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find green objects
    green_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_detected = False
    for contour in green_contours:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)  # <-- اینجا درست شد
            # Map coordinates back to original frame
            scale_x = frame.shape[1] / frame_resized.shape[1]
            scale_y = frame.shape[0] / frame_resized.shape[0]
            x, y, w, h = int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Green Object", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            green_detected = True

            if time.time() - last_green_save_time > 5:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                threading.Thread(target=save_frame, args=(f"{green_save_dir}/green_object_{timestamp}.jpg", frame.copy())).start()
                last_green_save_time = time.time()
                print("Saved green object frame!")

    # Face detection
    gray_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_resized,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Map faces to original frame
    scale_x = frame.shape[1] / frame_resized.shape[1]
    scale_y = frame.shape[0] / frame_resized.shape[0]
    faces = [(int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y)) for (x, y, w, h) in faces]

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        if time.time() - last_face_save_time > 3:
            face_roi = frame[y:y+h, x:x+w].copy()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            threading.Thread(target=save_frame, args=(f"{face_save_dir}/face_{timestamp}.jpg", face_roi)).start()
            last_face_save_time = time.time()
            print("Saved face frame!")

    # Show result
    cv2.imshow("Green Object and Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
