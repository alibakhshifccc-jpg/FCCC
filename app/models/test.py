import cv2
import numpy as np
import time
import os

# RTSP stream
url = "rtsp://it:123456aA@192.168.2.66:554/cam/realmonitor?channel=1&subtype=1" 
cap = cv2.VideoCapture(url)

# Define green color range in HSV
green_lower = np.array([40, 50, 50], dtype="uint8")
green_upper = np.array([80, 255, 255], dtype="uint8")

# Load face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Alternatively, you can use DNN for better accuracy
# prototxt_path = "deploy.prototxt"
# model_path = "res10_300x300_ssd_iter_140000.caffemodel"
# face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Directories to save frames
green_save_dir = "saved_green_frames"
face_save_dir = "saved_face_frames"

for directory in [green_save_dir, face_save_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

last_green_save_time = 0
last_face_save_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert to HSV for green detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, green_lower, green_upper)
    
    # Morphological operations
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find green objects
    green_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    green_detected = False
    for contour in green_contours:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Green Object", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            green_detected = True

            # Save green frame every 5 seconds
            if time.time() - last_green_save_time > 5:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"{green_save_dir}/green_object_{timestamp}.jpg", frame)
                last_green_save_time = time.time()
                print("Saved green object frame!")

    # Face detection using Haar Cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Alternative: Face detection using DNN (uncomment if you have the model files)
    """
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
    face_net.setInput(blob)
    detections = face_net.forward()
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x, y, x2, y2) = box.astype("int")
            cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Face: {confidence:.2f}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    """

    # Draw rectangles around detected faces and save
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # Save face frame every 3 seconds
        if time.time() - last_face_save_time > 3:
            face_roi = frame[y:y+h, x:x+w]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"{face_save_dir}/face_{timestamp}.jpg", face_roi)
            last_face_save_time = time.time()
            print("Saved face frame!")

    # Show the result
    cv2.imshow("Green Object and Face Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()