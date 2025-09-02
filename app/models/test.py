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

# Directory to save frames
save_dir = "saved_green_frames"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

last_save_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV and create mask for green color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, green_lower, green_upper)
    
    # Morphological operations to reduce noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours of green objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # filter small areas
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Green Bottle", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Save frame every 5 seconds
            if time.time() - last_save_time > 5:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"{save_dir}/green_bottle_{timestamp}.jpg", frame)
                last_save_time = time.time()
                print("Saved green bottle frame!")

    # Show the result
    cv2.imshow("Green Bottle Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
