import cv2
import numpy as np
import time
import os


url = "rtsp://it:123456aA@192.168.2.66:554/cam/realmonitor?channel=1&subtype=1" 
cap = cv2.VideoCapture(url)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


skin_lower = np.array([0, 20, 70], dtype="uint8")
skin_upper = np.array([20, 255, 255], dtype="uint8")


save_dir = "saved_frames"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

frame_count = 0
last_save_time = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 2 != 0: 
        continue


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5) 


    faces = face_cascade.detectMultiScale(
        gray_small,
        scaleFactor=1.3,
        minNeighbors=6,
        minSize=(60, 60)
    )


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
    
    skin_mask = cv2.erode(skin_mask, None, iterations=2)
    skin_mask = cv2.dilate(skin_mask, None, iterations=2)


    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    hand_detected = False
    finger_count_total = 0
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)
            
            finger_count = 0
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])
                    if d > 1000: 
                        finger_count += 1

                        cv2.circle(frame, far, 5, (0, 0, 255), -1)


            finger_count_total = finger_count + 1 if finger_count > 0 else 0
            
            if 3 <= finger_count <= 5:  
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"Object ({finger_count_total} )", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                hand_detected = True


                if finger_count_total == 5 and time.time() - last_save_time > 5: 
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"{save_dir}/hand_5fingers_{timestamp}.jpg", frame)
                    last_save_time = time.time()
                    print("save!")


    cv2.putText(frame, f"Faces: {len(faces)} | Fingers: {finger_count_total}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


    for (x, y, w, h) in faces:
        x, y, w, h = x*2, y*2, w*2, h*2
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Human (Face)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


    cv2.imshow("Enhanced Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()