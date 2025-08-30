import cv2
from cvzone.HandTrackingModule import HandDetector
import time

# Load DNN face detector
modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Hand detector
detector = HandDetector(maxHands=1, detectionCon=0.7)

cap = cv2.VideoCapture(0)  # یا rtsp بزار

last_save_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---- Face Detection (DNN) ----
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # ---- Hand Detection (cvzone/mediapipe) ----
    hands, img = detector.findHands(frame, draw=True)
    if hands:
        fingers = detector.fingersUp(hands[0])
        totalFingers = fingers.count(1)
        cv2.putText(frame, f"Fingers: {totalFingers}", (50,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 2)

        if totalFingers == 5 and time.time() - last_save_time > 5:
            cv2.imwrite(f"saved_frames/frame_{int(time.time())}.jpg", frame)
            last_save_time = time.time()
            print("Saved!")

    cv2.imshow("Enhanced Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
