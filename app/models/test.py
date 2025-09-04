import cv2
from ultralytics import YOLO
import logging 

logger = logging.getLogger(__name__)

url = "rtsp://it:123456aA@192.168.2.66:554/cam/realmonitor?channel=1&subtype=1"
cam = cv2.VideoCapture(url)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

model = YOLO("yolov8n.pt")

while True:
    ret, img = cam.read()
    if not  ret:
        logger.info("There is Error about fream")
        break 

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    logger.info(f"there is gray:{gray}")

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    logger.info(f"There is faces:{faces}")

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y),(x+w, y+h),(255, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes =  eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh),(0, 127, 255), 2)

    results = model(img, classes=[0])
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(img, (x1,y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("IP Camera - Face & YOLO Detection", img)
    if cv2.waitKey(1)  & 0xFF == ord('q'):
        break



cam.release()
cv2.destroyAllWindows()