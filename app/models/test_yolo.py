from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # مدل از پیش آموزش‌دیده
image = cv2.imread("sample.jpg")
results = model(image)
results.show()  # نمایش تصویر با کادرهای تشخیص