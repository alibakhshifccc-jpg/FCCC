import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import matplotlib.pyplot as plt

class FaceTrainer:
    def __init__(self, dataset_path="dataset"):
        self.dataset_path = dataset_path
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_encoder = LabelEncoder()
        self.model_path = "face_model.yml"
        self.label_encoder_path = "label_encoder.pkl"
        
    def prepare_data(self):
        faces = []
        labels = []
        label_names = []
        
        # لیست تمام پوشه‌ها (افراد)
        people = [name for name in os.listdir(self.dataset_path) 
                 if os.path.isdir(os.path.join(self.dataset_path, name))]
        
        for i, person in enumerate(people):
            person_path = os.path.join(self.dataset_path, person)
            
            # خواندن تمام تصاویر هر شخص
            for image_name in os.listdir(person_path):
                if image_name.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_path, image_name)
                    
                    # خواندن و پیش‌پردازش تصویر
                    image = cv2.imread(image_path)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    # تشخیص چهره (اگر قبلاً تشخیص داده نشده)
                    face_cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    for (x, y, w, h) in faces_detected:
                        face_roi = gray[y:y+h, x:x+w]
                        face_resized = cv2.resize(face_roi, (100, 100))
                        
                        faces.append(face_resized)
                        labels.append(i)
                        label_names.append(person)
        
        return np.array(faces), np.array(labels), label_names
    
    def train_model(self):
        print("آماده‌سازی داده‌ها...")
        faces, labels, label_names = self.prepare_data()
        
        if len(faces) == 0:
            print("هیچ تصویری برای آموزش پیدا نشد!")
            return False
        
        print(f"تعداد تصاویر آموزشی: {len(faces)}")
        print(f"تعداد افراد: {len(set(labels))}")
        
        # کدگذاری برچسب‌ها
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # تقسیم داده به آموزش و تست
        X_train, X_test, y_train, y_test = train_test_split(
            faces, encoded_labels, test_size=0.2, random_state=42)
        
        print("آموزش مدل...")
        # آموزش مدل LBPH
        self.face_recognizer.train(X_train, y_train)
        
        # ذخیره مدل
        self.face_recognizer.save(self.model_path)
        
        # ذخیره label encoder
        with open(self.label_encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print("مدل با موفقیت آموزش داده و ذخیره شد!")
        return True
    
    def test_model(self, test_image_path):
        # بارگذاری مدل آموزش‌دیده
        self.face_recognizer.read(self.model_path)
        
        with open(self.label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # تست روی یک تصویر جدید
        image = cv2.imread(test_image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (100, 100))
            
            label, confidence = self.face_recognizer.predict(face_resized)
            person_name = self.label_encoder.inverse_transform([label])[0]
            
            print(f"شناسایی شده: {person_name} - اطمینان: {confidence}")
            
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, f"{person_name} ({confidence:.2f})", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow('Test Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# نحوه استفاده
if __name__ == "__main__":
    # ۱. ایجاد نمونه از کلاس
    trainer = FaceTrainer("dataset")
    
    # ۲. آموزش مدل
    trainer.train_model()
    
    # ۳. تست مدل (اختیاری)
    # trainer.test_model("test_image.jpg")