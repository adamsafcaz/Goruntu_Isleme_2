import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


try:
    base_options = python.BaseOptions(model_asset_path='C:/Users/a9912/Downloads/face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)
except Exception as e:
    print(f"MediaPipe FaceLandmarker modeli yüklenirken hata oluştu: {e}")
    print("Lütfen 'face_landmarker_v2_with_blendshapes.task' dosyasının doğru yolda olduğundan emin olun.")
    exit()


cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Kamera açılamadı! Lütfen kamera bağlantınızı veya izinleri kontrol edin.")
    exit()

print("Kamera başlatıldı. Yüzünüzü kameraya tutun.")
print("'q' tuşuna basarak çıkabilirsiniz.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Kareden veri alınamadı, çıkılıyor...")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

   
    detection_result = detector.detect(mp_image)

    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
           
            h, w, _ = frame.shape
            min_x, min_y = w, h
            max_x, max_y = 0, 0

            for landmark in face_landmarks:
                
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                min_x = min(min_x, cx)
                min_y = min(min_y, cy)
                max_x = max(max_x, cx)
                max_y = max(max_y, cy)

            
            padding = 15  
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(w, max_x + padding)
            max_y = min(h, max_y + padding)

            
            if max_x > min_x and max_y > min_y:
                face_roi = frame[min_y:max_y, min_x:max_x]

              
                mosaic_factor = 25 

                
                h_roi, w_roi, _ = face_roi.shape 
              
              
                mosaic_w = max(1, w_roi // mosaic_factor)
                mosaic_h = max(1, h_roi // mosaic_factor)
                
                small_face = cv2.resize(face_roi, (mosaic_w, mosaic_h), interpolation=cv2.INTER_LINEAR)
              
             
                mosaic_face = cv2.resize(small_face, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)

             
                frame[min_y:max_y, min_x:max_x] = mosaic_face

    
    cv2.imshow('Yuz Mozaikleyici (MediaPipe Task Dosyasi)', frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()