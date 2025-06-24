import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 1: MediaPipe FaceLandmarker modelini yükle
# 'face_landmarker_v2_with_blendshapes.task' dosyasının bu kodun çalıştığı dizinde olduğundan emin olun.
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,  # Blendshape'lere ihtiyacımız yok
    output_facial_transformation_matrixes=False, # Dönüşüm matrislerine ihtiyacımız yok
    num_faces=1  # Tek bir yüz algılaması yeterli
)
detector = vision.FaceLandmarker.create_from_options(options)

# Kamerayı başlat
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılamadı! Lütfen kamera bağlantınızı kontrol edin.")
    exit()

print("Kamera başlatıldı. Yüzünüzü kameraya tutun.")
print("'q' tuşuna basarak çıkabilirsiniz.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Kareden veri alınamadı. Çıkılıyor...")
        break

    # OpenCV (BGR) formatından MediaPipe (RGB) formatına dönüştür
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Yüz noktalarını algıla
    detection_result = detector.detect(mp_image)

    # Algılanan yüzler varsa işle
    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            # Yüz noktalarından yüzün sınırlayıcı kutusunu bul
            h, w, _ = frame.shape
            min_x, min_y = w, h
            max_x, max_y = 0, 0

            for landmark in face_landmarks:
                # MediaPipe normalleştirilmiş koordinatlar (0-1 arası) verir, piksel koordinatlarına dönüştür
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                min_x = min(min_x, cx)
                min_y = min(min_y, cy)
                max_x = max(max_x, cx)
                max_y = max(max_y, cy)

            # Sınırlayıcı kutuya biraz boşluk ekleyebiliriz (isteğe bağlı)
            padding = 20 # Piksel cinsinden boşluk
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(w, max_x + padding)
            max_y = min(h, max_y + padding)

            # Yüz bölgesini seç
            # Mozaik uygulamadan önce koordinatların geçerli olduğundan emin ol
            if max_x > min_x and max_y > min_y:
                face_roi = frame[min_y:max_y, min_x:max_x]

                # Mozaik derecesini ayarlamak için bölme faktörünü değiştirebilirsiniz
                # Daha küçük değerler (örneğin w_roi // 20) daha büyük mozaik blokları oluşturur.
                w_roi, h_roi, _ = face_roi.shape
                
                mosaic_factor = 25 # Mozaik büyüklüğünü kontrol eden faktör
                mosaic_w = w_roi // mosaic_factor
                mosaic_h = h_roi // mosaic_factor
                
                # Minimum boyutların 1 olmasını sağla
                if mosaic_w == 0: mosaic_w = 1
                if mosaic_h == 0: mosaic_h = 1

                # Yüz bölgesini küçült (pikselleşme için)
                small_face = cv2.resize(face_roi, (mosaic_w, mosaic_h), interpolation=cv2.INTER_LINEAR)
                
                # Küçültülmüş yüzü orijinal boyutuna geri büyüt (mozaik efekti için en yakın komşu interpolasyonu)
                mosaic_face = cv2.resize(small_face, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)

                # Mozaiklenmiş yüzü orijinal kareye geri yerleştir
                frame[min_y:max_y, min_x:max_x] = mosaic_face

    # İşlenmiş kareyi göster
    cv2.imshow('Yuz Mozaikleyici (MediaPipe)', frame)

    # 'q' tuşuna basıldığında döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()