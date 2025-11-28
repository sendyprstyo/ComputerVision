import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Inisialisasi Warna ---
# Daftar warna BGR yang akan digunakan
COLORS = [
    (255, 0, 0),    # Biru (Kode default di kode asli Anda)
    (0, 0, 255),    # Merah
    (0, 255, 0),    # Hijau
    (0, 255, 255)   # Kuning
]
color_index = 0

# --- Setup MediaPipe Tasks ---
# PASTIKAN JALUR INI BENAR!
model_path = r'D:\VS-TUGAS\hair_segmenter.tflite' 

# Opsi dasar untuk model TFLite
base_options = python.BaseOptions(model_asset_path=model_path)
# Opsi untuk Image Segmenter
options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)

# Inisialisasi Webcam
cap = cv2.VideoCapture(0)

# --- Proses Segmentasi ---
with vision.ImageSegmenter.create_from_options(options) as segmenter:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mendapatkan warna saat ini
        current_color = COLORS[color_index % len(COLORS)]

        # cv2.imshow("Input", frame) # Opsional: menampilkan frame input

        # 1. Konversi Frame ke Format MediaPipe
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # 2. Lakukan Segmentasi
        segmentation_result = segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask.numpy_view()

        # 3. Pemrosesan dan Blending (Pewarnaan)
        
        frame_height, frame_width, _ = frame.shape
        
        # Ubah ukuran masker agar sesuai dengan resolusi frame
        category_mask = cv2.resize(category_mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

        # Buat 'condition' untuk area rambut (masker boolean 3-channel)
        condition = np.stack((category_mask,) * 3, axis=-1) > 0.2

        # *** MODIFIKASI: Gunakan warna saat ini dari array COLORS ***
        hair_tint = np.full(frame.shape, current_color, dtype=np.uint8) 
        
        # Blending menggunakan np.where
        hair_region = np.where(condition, cv2.addWeighted(frame, 0.5, hair_tint, 0.5, 0), frame)

        # 4. Tampilkan Output
        # Tambahkan teks untuk menunjukkan tombol 'c'
        cv2.putText(hair_region, f"Warna BG: {current_color} (Tekan 'c')", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
        cv2.imshow("Output", hair_region)

        # 5. Kontrol Keluar dan Ganti Warna
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Ganti warna rambut ke warna berikutnya
            color_index += 1

# --- Pembersihan ---
cap.release()
cv2.destroyAllWindows()