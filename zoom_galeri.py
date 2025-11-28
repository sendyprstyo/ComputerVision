import cv2
import os
import numpy as np

# --- KONFIGURASI ---
FOLDER_GAMBAR = "gambar_galeri" # Nama folder tempat menyimpan gambar
EKSTENSI_YANG_DIBACA = ('.jpg', '.jpeg', '.png', '.bmp')

# --- FUNGSI BANTUAN UNTUK LOAD SEMUA GAMBAR ---
def muat_daftar_gambar(folder):
    daftar_file = []
    # Cek apakah folder ada
    if not os.path.exists(folder):
        print(f"ERROR: Folder '{folder}' tidak ditemukan!")
        print("Buat foldernya dan isi dengan beberapa gambar dulu.")
        return []
        
    for file in os.listdir(folder):
        if file.lower().endswith(EKSTENSI_YANG_DIBACA):
             daftar_file.append(os.path.join(folder, file))
    
    daftar_file.sort() # Urutkan berdasarkan nama
    return daftar_file

# --- FUNGSI INTI ZOOMING ---
def terapkan_zoom(image, zoom_factor):
    # Jika zoom factor 1.0, kembalikan gambar asli (tidak ada zoom)
    if zoom_factor <= 1.0:
        return image

    h, w = image.shape[:2] # Ambil tinggi dan lebar gambar asli

    # Hitung dimensi baru untuk area yang akan di-crop
    # Semakin besar zoom_factor, semakin kecil area yang di-crop
    new_h = int(h / zoom_factor)
    new_w = int(w / zoom_factor)

    # Hitung titik koordinat pojok kiri atas agar crop tepat di tengah
    start_y = (h - new_h) // 2
    start_x = (w - new_w) // 2

    # Lakukan cropping
    # img[y_awal:y_akhir, x_awal:x_akhir]
    cropped_image = image[start_y : start_y + new_h, 
                          start_x : start_x + new_w]

    # Resize kembali potongan gambar ke ukuran layar asli
    # Ini yang membuat efek "membesar"
    zoomed_image = cv2.resize(cropped_image, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return zoomed_image

# --- PROGRAM UTAMA ---
def main():
    daftar_gambar = muat_daftar_gambar(FOLDER_GAMBAR)
    
    if not daftar_gambar:
        print("Tidak ada gambar yang ditemukan. Keluar program.")
        return

    # Variabel Status
    current_index = 0       # Gambar mana yang sedang dilihat
    current_zoom = 1.0      # Level zoom awal (1.0 = normal)
    ZOOM_STEP = 0.1         # Seberapa banyak zoom bertambah/berkurang setiap tekan tombol
    MAX_ZOOM = 5.0          # Batas maksimal zoom

    cv2.namedWindow("Galeri Zoom", cv2.WINDOW_NORMAL)
    # Opsional: Set ukuran jendela agar tidak terlalu besar
    cv2.resizeWindow("Galeri Zoom", 800, 600) 

    print("--- KONTROL ---")
    print("Tekan 'n' : Gambar Berikutnya (Next)")
    print("Tekan 'p' : Gambar Sebelumnya (Previous)")
    print("Tekan '+' : Zoom In")
    print("Tekan '-' : Zoom Out")
    print("Tekan 'q' : Keluar")
    print("---------------")

    while True:
        # 1. Load gambar saat ini berdasarkan index
        path_gambar = daftar_gambar[current_index]
        img_asli = cv2.imread(path_gambar)

        if img_asli is None:
            print(f"Gagal membaca gambar: {path_gambar}")
            # Pindah ke gambar berikutnya jika gagal baca
            current_index = (current_index + 1) % len(daftar_gambar)
            continue

        # 2. Terapkan fungsi zoom pada gambar asli
        img_untuk_ditampilkan = terapkan_zoom(img_asli, current_zoom)

        # 3. Tambahkan teks informasi di layar
        info_teks = f"Img: {current_index+1}/{len(daftar_gambar)} | Zoom: {current_zoom:.1f}x | File: {os.path.basename(path_gambar)}"
        cv2.putText(img_untuk_ditampilkan, info_teks, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 4. Tampilkan gambar
        cv2.imshow("Galeri Zoom", img_untuk_ditampilkan)

        # 5. Handle Input Keyboard
        key = cv2.waitKey(1) & 0xFF

        # --- Kontrol Ganti Gambar ---
        if key == ord('n'): # Next image
            current_index = (current_index + 1) % len(daftar_gambar)
            current_zoom = 1.0 # Reset zoom saat ganti gambar
        
        elif key == ord('p'): # Previous image
            current_index = (current_index - 1 + len(daftar_gambar)) % len(daftar_gambar)
            current_zoom = 1.0 # Reset zoom saat ganti gambar

        # --- Kontrol Zoom ---
        elif key == ord('+') or key == ord('='): # Zoom In (tombol + biasanya butuh shift, jadi pakai = juga)
            current_zoom += ZOOM_STEP
            if current_zoom > MAX_ZOOM:
                current_zoom = MAX_ZOOM
                
        elif key == ord('-'): # Zoom Out
            current_zoom -= ZOOM_STEP
            if current_zoom < 1.0: # Jangan biarkan zoom lebih kecil dari 1.0 (mengecil)
                current_zoom = 1.0

        # --- Keluar ---
        elif key == ord('q') or key == 27: # 'q' atau ESC
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()