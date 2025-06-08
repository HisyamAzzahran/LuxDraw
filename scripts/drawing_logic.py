import cv2
import numpy as np
import time
import easyocr
import os

def draw_text_with_background(frame, text, position, font, font_scale, text_color, bg_color, thickness=1, alpha=0.6):
    """Tambahkan teks dengan latar belakang transparan."""
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position

    # Gambar kotak latar belakang
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - text_height - baseline), (x + text_width, y + baseline), bg_color, -1)

    # transparansi
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Tambahkan teks
    cv2.putText(frame, text, position, font, font_scale, text_color, thickness)

def draw_progress_bar(frame, progress, position, size, color):
    """Bar progres horizontal dengan animasi."""
    x, y = position
    width, height = size
    overlay = frame.copy()

    # Gambar background bar
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (50, 50, 50), -1)

    # Gambar bar progres
    progress_width = int(min(progress, 1.0) * width)  # Pastikan progress maksimum 1.0
    cv2.rectangle(overlay, (x, y), (x + progress_width, y + height), color, -1)

    # Gabungkan dengan transparansi
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

class DrawingLogic:
    def __init__(self):
        self.canvas = None
        self.prev_x, self.prev_y = None, None
        self.color = (255, 255, 255)  # Warna awal: putih
        self.thickness = 5  # Ketebalan awal
        self.drawing_mode = False
        self.erasing_mode = False
        self.status_text = "Idle"  # Teks status
        self.idle_start_time = None  # Waktu saat idle dimulai
        self.has_processed = False  # Flag untuk memastikan analisis hanya dilakukan sekali
        self.icons = {
            "Idle": "icons/idle.png",
            "Drawing": "icons/drawing.png",
            "Erasing": "icons/erasing.png",
            "Processing": "icons/process_icon.png"
        }

        # Palet warna
        self.colors = [(255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Putih, Merah, Hijau, Biru
        self.color_names = ["White", "Red", "Green", "Blue"]
        self.selected_color_index = 0  # Indeks warna aktif

        # Background grid
        self.background = cv2.imread("backgrounds/grid.png")

    def initialize_canvas(self, frame_shape):
        """Inisialisasi canvas."""
        self.canvas = np.zeros(frame_shape, dtype=np.uint8)

    def draw(self, frame, index_finger):
        """Menggambar atau menghapus berdasarkan mode."""
        index_x, index_y = index_finger

        frame_height, frame_width, _ = frame.shape

        self.draw_color_palette(frame)

        if self.background is not None:
            background = cv2.resize(self.background, (frame_width, frame_height))
            frame = cv2.addWeighted(frame, 0.7, background, 0.3, 0)

        self.draw_color_palette(frame)

        # Mode menggambar
        if self.drawing_mode:
            if self.prev_x is not None and self.prev_y is not None:
                # Interpolasi linear untuk membuat garis
                for i in np.linspace(0, 1, num=10):  # 10 titik interpolasi
                    x = int(self.prev_x + i * (index_x - self.prev_x))
                    y = int(self.prev_y + i * (index_y - self.prev_y))
                    cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), self.color, self.thickness)
            self.prev_x, self.prev_y = index_x, index_y  # Perbarui koordinat

        # Mode menghapus
        elif self.erasing_mode:
            radius = 30
            cv2.circle(self.canvas, (index_x, index_y), radius, (0, 0, 0), -1)

        else:
            self.prev_x, self.prev_y = None, None  # Reset koordinat saat idle

        # Tampilkan status
        status_position = (50, frame_height - 30)
        draw_text_with_background(frame, f"Status: {self.status_text}", status_position, cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                    (0, 255, 0), (0, 0, 0), 2)

        # Hitungan detik Idle di kanan bawah
        if self.status_text == "Idle" and self.idle_start_time:
            elapsed_time = int(time.time() - self.idle_start_time)
            if elapsed_time >= 2:  # Timer mulai dari detik ke-2
                timer_position = (frame_width - 200, frame_height - 30)
                draw_progress_bar(frame, (elapsed_time - 1) / 7, timer_position, (150, 20), (0, 255, 255))
        # Tampilkan progress bar jika dalam mode Idle
        if self.status_text == "Idle" and self.idle_start_time:
            elapsed_time = time.time() - self.idle_start_time
            progress = elapsed_time / 7  # Hitung progress (0.0 hingga 1.0)

            # Pastikan progress tidak lebih dari 1.0
            if elapsed_time >= 7:
                progress = 1.0

            # Gambar bar progres di kanan bawah
            timer_position = (frame_width - 200, frame_height - 30)
            draw_progress_bar(frame, progress, timer_position, (150, 20), (0, 255, 255))
        # Menentukan ikon berdasarkan status
        icon = self.icons.get(self.status_text, None)
        if icon is not None:
            # Tampilkan ikon di pojok kanan atas
            self.overlay_icon(frame, icon, (frame_width - 100, 10))

    def overlay_icon(self, frame, icon_path, position):
        """Overlay ikon pada frame."""
        x, y = position
        icon_size = 50  # Ukuran ikon dalam piksel (lebar dan tinggi)

        # Muat ikon dari path
        icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
        if icon is None:
            print(f"Error: Ikon tidak ditemukan di {icon_path}")
            return

        # Resize ikon
        icon = cv2.resize(icon, (icon_size, icon_size))

        # Pastikan ikon memiliki channel alpha (transparansi)
        if icon.shape[2] != 4:
            print("Error: Ikon harus memiliki channel alpha (RGBA).")
            return

        # Ekstrak channel alpha untuk transparansi
        alpha_channel = icon[:, :, 3] / 255.0
        overlay = icon[:, :, :3]

        # Area target pada frame
        h, w, _ = frame.shape
        if x + icon_size > w or y + icon_size > h:
            print("Error: Ikon melebihi batas frame.")
            return

        roi = frame[y:y+icon_size, x:x+icon_size]

        # Gabungkan ikon dengan frame
        for c in range(3):  # Iterasi untuk setiap channel (B, G, R)
            roi[:, :, c] = (1 - alpha_channel) * roi[:, :, c] + alpha_channel * overlay[:, :, c]

        # Tempelkan hasil ke frame
        frame[y:y+icon_size, x:x+icon_size] = roi

    def draw_status_icon(self, frame, position):
        """Gambar ikon status berdasarkan mode aktif."""
        x, y = position
        icon_size = 40

        # Pilih ikon berdasarkan status
        if self.status_text == "Drawing":
            color = (0, 255, 0)  # Hijau untuk menggambar
        elif self.status_text == "Erasing":
            color = (0, 0, 255)  # Merah untuk menghapus
        else:  # Default Idle
            color = (255, 255, 255)  # Putih untuk Idle

        # Gambar lingkaran sebagai ikon
        cv2.circle(frame, (x + icon_size // 2, y + icon_size // 2), icon_size // 2, color, -1)

        # Tambahkan label kecil di bawah ikon
        label_position = (x, y + icon_size + 10)
        cv2.putText(frame, self.status_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def draw_color_palette(self, frame):
        """Gambar palet warna di bagian atas layar."""
        palette_height = 50
        for i, color in enumerate(self.colors):
            x1, y1 = i * 100, 0
            x2, y2 = (i + 1) * 100, palette_height
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            if i == self.selected_color_index:
                # Highlight warna aktif
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(frame, self.color_names[i], (x1 + 10, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def check_color_selection(self, landmarks):
        """Periksa apakah jari telunjuk berada di atas palet warna."""
        index_finger_tip = landmarks[8]  # Landmark jari telunjuk
        x = int(index_finger_tip.x * self.canvas.shape[1])
        y = int(index_finger_tip.y * self.canvas.shape[0])

        palette_height = 50
        if y < palette_height:  # Jika jari berada di area palet
            selected_index = x // 100  # Tentukan kolom berdasarkan posisi x
            if 0 <= selected_index < len(self.colors):  # Pastikan indeks valid
                self.selected_color_index = selected_index
                self.color = self.colors[self.selected_color_index]

    def toggle_modes(self, landmarks):
        """Mengubah mode berdasarkan posisi jari."""
        current_time = time.time()

        # Landmark jari telunjuk
        index_finger_tip = landmarks[8].y
        index_finger_base = landmarks[6].y

        # Mengepal: Semua jari lebih rendah dari landmark pangkalnya
        fist = all(landmarks[i].y > landmarks[i - 2].y for i in [8, 12, 16, 20])

        # Menunjuk: Hanya telunjuk yang terangkat lebih tinggi dari landmark pangkal
        pointing = (index_finger_tip < index_finger_base) and all(
            landmarks[i].y > landmarks[i - 2].y for i in [12, 16, 20]
        )

        # Mode menghapus (mengepal)
        if fist and not self.erasing_mode:
            self.drawing_mode = False
            self.erasing_mode = True
            self.status_text = "Erasing"
            self.idle_start_time = None  # Reset idle timer
            self.has_processed = False

        # Mode menggambar (menunjuk)
        elif pointing and not self.drawing_mode:
            self.drawing_mode = True
            self.erasing_mode = False
            self.status_text = "Drawing"
            self.idle_start_time = None  # Reset idle timer
            self.has_processed = False

        # Mode idle
        elif not fist and not pointing:
            self.drawing_mode = False
            self.erasing_mode = False
            if not self.idle_start_time:  # Mulai timer idle saat pertama kali masuk Idle
                self.idle_start_time = time.time()
                self.check_color_selection(landmarks)
            self.status_text = "Idle"

    def process_canvas(self):
        """Analisis gambar setelah idle 7 detik."""
        if self.idle_start_time:
            elapsed_time = int(time.time() - self.idle_start_time)

            if elapsed_time >= 7 and not self.has_processed:
                print("Idle for 7 seconds. Processing canvas...")
                self.status_text = "Processing..."
                self.has_processed = True

                # Preprocessing gambar untuk OCR
                processed_canvas = self.preprocess_for_ocr(self.canvas)

                # Gunakan EasyOCR untuk membaca teks
                import easyocr
                reader = easyocr.Reader(['en'], gpu=False)
                results = reader.readtext(processed_canvas, detail=0)

                self.detected_text = " ".join(results)  # Simpan teks yang terdeteksi
                print(f"Detected text: {self.detected_text}")

                # Evaluasi ekspresi matematika
                try:
                    result = eval(self.detected_text)
                    print(f"Expression: {self.detected_text} = {result}")
                    self.status_text = f"Result: {result}"
                except Exception as e:
                    print(f"Failed to evaluate expression: {self.detected_text}, Error: {e}")
                    self.status_text = "Invalid Expression"

                # Reset idle timer dan bersihkan kanvas
                self.idle_start_time = None
                self.clear_canvas()

    def get_result(self):
        """Mengembalikan hasil deteksi teks dan status."""
        return {
            "detected_text": self.detected_text,
            "status_text": self.status_text
        }

    def preprocess_for_ocr(self, canvas):
        """Preprocessing gambar untuk OCR."""
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        return resized

    def clear_canvas(self):
        """Clear Canvas."""
        self.canvas.fill(0)

    def change_color(self, distance):
        """Mengubah warna berdasarkan jarak antara dua titik."""
        if distance < 30:
            self.color = (0, 0, 255)  # Merah
        elif distance < 50:
            self.color = (0, 255, 0)  # Hijau
        else:
            self.color = (255, 255, 255)  # Putih

    def adjust_thickness(self, distance):
        """Menyesuaikan ketebalan garis berdasarkan jarak antara dua titik."""
        # Ketebalan minimum 1 dan maksimum 15
        self.thickness = max(1, min(15, int(distance / 10)))

    def save_drawing(self, image, filename="drawing.png"):
        if not os.path.exists("outputs"):
            os.makedirs("outputs")
        cv2.imwrite(f"outputs/{filename}", image)