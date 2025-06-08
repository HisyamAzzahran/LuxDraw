import cv2
import numpy as np
import time
import easyocr
import re

# Fungsi untuk membersihkan dan mengevaluasi ekspresi matematika dengan aman
def solve_math_expression(expression_str):
    try:
        # Menghapus semua karakter yang tidak diizinkan, hanya menyisakan angka, operator, dan tanda kurung
        safe_expr = re.sub(r'[^0-9+\-*/(). ]', '', expression_str)
        # Mengganti 'x' dengan '*' untuk perkalian
        safe_expr = safe_expr.replace('x', '*')
        
        # Mengevaluasi ekspresi dengan aman, tanpa akses ke fungsi built-in berbahaya
        result = eval(safe_expr, {"__builtins__": {}}, {})
        return result
    except (SyntaxError, NameError, TypeError, ZeroDivisionError, ValueError) as e:
        print(f"Error evaluating expression '{expression_str}': {e}")
        return None

class DrawingLogic:
    def __init__(self):
        """
        Menginisialisasi semua variabel dan model yang diperlukan.
        Pemuatan model EasyOCR hanya dilakukan sekali untuk efisiensi.
        """
        self.canvas = None
        self.prev_x, self.prev_y = None, None
        self.color = (255, 255, 255)  # Warna default: Putih
        self.thickness = 10  # Ketebalan garis
        self.drawing_mode = False
        self.erasing_mode = False
        self.status_text = "Idle"
        self.idle_start_time = None
        self.has_processed = False
        self.detected_text = ""
        self.result_text = ""
        
        # Inisialisasi pembaca OCR sekali saja untuk performa
        # GPU dinonaktifkan karena tidak tersedia di lingkungan server umum
        self.reader = easyocr.Reader(['en'], gpu=False)

        self.colors = [(255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0)] # Putih, Merah, Hijau, Biru
        self.color_names = ["White", "Red", "Green", "Blue"]
        self.selected_color_index = 0

    def initialize_canvas(self, frame_shape):
        """Membuat canvas kosong sesuai dengan ukuran frame video."""
        self.canvas = np.zeros(frame_shape, dtype=np.uint8)

    def draw(self, frame, index_finger):
        """Menggambar di kanvas berdasarkan mode (menggambar atau menghapus)."""
        index_x, index_y = index_finger
        
        if self.drawing_mode:
            if self.prev_x is not None:
                cv2.line(self.canvas, (self.prev_x, self.prev_y), (index_x, index_y), self.color, self.thickness)
            self.prev_x, self.prev_y = index_x, index_y
        elif self.erasing_mode:
            cv2.circle(self.canvas, (index_x, index_y), 30, (0, 0, 0), -1)
        else:
            self.prev_x, self.prev_y = None, None

    def draw_color_palette(self, frame):
        """Menggambar UI palet warna di bagian atas frame."""
        palette_height = 50
        for i, color in enumerate(self.colors):
            x1, y1 = i * 100, 0
            x2, y2 = (i + 1) * 100, palette_height
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            if i == self.selected_color_index:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 4) # Highlight

    def check_color_selection(self, landmarks):
        """Memeriksa apakah jari menyentuh salah satu palet warna."""
        if not self.canvas: return
        index_finger_tip = landmarks[8]
        x = int(index_finger_tip.x * self.canvas.shape[1])
        y = int(index_finger_tip.y * self.canvas.shape[0])
        palette_height = 50
        if y < palette_height:
            selected_index = x // 100
            if 0 <= selected_index < len(self.colors):
                self.selected_color_index = selected_index
                self.color = self.colors[self.selected_color_index]

    def toggle_modes(self, landmarks):
        """Mengubah status (Idle, Drawing, Erasing) berdasarkan gestur tangan."""
        if not landmarks:
            if not self.status_text.startswith("Result"):
                self.drawing_mode, self.erasing_mode = False, False
                if not self.idle_start_time: self.idle_start_time = time.time()
                self.status_text = "Idle"
            return
            
        tip_y = {i: landmarks[i].y for i in [4, 8, 12, 16, 20]}
        pip_y = {i: landmarks[i-2].y for i in [8, 12, 16, 20]}
        
        is_fist = all(tip_y[i] > pip_y[i] for i in [8, 12, 16, 20])
        is_pointing = (tip_y[8] < pip_y[8]) and all(tip_y[i] > pip_y[i] for i in [12, 16, 20])
        
        if is_fist:
            self.drawing_mode, self.erasing_mode, self.status_text = False, True, "Erasing"
            self.idle_start_time, self.has_processed = None, False
        elif is_pointing:
            self.drawing_mode, self.erasing_mode, self.status_text = True, False, "Drawing"
            self.idle_start_time, self.has_processed = None, False
        else:
            self.drawing_mode, self.erasing_mode = False, False
            if not self.status_text.startswith("Result") and self.status_text != "Processing":
                if not self.idle_start_time:
                    self.idle_start_time = time.time()
                    self.check_color_selection(landmarks)
                self.status_text = "Idle"

    def process_canvas_for_ocr(self):
        """Memproses gambar di kanvas menggunakan OCR jika sudah idle cukup lama."""
        if self.status_text == "Idle" and self.idle_start_time and not self.has_processed:
            if time.time() - self.idle_start_time >= 7:
                self.status_text = "Processing..."
                self.has_processed = True
                
                # Cek jika kanvas kosong
                if np.count_nonzero(self.canvas) == 0:
                    print("Kanvas kosong, tidak ada yang diproses.")
                    self.status_text = "Idle" # Kembali ke idle
                    self.idle_start_time = None
                    self.has_processed = False
                    return

                # Pre-processing untuk OCR
                gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
                inverted = cv2.bitwise_not(gray_canvas)
                _, thresh = cv2.threshold(inverted, 50, 255, cv2.THRESH_BINARY)
                
                # Baca teks
                results = self.reader.readtext(thresh, detail=0)
                self.detected_text = " ".join(results)
                print(f"Teks terdeteksi: {self.detected_text}")
                
                # Hitung hasil
                calculation_result = solve_math_expression(self.detected_text)
                if calculation_result is not None:
                    self.result_text = f"{calculation_result}"
                    self.status_text = "Result"
                else:
                    self.result_text = "Error"
                    self.status_text = "Invalid Expression"

    def get_response_data(self, frame):
        """Menyiapkan data yang akan dikirim sebagai JSON ke frontend."""
        # Gabungkan frame kamera dengan kanvas gambar untuk tampilan
        combined_frame = cv2.addWeighted(frame, 0.4, self.canvas, 1, 0)

        # Gambar UI di atasnya
        self.draw_color_palette(combined_frame)
        status_position = (10, combined_frame.shape[0] - 10)
        draw_text_with_background(combined_frame, f"Mode: {self.status_text}", status_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), (0,0,0,128))

        if self.status_text == "Idle" and self.idle_start_time:
            elapsed_time = time.time() - self.idle_start_time
            progress = min(1.0, elapsed_time / 7.0)
            if progress > 0.01:
                timer_position = (combined_frame.shape[1] - 160, combined_frame.shape[0] - 30)
                draw_progress_bar(combined_frame, progress, timer_position, (150, 20), (0, 255, 255))
        
        # Encode gambar ke base64
        _, buffer = cv2.imencode('.jpg', combined_frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        return {
            'image': f'data:image/jpeg;base64,{jpg_as_text}',
            'status': self.status_text,
            'detected_text': self.detected_text,
            'result': self.result_text,
        }
