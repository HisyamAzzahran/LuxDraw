import cv2
import numpy as np
import time
import easyocr
import re
import base64


def solve_math_expression(expression_str):
    """Fungsi untuk membersihkan dan mengevaluasi ekspresi matematika dengan aman."""
    try:
        safe_expr = re.sub(r'[^0-9+\-*/(). ]', '', expression_str)
        safe_expr = safe_expr.replace('x', '*')
        if not safe_expr: return None
        return eval(safe_expr, {"__builtins__": {}}, {})
    except Exception:
        return None

class DrawingLogic:
    def __init__(self):
        """
        Menginisialisasi semua variabel.
        PERBAIKAN: Tidak ada lagi pemanggilan cv2.imread() di sini.
        """
        self.canvas = None
        self.prev_x, self.prev_y = None, None
        self.color = (255, 255, 255)
        self.thickness = 10
        self.drawing_mode = False
        self.erasing_mode = False
        self.status_text = "Idle"
        self.idle_start_time = None
        self.has_processed = False
        self.detected_text = ""
        self.result_text = ""
        
        # PERBAIKAN: Dihapus semua referensi ke file lokal
        self.background = None 
        self.icons = {}
        
        self.reader = easyocr.Reader(['en'], gpu=False)

        self.colors = [(255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0)]
        self.color_names = ["White", "Red", "Green", "Blue"]
        self.selected_color_index = 0

    def initialize_canvas(self, frame_shape):
        self.canvas = np.zeros(frame_shape, dtype=np.uint8)

    def draw_ui_elements(self, frame):
        """Menggambar semua elemen UI pada frame yang diberikan."""
        frame_height, frame_width, _ = frame.shape
        
        # Gambar palet warna
        palette_height = 50
        for i, color in enumerate(self.colors):
            x1, y1 = i * 60 + 10, 10
            x2, y2 = (i + 1) * 60, palette_height
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            if i == self.selected_color_index:
                cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (0, 255, 255), 3)

        # Gambar status teks
        status_position = (10, frame_height - 10)
        cv2.putText(frame, f"Mode: {self.status_text}", status_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Gambar progress bar
        if self.status_text == "Idle" and self.idle_start_time:
            elapsed_time = time.time() - self.idle_start_time
            progress = min(1.0, elapsed_time / 7.0)
            if progress > 0.01:
                cv2.rectangle(frame, (frame_width - 160, frame_height - 30), (frame_width - 10, frame_height - 10), (50, 50, 50), -1)
                cv2.rectangle(frame, (frame_width - 160, frame_height - 30), (frame_width - 160 + int(progress * 150), frame_height - 10), (0, 255, 255), -1)

    def draw_on_canvas(self, coords):
        if self.drawing_mode:
            if self.prev_x is not None:
                cv2.line(self.canvas, (self.prev_x, self.prev_y), coords, self.color, self.thickness)
            self.prev_x, self.prev_y = coords
        elif self.erasing_mode:
            cv2.circle(self.canvas, coords, 30, (0, 0, 0), -1)
        else:
            self.prev_x, self.prev_y = None, None
            
    def check_color_selection(self, landmarks):
        if not hasattr(self, 'canvas') or self.canvas is None: return
        index_finger_tip = landmarks[8]
        x = int(index_finger_tip.x * self.canvas.shape[1])
        y = int(index_finger_tip.y * self.canvas.shape[0])
        palette_height = 60
        if y < palette_height:
            selected_index = (x - 10) // 60
            if 0 <= selected_index < len(self.colors):
                self.selected_color_index = selected_index
                self.color = self.colors[self.selected_color_index]

    def toggle_modes(self, landmarks):
        if not landmarks:
            if not self.status_text.startswith("Result"):
                self.drawing_mode, self.erasing_mode = False, False
                if not self.idle_start_time: self.idle_start_time = time.time()
                self.status_text = "Idle"
            return
            
        tip_y = {i: landmarks[i].y for i in [8, 12, 16, 20]}
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
        if self.status_text == "Idle" and self.idle_start_time and not self.has_processed:
            if time.time() - self.idle_start_time >= 7:
                self.status_text = "Processing"
                self.has_processed = True
                
                if np.count_nonzero(self.canvas) == 0:
                    self.status_text = "Idle"
                    self.has_processed = False
                    self.idle_start_time = None
                    return
                
                processed_canvas = self.preprocess_for_ocr(self.canvas)
                results = self.reader.readtext(processed_canvas, detail=0)
                
                self.detected_text = " ".join(results)
                
                calculation_result = solve_math_expression(self.detected_text)
                if calculation_result is not None:
                    self.result_text = f"{calculation_result}"
                    self.status_text = f"Result: {calculation_result}"
                else:
                    self.result_text = "Error"
                    self.status_text = "Invalid Expression"

    def preprocess_for_ocr(self, canvas):
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        _, thresh = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)
        return thresh

    def get_response_data(self, frame):
        # Gabungkan frame kamera dengan kanvas gambar untuk tampilan
        if self.canvas is None:
            self.initialize_canvas(frame.shape)
        combined_frame = cv2.addWeighted(frame, 0.4, self.canvas, 1, 0)

        # Gambar UI di atasnya
        self.draw_ui_elements(combined_frame)

        # Encode gambar ke base64
        _, buffer = cv2.imencode('.jpg', combined_frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        return {
            'image': f'data:image/jpeg;base64,{jpg_as_text}',
            'status': self.status_text,
            'detected_text': self.detected_text,
            'result': self.result_text,
        }
    
    def clear_canvas(self):
        """Membersihkan kanvas dan mereset state."""
        if self.canvas is not None:
            self.canvas.fill(0)
        self.detected_text = ""
        self.result_text = ""
        self.status_text = "Idle"
        self.has_processed = False
        self.idle_start_time = None
        self.prev_x, self.prev_y = None, None