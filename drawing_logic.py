import cv2
import numpy as np
import time
import easyocr
import os

def draw_text_with_background(frame, text, position, font, font_scale, text_color, bg_color, thickness=1, alpha=0.6):
    """Tambahkan teks dengan latar belakang transparan."""
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - text_height - baseline), (x + text_width, y + baseline), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, position, font, font_scale, text_color, thickness)

def draw_progress_bar(frame, progress, position, size, color):
    """Bar progres horizontal dengan animasi."""
    x, y = position
    width, height = size
    overlay = frame.copy()

    cv2.rectangle(overlay, (x, y), (x + width, y + height), (50, 50, 50), -1)
    progress_width = int(min(progress, 1.0) * width)
    cv2.rectangle(overlay, (x, y), (x + progress_width, y + height), color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

class DrawingLogic:
    def __init__(self):
        self.canvas = None
        self.prev_x, self.prev_y = None, None
        self.color = (255, 255, 255)
        self.thickness = 5
        self.drawing_mode = False
        self.erasing_mode = False
        self.status_text = "Idle"
        self.idle_start_time = None
        self.has_processed = False
        self.detected_text = "" # Inisialisasi agar selalu ada

        # Nonaktifkan pemuatan ikon dan background untuk stabilitas di server
        self.icons = {}
        self.background = None

        self.colors = [(255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0)]
        self.color_names = ["White", "Red", "Green", "Blue"]
        self.selected_color_index = 0

    def initialize_canvas(self, frame_shape):
        self.canvas = np.zeros(frame_shape, dtype=np.uint8)

    def draw(self, frame, index_finger):
        index_x, index_y = index_finger
        frame_height, frame_width, _ = frame.shape

        self.draw_color_palette(frame)

        # Fungsionalitas background dinonaktifkan
        # if self.background is not None:
        #     ...

        if self.drawing_mode:
            if self.prev_x is not None and self.prev_y is not None:
                for i in np.linspace(0, 1, num=10):
                    x = int(self.prev_x + i * (index_x - self.prev_x))
                    y = int(self.prev_y + i * (index_y - self.prev_y))
                    cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), self.color, self.thickness)
            self.prev_x, self.prev_y = index_x, index_y
        elif self.erasing_mode:
            radius = 30
            cv2.circle(self.canvas, (index_x, index_y), radius, (0, 0, 0), -1)
        else:
            self.prev_x, self.prev_y = None, None

        status_position = (50, frame_height - 30)
        draw_text_with_background(frame, f"Status: {self.status_text}", status_position, cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), (0, 0, 0), 2)

        if self.status_text == "Idle" and self.idle_start_time:
            elapsed_time = time.time() - self.idle_start_time
            progress = min(1.0, elapsed_time / 7.0)
            timer_position = (frame_width - 200, frame_height - 30)
            draw_progress_bar(frame, progress, timer_position, (150, 20), (0, 255, 255))
        
        # Fungsionalitas ikon dinonaktifkan
        # icon = self.icons.get(self.status_text, None)
        # if icon is not None:
        #     self.overlay_icon(frame, icon, (frame_width - 100, 10))

    def draw_color_palette(self, frame):
        palette_height = 50
        for i, color in enumerate(self.colors):
            x1, y1 = i * 100, 0
            x2, y2 = (i + 1) * 100, palette_height
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            if i == self.selected_color_index:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(frame, self.color_names[i], (x1 + 10, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def check_color_selection(self, landmarks):
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
        index_finger_tip_y = landmarks[8].y
        index_finger_base_y = landmarks[6].y
        fist = all(landmarks[i].y > landmarks[i - 2].y for i in [8, 12, 16, 20])
        pointing = (index_finger_tip_y < index_finger_base_y) and all(landmarks[i].y > landmarks[i - 2].y for i in [12, 16, 20])

        if fist and not self.erasing_mode:
            self.drawing_mode, self.erasing_mode = False, True
            self.status_text = "Erasing"
            self.idle_start_time = None
            self.has_processed = False
        elif pointing and not self.drawing_mode:
            self.drawing_mode, self.erasing_mode = True, False
            self.status_text = "Drawing"
            self.idle_start_time = None
            self.has_processed = False
        elif not fist and not pointing:
            self.drawing_mode, self.erasing_mode = False, False
            if not self.idle_start_time:
                self.idle_start_time = time.time()
                self.check_color_selection(landmarks)
            self.status_text = "Idle"

    def process_canvas(self):
        if self.status_text == "Idle" and self.idle_start_time:
            elapsed_time = time.time() - self.idle_start_time
            if elapsed_time >= 7 and not self.has_processed:
                print("Idle for 7 seconds. Processing canvas...")
                self.status_text = "Processing"
                self.has_processed = True
                
                processed_canvas = self.preprocess_for_ocr(self.canvas)
                reader = easyocr.Reader(['en'], gpu=False)
                results = reader.readtext(processed_canvas, detail=0)
                self.detected_text = " ".join(results)
                print(f"Detected text: {self.detected_text}")
                
                try:
                    # Ganti 'x' atau simbol perkalian lain dengan '*'
                    math_expr = self.detected_text.lower().replace('x', '*')
                    result = eval(math_expr)
                    print(f"Expression: {math_expr} = {result}")
                    self.status_text = f"Result: {result}"
                except Exception as e:
                    print(f"Failed to evaluate expression: {self.detected_text}, Error: {e}")
                    self.status_text = "Invalid Expression"
                
                # Jangan reset idle timer & clear canvas agar hasilnya bisa dilihat
                # self.idle_start_time = None
                # self.clear_canvas()

    def preprocess_for_ocr(self, canvas):
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        return resized

    def clear_canvas(self):
        if self.canvas is not None:
            self.canvas.fill(0)
        self.detected_text = ""
        self.status_text = "Idle"
        self.has_processed = False
        self.idle_start_time = None