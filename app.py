from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64
from drawing_logic import DrawingLogic

# Pindahkan inisialisasi ke dalam factory agar setiap worker punya instance sendiri
# Ini mencegah masalah state bersama antar request
def create_app():
    """Application Factory untuk membuat dan mengkonfigurasi instance Flask."""
    app = Flask(__name__)
    CORS(app)  # Mengaktifkan CORS untuk semua route

    # Inisialisasi komponen yang akan digunakan di dalam scope aplikasi
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils
    drawing_logic = DrawingLogic()
    is_canvas_initialized = False

    @app.route('/')
    def index():
        """Route sederhana untuk mengecek apakah server berjalan."""
        return "LuxDraw Backend is running and ready!"

    @app.route('/process_frame', methods=['POST'])
    def process_frame():
        """Menerima frame gambar, memprosesnya, dan mengembalikan hasilnya."""
        nonlocal is_canvas_initialized
        
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        try:
            # Decode gambar dari format base64
            image_data = base64.b64decode(data['image'].split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return jsonify({'error': f'Error decoding image: {str(e)}'}), 400

        if frame is None:
            return jsonify({'error': 'Failed to decode image, frame is None'}), 400

        # Inisialisasi canvas hanya sekali saat frame pertama diterima
        if not is_canvas_initialized or drawing_logic.canvas.shape[:2] != frame.shape[:2]:
            drawing_logic.initialize_canvas(frame.shape)
            is_canvas_initialized = True

        # Balik frame agar seperti cermin
        frame = cv2.flip(frame, 1)
        
        # Proses gambar dengan MediaPipe untuk deteksi tangan
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Jika tangan terdeteksi, lakukan logika menggambar
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Gambar overlay landmarks tangan
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_coords = (
                    int(index_finger_tip.x * frame.shape[1]),
                    int(index_finger_tip.y * frame.shape[0]),
                )
                
                drawing_logic.toggle_modes(hand_landmarks.landmark)
                drawing_logic.draw(frame, index_finger_coords)
        else:
            # Jika tidak ada tangan, pastikan mode kembali ke Idle
            drawing_logic.toggle_modes([])

        # Periksa apakah sudah waktunya memproses gambar untuk OCR
        drawing_logic.process_canvas_for_ocr()
        
        # Siapkan data respon untuk dikirim kembali ke frontend
        response_data = drawing_logic.get_response_data(frame)
        return jsonify(response_data)

    @app.route('/clear', methods=['POST'])
    def clear_drawing():
        """Endpoint untuk membersihkan kanvas gambar dari frontend."""
        drawing_logic.clear_canvas()
        return jsonify({'message': 'Canvas cleared successfully'}), 200

    return app

# Bagian ini hanya untuk testing lokal, Gunicorn tidak akan menggunakannya
if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)