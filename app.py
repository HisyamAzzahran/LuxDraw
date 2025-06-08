from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64
from drawing_logic import DrawingLogic

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
        return "LuxDraw Backend is running and ready!"

    @app.route('/process_frame', methods=['POST'])
    def process_frame():
        nonlocal is_canvas_initialized
        
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        try:
            image_data = base64.b64decode(data['image'].split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return jsonify({'error': f'Error decoding image: {str(e)}'}), 400

        if frame is None:
            return jsonify({'error': 'Failed to decode image, frame is None'}), 400

        if not is_canvas_initialized or (hasattr(drawing_logic.canvas, 'shape') and drawing_logic.canvas.shape[:2] != frame.shape[:2]):
            drawing_logic.initialize_canvas(frame.shape)
            is_canvas_initialized = True

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Balik frame secara horizontal agar seperti cermin
        frame = cv2.flip(frame, 1)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_coords = (
                    int(index_finger_tip.x * frame.shape[1]),
                    int(index_finger_tip.y * frame.shape[0]),
                )
                drawing_logic.toggle_modes(hand_landmarks.landmark)
                drawing_logic.draw_on_canvas(index_finger_coords)
        else:
            drawing_logic.toggle_modes([])

        drawing_logic.process_canvas_for_ocr()
        
        response_data = drawing_logic.get_response_data(frame)
        return jsonify(response_data)

    @app.route('/clear', methods=['POST'])
    def clear_drawing():
        drawing_logic.clear_canvas()
        return jsonify({'message': 'Canvas cleared successfully'}), 200

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)