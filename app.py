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
    app = Flask(__name__)
    CORS(app)

    # Inisialisasi per-worker
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils
    drawing_logic = DrawingLogic()
    is_canvas_initialized_for_worker = False

    @app.route('/')
    def index():
        return "LuxDraw Backend is running and ready!"

    @app.route('/process_frame', methods=['POST'])
    def process_frame():
        nonlocal is_canvas_initialized_for_worker
        
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'Invalid request: No image data provided'}), 400

        try:
            image_data = base64.b64decode(data['image'].split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return jsonify({'error': f'Error decoding image: {str(e)}'}), 400

        if frame is None:
            return jsonify({'error': 'Failed to decode image, frame is None'}), 400

        if not is_canvas_initialized_for_worker:
            drawing_logic.initialize_canvas(frame.shape)
            is_canvas_initialized_for_worker = True

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_coords = (
                    int(index_finger_tip.x * frame.shape[1]),
                    int(index_finger_tip.y * frame.shape[0]),
                )
                drawing_logic.toggle_modes(hand_landmarks.landmark)
                drawing_logic.draw(frame, index_finger_coords)
        else:
            drawing_logic.toggle_modes([])

        drawing_logic.process_canvas()
        
        combined_frame = cv2.addWeighted(frame, 0.6, drawing_logic.canvas, 1, 0)

        _, buffer = cv2.imencode('.jpg', combined_frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        response_data = {
            'image': f'data:image/jpeg;base64,{jpg_as_text}',
            'status': drawing_logic.status_text,
            'detected_text': drawing_logic.detected_text
        }
        return jsonify(response_data)

    @app.route('/clear', methods=['POST'])
    def clear_drawing():
        drawing_logic.clear_canvas()
        return jsonify({'message': 'Canvas cleared successfully'}), 200

    return app

# Baris ini hanya untuk menjalankan di komputer lokal, Gunicorn tidak akan menggunakannya
if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)