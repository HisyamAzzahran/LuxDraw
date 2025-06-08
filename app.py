from flask import Flask, request, jsonify, Response
import cv2
import mediapipe as mp
import numpy as np
import base64
from drawing_logic import DrawingLogic

# Inisialisasi di luar untuk efisiensi
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
drawing_logic = DrawingLogic()
is_canvas_initialized = False

def create_app():
    app = Flask(__name__)

    @app.route('/')
    def index():
        return "LuxDraw Backend is running!"

    @app.route('/process_frame', methods=['POST'])
    def process_frame():
        global is_canvas_initialized
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        try:
            image_data = base64.b64decode(data['image'].split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return jsonify({'error': f'Error decoding image: {str(e)}'}), 400

        if frame is None:
            return jsonify({'error': 'Failed to decode image, frame is None'}), 400

        if not is_canvas_initialized:
            drawing_logic.initialize_canvas(frame.shape)
            is_canvas_initialized = True

        frame = cv2.flip(frame, 1)
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
            # Jika tidak ada tangan terdeteksi, anggap Idle
            drawing_logic.toggle_modes([]) # kirim list kosong untuk memicu mode Idle

        drawing_logic.process_canvas()
        
        # Gabungkan frame dari kamera dengan kanvas gambar
        combined_frame = cv2.addWeighted(frame, 0.5, drawing_logic.canvas, 1, 0)

        _, buffer = cv2.imencode('.jpg', combined_frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        # Kirim respon yang berisi gambar dan status real-time
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