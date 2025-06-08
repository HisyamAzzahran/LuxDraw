from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import base64
from drawing_logic import DrawingLogic

app = Flask(__name__)

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Inisialisasi DrawingLogic
drawing_logic = DrawingLogic()
is_canvas_initialized = False

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global is_canvas_initialized
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    # Decode a imagem de base64
    try:
        image_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': f'Error decoding image: {str(e)}'}), 400


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

    drawing_logic.process_canvas()
    combined_frame = cv2.addWeighted(frame, 1, drawing_logic.canvas, 1, 0)

    # Encode frame ke base64 untuk dikirim kembali ke frontend
    _, buffer = cv2.imencode('.jpg', combined_frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    result_data = drawing_logic.get_result()
    result_data['image'] = f'data:image/jpeg;base64,{jpg_as_text}'

    return jsonify(result_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))