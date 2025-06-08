import base64
import cv2
import mediapipe as mp
import numpy as np
import os
from flask import Flask, jsonify, request
from flask_cors import CORS

from drawing_logic import DrawingLogic

# --- Initialization ---
# Create the Flask app instance directly
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# Initialize MediaPipe and your custom drawing logic
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
drawing_logic = DrawingLogic()
is_canvas_initialized = False

# --- API Routes ---

@app.route('/')
def index():
    """A simple route to check if the backend is running."""
    return "LuxDraw Backend is running and ready!"

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """
    Receives a video frame, processes it for hand tracking and drawing,
    and returns the annotated frame.
    """
    global is_canvas_initialized
    
    # Get the image data from the request
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    try:
        # Decode the base64 image data
        image_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': f'Error decoding image: {str(e)}'}), 400

    if frame is None:
        return jsonify({'error': 'Failed to decode image, frame is None'}), 400

    # Initialize the canvas with the correct dimensions on the first frame
    if not is_canvas_initialized:
        drawing_logic.initialize_canvas(frame.shape)
        is_canvas_initialized = True

    # The frontend already mirrors the video, so we don't need to flip it again
    # frame = cv2.flip(frame, 1) # This line is commented out

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Process hand landmarks if detected
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
        # If no hands are detected, ensure the mode is set to Idle
        drawing_logic.toggle_modes([])

    # Check if it's time to process the canvas for text recognition
    drawing_logic.process_canvas()
    
    # Combine the camera frame with the drawing canvas
    combined_frame = cv2.addWeighted(frame, 0.6, drawing_logic.canvas, 1, 0)

    # Encode the final image back to base64 to send to the frontend
    _, buffer = cv2.imencode('.jpg', combined_frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    # Create the JSON response object for the frontend
    response_data = {
        'image': f'data:image/jpeg;base64,{jpg_as_text}',
        'status': drawing_logic.status_text,
        'detected_text': drawing_logic.detected_text
    }
    return jsonify(response_data)

@app.route('/clear', methods=['POST'])
def clear_drawing():
    """Endpoint to clear the drawing canvas."""
    drawing_logic.clear_canvas()
    return jsonify({'message': 'Canvas cleared successfully'}), 200

# This part is only for running the app locally, not used by Gunicorn on Railway
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)