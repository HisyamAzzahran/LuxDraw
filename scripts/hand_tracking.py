import cv2
import mediapipe as mp
import numpy as np
from scripts.drawing_logic import DrawingLogic

def classify_gesture(hand_landmarks):
    # Logika sederhana untuk menganalisis gestur
    if hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y:  # Ibu jari di atas
        return "Thumbs Up"
    return "Unknown Gesture"

def hand_tracking():
    # Inisialisasi MediaPipe Hands
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        # Mengakses webcam
        cap = cv2.VideoCapture(0)
        
        # Pastikan kamera menggunakan resolusi yang tinggi
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Membuat jendela fullscreen
        cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)  # Jendela normal
        cv2.setWindowProperty("Hand Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Fullscreen

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame.")
                break

            # Proses frame dengan MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe memproses gambar RGB
            results = hands.process(rgb_frame)

            # Jika tangan terdeteksi
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    gesture = classify_gesture(hand_landmarks)
                    print(f"Detected gesture: {gesture}")

                    # Gambar landmarks di tangan
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            # Tampilkan frame di jendela fullscreen
            cv2.imshow("Hand Tracking", frame)

            # Keluar jika tombol 'q' ditekan
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Melepaskan resource kamera dan menutup semua jendela
        cap.release()
        cv2.destroyAllWindows()


def hand_tracking(save_canvas=False):
    import mediapipe as mp
    import cv2
    from scripts.drawing_logic import DrawingLogic

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

    drawing_logic = DrawingLogic()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access the webcam.")
        return {"error": "Webcam not accessible"}

    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read from webcam.")
        return {"error": "Cannot read from webcam"}

    drawing_logic.initialize_canvas(frame.shape)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

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

        # Proses kanvas setelah idle
        drawing_logic.process_canvas()

        combined_frame = cv2.addWeighted(frame, 1, drawing_logic.canvas, 1, 0)
        cv2.imshow("Trinity - Hand Tracking", combined_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC untuk keluar
            break

    cap.release()
    cv2.destroyAllWindows()

    # Simpan canvas jika diinginkan
    if save_canvas:
        canvas_path = "static/outputs/drawing_result.png"
        cv2.imwrite(canvas_path, drawing_logic.canvas)
    else:
        canvas_path = None

    # Ambil hasil dari DrawingLogic
    result_data = drawing_logic.get_result()
    result_data["canvas_path"] = canvas_path
    return result_data

