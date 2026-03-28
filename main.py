import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

# ===============================
# UTILITY FUNCTIONS
# ===============================

def load_sign_language_model(model_path='asl_model.h5'):
    """
    Load the pre-trained TensorFlow/Keras model for ASL classification.
    """
    print("[INFO] Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("[INFO] Model loaded successfully.")
    return model


def get_label_list():
    """
    Return the list of classification labels (A-Z).
    """
    return [chr(i) for i in range(65, 91)]  # ASCII A-Z


def initialize_camera():
    """
    Initialize webcam for capturing live video feed.
    """
    print("[INFO] Starting camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("[ERROR] Cannot open webcam")
    return cap


def setup_mediapipe():
    """
    Set up the MediaPipe Hands pipeline.
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils
    return hands, mp_draw


def extract_landmark_vector(hand_landmarks):
    """
    Extract normalized (x, y) values from 21 landmarks.
    Flatten into a 1D array.
    """
    return np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()


def predict_sign(model, vector, labels):
    """
    Predict the ASL letter from landmark vector.
    """
    if len(vector) != 42:
        return None, 0.0

    prediction = model.predict(np.expand_dims(vector, axis=0), verbose=0)
    class_id = np.argmax(prediction)
    confidence = float(prediction[0][class_id])

    return labels[class_id], confidence


def draw_info(frame, text, confidence, fps):
    """
    Draw prediction text, confidence score, and FPS on the frame.
    """
    cv2.putText(frame, f'Sign: {text}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.putText(frame, f'Confidence: {confidence:.2f}', (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.putText(frame, "Press 'q' to quit", (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)


# ===============================
# MAIN FUNCTION
# ===============================

def main():
    # Load resources
    model = load_sign_language_model()
    labels = get_label_list()
    cap = initialize_camera()
    hands, mp_draw = setup_mediapipe()

    prev_time = 0  # For FPS calculation

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("[WARN] Failed to read frame from camera.")
                break

            # Mirror the frame
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hands
            result = hands.process(img_rgb)

            prediction_text = "None"
            confidence_score = 0.0

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                    vector = extract_landmark_vector(hand_landmarks)
                    prediction_text, confidence_score = predict_sign(model, vector, labels)

            # FPS calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time

            # Draw info
            draw_info(frame, prediction_text, confidence_score, fps)

            # Show window
            cv2.imshow("Real-Time Sign Language Translator", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Quitting...")
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Resources released.")


# ===============================
# RUN
# ===============================

if __name__ == "__main__":
    main()
