import cv2
import mediapipe as mp
import numpy as np
from utils.config_loader import load_config
from detectors.eye_detector import eye_aspect_ratio
from detectors.mouth_detector import mouth_aspect_ratio
from detectors.yolo_detector import YoloDrowsinessDetector
from calibration.calibrator import EARCalibrator
from utils.draw_utils import draw_text, draw_contour
import time
import pygame

config = load_config()

EYE_AR_CONSEC_FRAMES = config['mediapipe_settings']['eye_ar_consec_frames']
MOUTH_AR_THRESH = config['mediapipe_settings']['mouth_ar_thresh']
MOUTH_AR_CONSEC_FRAMES = config['mediapipe_settings']['mouth_ar_consec_frames']
YOLO_CONF_THRESH = config['yolo_settings']['confidence_thresh']
YOLO_CONSEC_FRAMES = config['yolo_settings']['consec_frames']
CALIBRATION_FRAMES = config['calibration_settings']['calibration_frames']
EAR_CALIBRATION_FACTOR = config['calibration_settings']['ear_calibration_factor']

EYE_COUNTER = 0
MOUTH_COUNTER = 0
YOLO_COUNTER = 0
is_calibrated = False
calibration = EARCalibrator(CALIBRATION_FRAMES, EAR_CALIBRATION_FACTOR)
EYE_AR_THRESH = 0

# === Initialize pygame for audio playback ===
pygame.mixer.init()
alert_sound = pygame.mixer.Sound('sound//TrinhAiCham.wav')

def play_alert_sound():
    if not pygame.mixer.get_busy():
        alert_sound.play(loops=-1)

def stop_alert_sound():
    if pygame.mixer.get_busy():
        alert_sound.stop()

yolo_model = YoloDrowsinessDetector('assets/best.pt', YOLO_CONF_THRESH)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
MOUTH_IDXS = [61, 291, 13, 14]

print("[INFO] Mở webcam...")
cap = cv2.VideoCapture(0)
time.sleep(1.0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_results = face_mesh.process(frame_rgb)

    if mp_results.multi_face_landmarks:
        face_landmarks = mp_results.multi_face_landmarks[0]
        shape = np.array([[int(lm.x * w), int(lm.y * h)] for lm in face_landmarks.landmark], dtype=np.int32)

        leftEye = shape[LEFT_EYE_IDXS]
        rightEye = shape[RIGHT_EYE_IDXS]
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

        draw_contour(frame, leftEye)
        draw_contour(frame, rightEye)

        if not is_calibrated:
            draw_text(frame, "HIEN CHINH: NHIN THANG & MO MAT BINH THUONG", (10, 30), (0, 255, 255))
            if calibration.update(ear):
                EYE_AR_THRESH = calibration.get_threshold()
                is_calibrated = True
                print(f"[INFO] Hieu chinh hoan tat. Nguong EAR ca nhan: {EYE_AR_THRESH:.3f}")
        else:
            mouth = shape[MOUTH_IDXS]
            mar = mouth_aspect_ratio(mouth)
            draw_contour(frame, mouth)
            draw_text(frame, f"MAR: {mar:.2f}", (w - 150, 60))

            if ear < EYE_AR_THRESH: EYE_COUNTER += 1
            else: EYE_COUNTER = 0

            if mar > MOUTH_AR_THRESH: MOUTH_COUNTER += 1
            else: MOUTH_COUNTER = 0

        draw_text(frame, f"EAR: {ear:.2f}", (w - 150, 30))

    if is_calibrated:
        detected, coords, conf = yolo_model.detect(frame)
        if detected:
            YOLO_COUNTER += 1
            x1, y1, x2, y2 = coords
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            draw_text(frame, f"YOLO: Drowsy ({conf:.2f})", (x1, y1 - 10), (255, 0, 255))
        else:
            YOLO_COUNTER = 0

        if YOLO_COUNTER >= YOLO_CONSEC_FRAMES or EYE_COUNTER >= EYE_AR_CONSEC_FRAMES or MOUTH_COUNTER >= MOUTH_AR_CONSEC_FRAMES:
            draw_text(frame, "!!! CANH BAO BUON NGU !!!", (10, 50), (0, 0, 255), 1, 3)
            play_alert_sound()
        else:
            stop_alert_sound()

    cv2.imshow("Hybrid Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()