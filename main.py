# ===============================
# 📌 main.py: Hybrid Drowsiness + Simple Nodding Detection (BBox Y)
# ===============================

import cv2
import numpy as np
import mediapipe as mp
import time
import pygame
from collections import deque
from ultralytics import YOLO
from utils.config_loader import load_config
from detectors.eye_detector import eye_aspect_ratio
from detectors.mouth_detector import mouth_aspect_ratio
from detectors.yolo_detector import YoloDrowsinessDetector
from calibration.calibrator import EARCalibrator
from utils.draw_utils import draw_text, draw_contour

# --- Load configuration ---
config = load_config()
EYE_AR_CONSEC_FRAMES   = config['mediapipe_settings']['eye_ar_consec_frames']
MOUTH_AR_THRESH        = config['mediapipe_settings']['mouth_ar_thresh']
MOUTH_AR_CONSEC_FRAMES = config['mediapipe_settings']['mouth_ar_consec_frames']
YOLO_CONF_THRESH       = config['yolo_settings']['confidence_thresh']
YOLO_CONSEC_FRAMES     = config['yolo_settings']['consec_frames']
CALIBRATION_FRAMES     = config['calibration_settings']['calibration_frames']
EAR_CALIBRATION_FACTOR = config['calibration_settings']['ear_calibration_factor']

# --- State counters ---
EYE_COUNTER   = 0
MOUTH_COUNTER = 0
YOLO_COUNTER  = 0
is_calibrated = False

# --- Simple nodding detection (bbox center Y) ---
CALIBRATION_FRAMES_NOD = 30  # frames to calibrate baseline
THRESHOLD_NOD_PX      = 20   # px deviation for nod
baseline_list         = []
baseline_center_y     = None
baseline_calibrated   = False
nodding_detected      = False

# --- Audio setup ---
pygame.mixer.init()
alert_sound = pygame.mixer.Sound('sound//TrinhAiCham.wav')

def play_alert_sound():
    if not pygame.mixer.get_busy():
        alert_sound.play(loops=-1)

def stop_alert_sound():
    if pygame.mixer.get_busy():
        alert_sound.stop()

# --- Initialize models ---
yolo_model = YoloDrowsinessDetector('assets/best.pt', YOLO_CONF_THRESH)
yolo_face  = YOLO('C:/Users/ADMIN/Downloads/yolov8n-face.pt')

mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Indices for EAR/MAR
LEFT_EYE_IDXS  = [362,385,387,263,373,380]
RIGHT_EYE_IDXS = [33,160,158,133,153,144]
MOUTH_IDXS     = [61,291,13,14]

# EAR calibrator
calibrator = EARCalibrator(CALIBRATION_FRAMES, EAR_CALIBRATION_FACTOR)

# --- Start capture ---
print('[INFO] Starting webcam...')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- FaceMesh for EAR/MAR ---
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        pts = np.array([[int(p.x*w), int(p.y*h)] for p in lm], dtype=np.int32)

        # Compute EAR
        ear = (eye_aspect_ratio(pts[LEFT_EYE_IDXS]) + eye_aspect_ratio(pts[RIGHT_EYE_IDXS])) / 2.0
        draw_contour(frame, pts[LEFT_EYE_IDXS])
        draw_contour(frame, pts[RIGHT_EYE_IDXS])
        draw_text(frame, f'EAR: {ear:.2f}', (w-150, 30))

        # Compute MAR
        mar = mouth_aspect_ratio(pts[MOUTH_IDXS])
        draw_contour(frame, pts[MOUTH_IDXS])
        draw_text(frame, f'MAR: {mar:.2f}', (w-150, 60))

        # Calibrate EAR threshold
        if not is_calibrated:
            draw_text(frame, 'Calibrating EAR...', (10,30), (0,255,255))
            if calibrator.update(ear):
                EYE_AR_THRESH = calibrator.get_threshold()
                is_calibrated = True
                print(f'[INFO] EAR threshold: {EYE_AR_THRESH:.3f}')
        else:
            # Update counters
            EYE_COUNTER   = EYE_COUNTER + 1 if ear < EYE_AR_THRESH else 0
            MOUTH_COUNTER = MOUTH_COUNTER + 1 if mar > MOUTH_AR_THRESH else 0

    # --- YOLO face detection for nodding ---
    nodding_detected = False
    res_face = yolo_face.predict(frame, verbose=False)
    if res_face and res_face[0].boxes.xyxy:
        x1,y1,x2,y2 = res_face[0].boxes.xyxy[0].int().tolist()
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)

        # Calibrate baseline center Y
        if not baseline_calibrated:
            baseline_list.append(cy)
            if len(baseline_list) >= CALIBRATION_FRAMES_NOD:
                baseline_center_y = sum(baseline_list) / len(baseline_list)
                baseline_calibrated = True
                print(f'[INFO] Baseline center Y: {baseline_center_y:.1f}')
        else:
            # Detect nodding if Y deviates by threshold
            if abs(cy - baseline_center_y) > THRESHOLD_NOD_PX:
                nodding_detected = True

    # --- YOLO Drowsiness Detector ---
    if is_calibrated:
        detected, coords, conf = yolo_model.detect(frame)
        if detected:
            YOLO_COUNTER += 1
            x1,y1,x2,y2 = coords
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255), 2)
            draw_text(frame, f'YOLO: Drowsy ({conf:.2f})', (x1, y1-10), (255,0,255))
        else:
            YOLO_COUNTER = 0

    # --- Combined Alert Logic ---
    alert = (
        EYE_COUNTER   >= EYE_AR_CONSEC_FRAMES or
        MOUTH_COUNTER >= MOUTH_AR_CONSEC_FRAMES or
        YOLO_COUNTER  >= YOLO_CONSEC_FRAMES or
        nodding_detected
    )
    if alert:
        play_alert_sound()
        if nodding_detected:
            draw_text(frame, 'WARNING: Nodding detected', (20,140), (0,0,255), 1, 2)
    else:
        stop_alert_sound()

    cv2.imshow('Hybrid Drowsiness Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
