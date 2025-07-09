# CODE HOÀN CHỈNH - TÍCH HỢP HIỆU CHỈNH TỰ ĐỘNG

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from ultralytics import YOLO
import yaml
import time

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5]); B = dist.euclidean(eye[2], eye[4]); C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    horizontal_dist = dist.euclidean(mouth[0], mouth[1]); vertical_dist = dist.euclidean(mouth[2], mouth[3])
    return vertical_dist / horizontal_dist

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

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
calibration_ear_values = []
is_calibrated = False
EYE_AR_THRESH = 0 

yolo_model = YOLO('best.pt')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Các chỉ số điểm mốc
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
        
        leftEye = shape[LEFT_EYE_IDXS]; rightEye = shape[RIGHT_EYE_IDXS]
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

        if not is_calibrated:
            cv2.putText(frame, "HIEN CHINH: NHIN THANG & MO MAT BINH THUONG", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            calibration_ear_values.append(ear)

            if len(calibration_ear_values) >= CALIBRATION_FRAMES:
                avg_ear = np.mean(calibration_ear_values)
                EYE_AR_THRESH = avg_ear * EAR_CALIBRATION_FACTOR
                is_calibrated = True
                print(f"[INFO] Hieu chinh hoan tat. Nguong EAR ca nhan: {EYE_AR_THRESH:.3f}")
        
        else:
            mouth = shape[MOUTH_IDXS]
            mar = mouth_aspect_ratio(mouth)
            cv2.drawContours(frame, [cv2.convexHull(shape[MOUTH_IDXS])], -1, (0, 255, 0), 1)
            cv2.putText(frame, f"MAR: {mar:.2f}", (w - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            if ear < EYE_AR_THRESH: EYE_COUNTER += 1
            else: EYE_COUNTER = 0

            if mar > MOUTH_AR_THRESH: MOUTH_COUNTER += 1
            else: MOUTH_COUNTER = 0
        
        cv2.putText(frame, f"EAR: {ear:.2f}", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    if is_calibrated:
        yolo_results = yolo_model(frame, verbose=False)
        yolo_drowsy_detected_this_frame = False
        
        for r in yolo_results:
            for box in r.boxes:
                class_id = int(box.cls[0]); class_name = yolo_model.names[class_id]; confidence = box.conf[0].item()
                if class_name == 'drowsy' and confidence > YOLO_CONF_THRESH:
                    yolo_drowsy_detected_this_frame = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(frame, f"YOLO: Drowsy ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    break
            if yolo_drowsy_detected_this_frame: break

        if yolo_drowsy_detected_this_frame: YOLO_COUNTER += 1
        else: YOLO_COUNTER = 0

        if YOLO_COUNTER >= YOLO_CONSEC_FRAMES or EYE_COUNTER >= EYE_AR_CONSEC_FRAMES or MOUTH_COUNTER >= MOUTH_AR_CONSEC_FRAMES:
            cv2.putText(frame, "!!! CANH BAO BUON NGU !!!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Hybrid Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()