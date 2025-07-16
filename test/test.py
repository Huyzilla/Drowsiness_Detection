import cv2
import mediapipe as mp
import numpy as np
from my_utils.config_loader import load_config
from detectors.eye_detector import eye_aspect_ratio
from detectors.mouth_detector import mouth_aspect_ratio
from detectors.yolo_detector import YoloDrowsinessDetector
from calibration.calibrator import EARCalibrator
from my_utils.draw_utils import draw_text, draw_contour
import time
import pygame
import math

# --- Tải cấu hình ---
config = load_config('config.yaml')

# Cấu hình cho MediaPipe
EYE_AR_CONSEC_FRAMES = config['mediapipe_settings']['eye_ar_consec_frames']
MOUTH_AR_THRESH = config['mediapipe_settings']['mouth_ar_thresh']
MOUTH_AR_CONSEC_FRAMES = config['mediapipe_settings']['mouth_ar_consec_frames']

# Cấu hình cho Head Pose
HEAD_PITCH_DEVIATION_THRESH = config['head_pose_settings']['pitch_deviation_thresh']
HEAD_ROLL_DEVIATION_THRESH = config['head_pose_settings']['roll_deviation_thresh']
HEAD_CONSEC_FRAMES = config['head_pose_settings']['consec_frames']

# Cấu hình cho YOLO
YOLO_CONF_THRESH = config['yolo_settings']['confidence_thresh']
YOLO_SCORE_THRESH = config['yolo_settings']['score_thresh']
YOLO_SCORE_INCREMENT = config['yolo_settings']['score_increment']
YOLO_SCORE_DECREMENT = config['yolo_settings']['score_decrement']

# Cấu hình cho hiệu chỉnh
CALIBRATION_FRAMES = config['calibration_settings']['calibration_frames']
EAR_CALIBRATION_FACTOR = config['calibration_settings']['ear_calibration_factor']
SOUND_PATH = config['sound_settings']['sound_path']

# --- KHỞI TẠO ---
EYE_COUNTER = 0; MOUTH_COUNTER = 0; HEAD_POSE_COUNTER = 0; yolo_score = 0
calibration_started = False; is_calibrated = False; monitoring_started = False
EYE_AR_THRESH = 0; baseline_pitch = 0; baseline_roll = 0
calibration_pitch_list = []; calibration_roll_list = []
ear_calibrator = EARCalibrator(CALIBRATION_FRAMES, EAR_CALIBRATION_FACTOR)

# Khởi tạo pygame
pygame.mixer.init()
alert_sound = pygame.mixer.Sound(SOUND_PATH)
def play_alert_sound():
    if not pygame.mixer.get_busy(): alert_sound.play(-1)
def stop_alert_sound():
    alert_sound.stop()

# Tải các mô hình
print("[INFO] Tải các mô hình...")
yolo_model = YoloDrowsinessDetector('best.pt', YOLO_CONF_THRESH) 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Các chỉ số điểm mốc
LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]; RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
MOUTH_IDXS = [61, 291, 13, 14]
HEAD_POSE_IDXS = [33, 263, 1, 61, 291, 199] 

# --- VÒNG LẶP CHÍNH ---
print("[INFO] Mở webcam...")
cap = cv2.VideoCapture(0)
time.sleep(1.0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    alert_triggered = False

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_results = face_mesh.process(frame_rgb)
    
    key = cv2.waitKey(1) & 0xFF

    # TRẠNG THÁI 1: CHỜ BẮT ĐẦU HIỆU CHỈNH
    if not calibration_started:
        draw_text(frame, "NHAN PHIM 'c' DE BAT DAU HIEU CHINH", (10, 30), color=(0, 255, 255))
        if key == ord('c'):
            calibration_started = True
    
    # TRẠNG THÁI 2: ĐANG HIỆU CHỈNH
    elif not is_calibrated:
        draw_text(frame, "DANG HIEU CHINH: NHIN THANG, MO MAT BINH THUONG", (10, 30), color=(0, 255, 255))
        if mp_results.multi_face_landmarks:
            face_landmarks = mp_results.multi_face_landmarks[0]
            shape = np.array([[int(lm.x * w), int(lm.y * h)] for lm in face_landmarks.landmark], dtype=np.int32)
            ear = (eye_aspect_ratio(shape[LEFT_EYE_IDXS]) + eye_aspect_ratio(shape[RIGHT_EYE_IDXS])) / 2.0
            
            # Tính và thu thập các góc đầu
            image_points = np.array([shape[i] for i in HEAD_POSE_IDXS], dtype="double")
            model_points = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0), (0.0, 150.0, 0.0)])
            camera_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype="double")
            (success, rotation_vector, _) = cv2.solvePnP(model_points, image_points, camera_matrix, np.zeros((4,1)))
            rmat, _ = cv2.Rodrigues(rotation_vector); angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            pitch, yaw, roll = angles[0], angles[1], angles[2]

            calibration_pitch_list.append(pitch)
            calibration_roll_list.append(roll)
            
            # Khi thu thập đủ frame
            if ear_calibrator.update(ear):
                EYE_AR_THRESH = ear_calibrator.get_threshold()
                baseline_pitch = np.mean(calibration_pitch_list)
                baseline_roll = np.mean(calibration_roll_list)
                is_calibrated = True
                print("Hieu chinh hoan tat.")
                print(f"--> Nguong EAR: {EYE_AR_THRESH:.3f} | Pitch ban dau: {baseline_pitch:.2f} | Roll ban dau: {baseline_roll:.2f}")
        else:
             draw_text(frame, "Khong tim thay khuon mat!", (10, 60), color=(0, 0, 255))
    
    # TRẠNG THÁI 3: CHỜ BẮT ĐẦU GIÁM SÁT
    elif not monitoring_started:
        draw_text(frame, "DA HIEU CHINH. NHAN 'r' DE BAT DAU GIAM SAT", (10, 30), color=(0, 255, 0))
        if key == ord('r'):
            monitoring_started = True

    # TRẠNG THÁI 4: ĐANG GIÁM SÁT
    else:
        # Xử lý MediaPipe
        if mp_results.multi_face_landmarks:
            face_landmarks = mp_results.multi_face_landmarks[0]
            shape = np.array([[int(lm.x * w), int(lm.y * h)] for lm in face_landmarks.landmark], dtype=np.int32)
            leftEye = shape[LEFT_EYE_IDXS]; rightEye = shape[RIGHT_EYE_IDXS]; mouth = shape[MOUTH_IDXS]
            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
            mar = mouth_aspect_ratio(mouth)
            image_points = np.array([shape[i] for i in HEAD_POSE_IDXS], dtype="double")
            model_points = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0), (0.0, 150.0, 0.0)])
            camera_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype="double")
            (_, rotation_vector, _) = cv2.solvePnP(model_points, image_points, camera_matrix, np.zeros((4,1)))
            rmat, _ = cv2.Rodrigues(rotation_vector); angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            pitch, yaw, roll = angles[0], angles[1], angles[2]
            
            if ear < EYE_AR_THRESH: EYE_COUNTER += 1
            else: EYE_COUNTER = 0
            if mar > MOUTH_AR_THRESH: MOUTH_COUNTER += 1
            else: MOUTH_COUNTER = 0

            pitch_deviation = pitch - baseline_pitch
            roll_deviation = roll - baseline_roll
            if pitch_deviation > HEAD_PITCH_DEVIATION_THRESH or abs(roll_deviation) > HEAD_ROLL_DEVIATION_THRESH:
                HEAD_POSE_COUNTER += 1
            else: HEAD_POSE_COUNTER = 0
            
            draw_contour(frame, leftEye); draw_contour(frame, rightEye); draw_contour(frame, mouth)
            draw_text(frame, f"EAR: {ear:.2f} (T: {EYE_AR_THRESH:.2f})", (w - 220, 30)); draw_text(frame, f"MAR: {mar:.2f}", (w - 220, 60))
            draw_text(frame, f"PITCH_D: {pitch_deviation:.2f}", (w - 220, 90)); draw_text(frame, f"ROLL_D: {roll_deviation:.2f}", (w - 220, 120))

        # Xử lý YOLO
        detected, coords, conf = yolo_model.detect(frame)
        if detected:
            yolo_score += YOLO_SCORE_INCREMENT
            x1, y1, x2, y2 = coords
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            draw_text(frame, f"YOLO: Drowsy ({conf:.2f})", (x1, y1 - 10), (255, 0, 255))
        else:
            yolo_score = max(0, yolo_score - YOLO_SCORE_DECREMENT)
        yolo_score = min(yolo_score, YOLO_SCORE_THRESH + 5)
        draw_text(frame, f"YOLO_SCORE: {yolo_score}", (w - 220, 150))

        # Tổng hợp cảnh báo
        if yolo_score >= YOLO_SCORE_THRESH or EYE_COUNTER >= EYE_AR_CONSEC_FRAMES or HEAD_POSE_COUNTER >= HEAD_CONSEC_FRAMES:
            alert_triggered = True
        
        if alert_triggered:
            draw_text(frame, "!!! CANH BAO BUON NGU !!!", (10, 50), (0, 0, 255), 1, 3)
            play_alert_sound()
        elif MOUTH_COUNTER >= MOUTH_AR_CONSEC_FRAMES:
            draw_text(frame, "Phat hien ngap", (10, 50), (0, 255, 255), 1, 2)
        else:
            stop_alert_sound()
    
    cv2.imshow("Ultimate Drowsiness Detection", frame)
    if key == ord('q'):
        break

stop_alert_sound()
cap.release()
cv2.destroyAllWindows()