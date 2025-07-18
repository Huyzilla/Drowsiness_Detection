import cv2
import mediapipe as mp
import numpy as np
import math
import time
import pygame
from my_utils.config_loader import load_config
from detectors.eye_detector import eye_aspect_ratio
from detectors.mouth_detector import mouth_aspect_ratio
from detectors.yolo_detector import YoloDrowsinessDetector
from detectors.yolox_onnx_detector import YoloDrowsinessONNXDetector
from calibration.calibrator import EARCalibrator
from my_utils.draw_utils import draw_text, draw_contour

config = load_config()

# MediaPipe
EYE_AR_CONSEC_FRAMES = config['mediapipe_settings']['eye_ar_consec_frames']
MOUTH_AR_THRESH = config['mediapipe_settings']['mouth_ar_thresh']
MOUTH_AR_CONSEC_FRAMES = config['mediapipe_settings']['mouth_ar_consec_frames']

# Head Pose 
HEAD_Y_RATIO_THRESH = config['head_pose_settings']['y_deviation_ratio_thresh']
HEAD_ANGLE_THRESH = config['head_pose_settings']['angle_deviation_thresh']
HEAD_SCORE_THRESH = config['head_pose_settings']['score_thresh']
HEAD_SCORE_INCREMENT = config['head_pose_settings']['score_increment']
HEAD_SCORE_DECREMENT = config['head_pose_settings']['score_decrement']

# YOLO
YOLO_CONF_THRESH = config['yolo_settings']['confidence_thresh']
YOLO_SCORE_THRESH = config['yolo_settings']['score_thresh']
YOLO_SCORE_INCREMENT = config['yolo_settings']['score_increment']
YOLO_SCORE_DECREMENT = config['yolo_settings']['score_decrement']

CALIBRATION_FRAMES = config['calibration_settings']['calibration_frames']
EAR_CALIBRATION_FACTOR = config['calibration_settings']['ear_calibration_factor']
SOUND_PATH = config['sound_settings']['sound_path']

EYE_COUNTER = 0; MOUTH_COUNTER = 0; yolo_score = 0
NOD_SCORE = 0; TILT_SCORE = 0;

calibration_started = False; is_calibrated = False; monitoring_started = False
EYE_AR_THRESH = 0
baseline_head_y = 0; baseline_head_angle = 0
calibration_y_list = []; calibration_angle_list = []
ear_calibrator = EARCalibrator(CALIBRATION_FRAMES, EAR_CALIBRATION_FACTOR)

pygame.mixer.init()
alert_sound = pygame.mixer.Sound(SOUND_PATH)
def play_alert_sound():
    if not pygame.mixer.get_busy(): alert_sound.play(-1)
def stop_alert_sound():
    alert_sound.stop()

yolo_model = YoloDrowsinessDetector('assets/best.pt', YOLO_CONF_THRESH) 
# yolo_model = YoloDrowsinessONNXDetector('best.pt', YOLO_CONF_THRESH) 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Các chỉ số điểm mốc
LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]; RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
MOUTH_IDXS = [61, 291, 13, 14]
TOP_HEAD_IDX = 10  # đỉnh trán
NOSE_TIP_IDX = 1   # chóp mũi

def calculate_head_angle(p1, p2):
    """Tính góc của đường thẳng p1-p2 so với phương thẳng đứng."""
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]
    angle_rad = math.atan2(delta_x, delta_y)
    return math.degrees(angle_rad)

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
        draw_text(frame, "DANG HIEU CHINH: NHIN THANG, KHONG NGHIENG DAU", (10, 30), color=(0, 255, 255))
        if mp_results.multi_face_landmarks:
            face_landmarks = mp_results.multi_face_landmarks[0]
            shape = np.array([[int(lm.x * w), int(lm.y * h)] for lm in face_landmarks.landmark], dtype=np.int32)
            
            ear = (eye_aspect_ratio(shape[LEFT_EYE_IDXS]) + eye_aspect_ratio(shape[RIGHT_EYE_IDXS])) / 2.0
            
            # Lấy tọa độ các điểm cần thiết cho tư thế đầu
            top_head_pt = shape[TOP_HEAD_IDX]
            nose_tip_pt = shape[NOSE_TIP_IDX]
            
            # Thu thập dữ liệu hiệu chỉnh
            calibration_y_list.append(top_head_pt[1])
            calibration_angle_list.append(calculate_head_angle(top_head_pt, nose_tip_pt))
            
            # Khi thu thập đủ frame
            if ear_calibrator.update(ear):
                EYE_AR_THRESH = ear_calibrator.get_threshold()
                baseline_head_y = np.mean(calibration_y_list)
                baseline_head_angle = np.mean(calibration_angle_list)
                is_calibrated = True
                print("Hieu chinh hoan tat.")
                print(f"--> Nguong EAR: {EYE_AR_THRESH:.3f} | Y ban dau: {baseline_head_y:.2f} | Goc ban dau: {baseline_head_angle:.2f}")
        else:
            draw_text(frame, "Khong tim thay khuon mat!", (10, 60), color=(0, 0, 255))
    
    # TRẠNG THÁI 3: CHỜ BẮT ĐẦU GIÁM SÁT
    elif not monitoring_started:
        draw_text(frame, "DA HIEU CHINH. NHAN 'r' DE BAT DAU GIAM SAT", (10, 30), color=(0, 255, 0))
        if key == ord('r'):
            monitoring_started = True

    # TRẠNG THÁI 4: ĐANG GIÁM SÁT
    else:
        if mp_results.multi_face_landmarks:
            face_landmarks = mp_results.multi_face_landmarks[0]
            shape = np.array([[int(lm.x * w), int(lm.y * h)] for lm in face_landmarks.landmark], dtype=np.int32)
            
            # Tính EAR, MAR
            leftEye = shape[LEFT_EYE_IDXS]; rightEye = shape[RIGHT_EYE_IDXS]; mouth = shape[MOUTH_IDXS]
            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
            mar = mouth_aspect_ratio(mouth)
            
            # Tính tư thế đầu
            top_head_pt = shape[TOP_HEAD_IDX]
            nose_tip_pt = shape[NOSE_TIP_IDX]
            chin_pt = shape[152] # cằm 

            face_height = np.linalg.norm(top_head_pt - chin_pt) # Chiều cao khuôn mặt
            
            current_head_y = top_head_pt[1]
            current_head_angle = calculate_head_angle(top_head_pt, nose_tip_pt)

            # Tính độ lệch
            y_deviation = current_head_y - baseline_head_y
            angle_deviation = current_head_angle - baseline_head_angle
            
            # Cập nhật các bộ đếm
            if ear < EYE_AR_THRESH: EYE_COUNTER += 1
            else: EYE_COUNTER = 0
            
            if mar > MOUTH_AR_THRESH: MOUTH_COUNTER += 1
            else: MOUTH_COUNTER = 0

            
            # Kiểm tra cúi/ngửa đầu
            if y_deviation > HEAD_Y_RATIO_THRESH * face_height:
                NOD_SCORE = min(HEAD_SCORE_THRESH + 5, NOD_SCORE + HEAD_SCORE_INCREMENT)
            else:
                NOD_SCORE = max(0, NOD_SCORE - HEAD_SCORE_DECREMENT)
            # Xét nghiêng đầu
            if abs(angle_deviation) > HEAD_ANGLE_THRESH:
                TILT_SCORE = min(HEAD_SCORE_THRESH + 5, TILT_SCORE + HEAD_SCORE_INCREMENT)
            else:
                TILT_SCORE = max(0, TILT_SCORE - HEAD_SCORE_DECREMENT)

            # Vẽ thông tin
            draw_contour(frame, leftEye); draw_contour(frame, rightEye); draw_contour(frame, mouth)
            cv2.circle(frame, tuple(top_head_pt), 3, (0, 255, 0), -1)
            cv2.circle(frame, tuple(nose_tip_pt), 3, (0, 255, 0), -1)
            cv2.line(frame, tuple(top_head_pt), tuple(nose_tip_pt), (0, 255, 0), 1)

            draw_text(frame, f"EAR: {ear:.2f} (T: {EYE_AR_THRESH:.2f})", (w - 250, 30))
            draw_text(frame, f"MAR: {mar:.2f}", (w - 250, 60))
            draw_text(frame, f"Y_Dev_Ratio: {y_deviation/face_height:.2f} (T: {HEAD_Y_RATIO_THRESH:.2f})", (w - 250, 90))
            draw_text(frame, f"Angle_Dev: {angle_deviation:.2f} (T: {HEAD_ANGLE_THRESH:.2f})", (w - 250, 120))
            draw_text(frame, f"NOD_SCORE: {NOD_SCORE}", (w - 280, 150))
            draw_text(frame, f"TILT_SCORE: {TILT_SCORE}", (w - 280, 180))

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
        draw_text(frame, f"YOLO_SCORE: {yolo_score}", (w - 280, 210))

        # Tổng hợp cảnh báo
        if EYE_COUNTER >= EYE_AR_CONSEC_FRAMES or NOD_SCORE >= HEAD_SCORE_THRESH or TILT_SCORE >= HEAD_SCORE_THRESH or yolo_score >= YOLO_SCORE_THRESH:
            alert_triggered = True
        
        if alert_triggered:
            draw_text(frame, "!!! CANH BAO BUON NGU !!!", (10, 50), (0, 0, 255), 1, 3)
            play_alert_sound()
        elif MOUTH_COUNTER >= MOUTH_AR_CONSEC_FRAMES:
            draw_text(frame, "Phat hien ngap", (10, 50), (0, 255, 255), 1, 2)
            stop_alert_sound() 
        else:
            stop_alert_sound()
    
    cv2.imshow("Drowsiness Detection", frame)
    if key == ord('q'):
        break

stop_alert_sound()
cap.release()
cv2.destroyAllWindows()