import cv2
import mediapipe as mp
import numpy as np
from utils.config_loader import load_config
from detectors.eye_detector import eye_aspect_ratio
from detectors.mouth_detector import mouth_aspect_ratio
from calibration.calibrator import EARCalibrator
from utils.draw_utils import draw_text, draw_contour
import time
import pygame

# --- Tải cấu hình ---
config = load_config('test_config.yaml')

EYE_AR_CONSEC_FRAMES = config['mediapipe_settings']['eye_ar_consec_frames']
MOUTH_AR_THRESH = config['mediapipe_settings']['mouth_ar_thresh']
MOUTH_AR_CONSEC_FRAMES = config['mediapipe_settings']['mouth_ar_consec_frames']
HEAD_NOD_PIXEL_THRESH = config['head_pose_settings']['nod_pixel_thresh']
HEAD_CONSEC_FRAMES = config['head_pose_settings']['consec_frames']
CALIBRATION_FRAMES = config['calibration_settings']['calibration_frames']
EAR_CALIBRATION_FACTOR = config['calibration_settings']['ear_calibration_factor']
SOUND_PATH = config['sound_settings']['sound_path']

# --- KHỞI TẠO ---
EYE_COUNTER = 0
MOUTH_COUNTER = 0
HEAD_NOD_COUNTER = 0 
calibration = EARCalibrator(CALIBRATION_FRAMES, EAR_CALIBRATION_FACTOR)
EYE_AR_THRESH = 0
initial_head_y = 0 

# THÊM CÁC BIẾN TRẠNG THÁI
calibration_started = False
is_calibrated = False

# Khởi tạo pygame
pygame.mixer.init()
alert_sound = pygame.mixer.Sound(SOUND_PATH)

def play_alert_sound():
    if not pygame.mixer.get_busy():
        alert_sound.play(-1)

def stop_alert_sound():
    alert_sound.stop()

# Tải mô hình MediaPipe
print("[INFO] Tải mô hình MediaPipe...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Các chỉ số điểm mốc
LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
MOUTH_IDXS = [61, 291, 13, 14]
TOP_OF_HEAD_IDX = 10

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

    # Lấy key press
    key = cv2.waitKey(1) & 0xFF

    # === LOGIC TRẠNG THÁI MỚI ===

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
            leftEye = shape[LEFT_EYE_IDXS]; rightEye = shape[RIGHT_EYE_IDXS]
            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
            
            if calibration.update(ear):
                EYE_AR_THRESH = calibration.get_threshold()
                initial_head_y = shape[TOP_OF_HEAD_IDX][1]
                is_calibrated = True
                print(f"[INFO] Hieu chinh hoan tat. Nguong EAR: {EYE_AR_THRESH:.3f}, Vi tri dau: {initial_head_y}")
        else:
             draw_text(frame, "Khong tim thay khuon mat!", (10, 60), color=(0, 0, 255))

    # TRẠNG THÁI 3: ĐÃ HIỆU CHỈNH, BẮT ĐẦU GIÁM SÁT
    else:
        if mp_results.multi_face_landmarks:
            face_landmarks = mp_results.multi_face_landmarks[0]
            shape = np.array([[int(lm.x * w), int(lm.y * h)] for lm in face_landmarks.landmark], dtype=np.int32)
            
            leftEye = shape[LEFT_EYE_IDXS]; rightEye = shape[RIGHT_EYE_IDXS]; mouth = shape[MOUTH_IDXS]
            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
            mar = mouth_aspect_ratio(mouth)
            
            # Kiểm tra và cập nhật các bộ đếm
            if ear < EYE_AR_THRESH: EYE_COUNTER += 1
            else: EYE_COUNTER = 0
            
            if mar > MOUTH_AR_THRESH: MOUTH_COUNTER += 1
            else: MOUTH_COUNTER = 0

            current_head_y = shape[TOP_OF_HEAD_IDX][1]
            y_difference = current_head_y - initial_head_y
            if y_difference > HEAD_NOD_PIXEL_THRESH: HEAD_NOD_COUNTER += 1
            else: HEAD_NOD_COUNTER = 0

            # Vẽ và hiển thị
            draw_contour(frame, leftEye); draw_contour(frame, rightEye); draw_contour(frame, mouth)
            draw_text(frame, f"EAR: {ear:.2f} (T: {EYE_AR_THRESH:.2f})", (w - 220, 30))
            draw_text(frame, f"MAR: {mar:.2f}", (w - 220, 60))
            draw_text(frame, f"HEAD_Y_DIFF: {y_difference}", (w - 220, 90))

            # Tổng hợp cảnh báo
            if HEAD_NOD_COUNTER >= HEAD_CONSEC_FRAMES or EYE_COUNTER >= EYE_AR_CONSEC_FRAMES:
                alert_triggered = True
            
            if alert_triggered:
                draw_text(frame, "!!! CANH BAO BUON NGU !!!", (10, 50), (0, 0, 255), 1, 3)
                play_alert_sound()
            elif MOUTH_COUNTER >= MOUTH_AR_CONSEC_FRAMES:
                draw_text(frame, "Phat hien ngap", (10, 50), (0, 255, 255), 1, 2)
            else: # Dừng âm thanh nếu không có cảnh báo nào
                stop_alert_sound()
        else:
            stop_alert_sound()

    cv2.imshow("Advanced Drowsiness Detection", frame)
    
    # Thoát bằng phím 'q'
    if key == ord('q'):
        break

stop_alert_sound()
cap.release()
cv2.destroyAllWindows()