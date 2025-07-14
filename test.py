# CODE CUỐI CÙNG - SỬ DỤNG HỆ THỐNG TÍCH ĐIỂM ĐỂ KHỬ NHIỄU

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from ultralytics import YOLO
import pygame

# --- CÁC HÀM TÍNH TOÁN (Không đổi) ---
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5]); B = dist.euclidean(eye[2], eye[4]); C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# --- CÁC THAM SỐ VÀ NGƯỠNG ---
# Ngưỡng về độ mở
EYE_AR_THRESH = 0.22
MOUTH_AR_THRESH = 0.5
YOLO_CONF_THRESH = 0.7

# *** THAY ĐỔI TỪ BỘ ĐẾM SANG HỆ THỐNG ĐIỂM ***
# Điểm số cần đạt để kích hoạt cảnh báo
# Bạn có thể tinh chỉnh các con số này
DROWSINESS_SCORE_THRESH = 25  # Ví dụ: cần 25 điểm để cảnh báo

# --- KHỞI TẠO CÁC BIẾN ĐIỂM ---
# Bắt đầu với 0 điểm
drowsiness_score = 0

# --- KHỞI TẠO ÂM THANH VÀ MODEL (Không đổi) ---
pygame.mixer.init()
alert_sound = pygame.mixer.Sound('sound/TrinhAiCham.wav') # Sửa lại tên file âm thanh của bạn

def play_alert_sound():
    if not pygame.mixer.get_busy():
        alert_sound.play(-1)

def stop_alert_sound():
    alert_sound.stop()

print("[INFO] Tải model YOLO...")
yolo_model = YOLO('assets/best.pt')

# --- VÒNG LẶP XỬ LÝ ---
print("[INFO] Mở webcam...")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # --- XỬ LÝ YOLO ---
    yolo_results = yolo_model(frame, verbose=False)
    yolo_drowsy_detected_this_frame = False
    
    for r in yolo_results:
        for box in r.boxes:
            class_id = int(box.cls[0]); class_name = yolo_model.names[class_id]; confidence = box.conf[0].item()
            if class_name == 'drowsy' and confidence > YOLO_CONF_THRESH:
                yolo_drowsy_detected_this_frame = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, f"YOLO: Drowsy ({confidence:.2f})", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                break
        if yolo_drowsy_detected_this_frame: break

    # --- LOGIC TÍCH ĐIỂM MỚI ---
    if yolo_drowsy_detected_this_frame:
        # Nếu phát hiện, cộng 2 điểm
        drowsiness_score += 2
    else:
        # Nếu không phát hiện, trừ 1 điểm (nhưng không xuống dưới 0)
        drowsiness_score = max(0, drowsiness_score - 1)
        
    # Giới hạn điểm số tối đa để không tăng vô hạn
    drowsiness_score = min(drowsiness_score, DROWSINESS_SCORE_THRESH + 5)


    # --- HIỂN THỊ ĐIỂM SỐ VÀ CẢNH BÁO ---
    # Hiển thị điểm số hiện tại
    cv2.putText(frame, f"SCORE: {drowsiness_score}", (w - 180, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Cảnh báo nếu điểm số vượt ngưỡng
    if drowsiness_score >= DROWSINESS_SCORE_THRESH:
        cv2.putText(frame, "!!! CANH BAO BUON NGU !!!", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        play_alert_sound()
    else:
        stop_alert_sound()

    cv2.imshow("Drowsiness Detection - Score System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stop_alert_sound()
cap.release()
cv2.destroyAllWindows()