import cv2
from ultralytics import YOLO
import pygame

YOLO_CONF_THRESH = 0.7
YOLO_CONSEC_FRAMES = 10
SOUND_PATH = 'sound/TrinhAiCham.wav' 

YOLO_COUNTER = 0
pygame.mixer.init()
alert_sound = pygame.mixer.Sound(SOUND_PATH)

def play_alert_sound():
    if not pygame.mixer.get_busy():
        alert_sound.play(-1)

def stop_alert_sound():
    alert_sound.stop()

yolo_model = YOLO('assets/best.pt')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    alert_triggered = False
    yolo_drowsy_detected_this_frame = False

    yolo_results = yolo_model(frame, verbose=False)
    
    for r in yolo_results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]
            confidence = box.conf[0].item()
            
            if class_name == 'drowsy' and confidence > YOLO_CONF_THRESH:
                yolo_drowsy_detected_this_frame = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, f"YOLO: Drowsy ({confidence:.2f})", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                break
        if yolo_drowsy_detected_this_frame: break

    if yolo_drowsy_detected_this_frame:
        YOLO_COUNTER += 1
    else:
        YOLO_COUNTER = 0

    cv2.putText(frame, f"YOLO_CTR: {YOLO_COUNTER}", (w - 180, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    if YOLO_COUNTER >= YOLO_CONSEC_FRAMES:
        alert_triggered = True
        cv2.putText(frame, "!!! CANH BAO GAT GU !!!", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
    if alert_triggered:
        play_alert_sound()
    else:
        stop_alert_sound()

    cv2.imshow("Debug YOLO Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stop_alert_sound()
cap.release()
cv2.destroyAllWindows()