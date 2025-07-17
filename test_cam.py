import cv2
from detectors.yolo_detector import YoloDrowsinessDetector

model_path = 'best.pt'  
confidence_threshold = 0.8
yolo = YoloDrowsinessDetector(model_path, confidence_threshold)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    detected, bbox, score = yolo.detect(frame)
    if detected:
        x1, y1, x2, y2 = bbox
        # cv2.rectangle(image, start_point, end_point, color, thickness: Do day)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"Drowsy ({score:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #else:
        # cv2.putText(image, text, org: tọa độ của góc dưới bên trái của chuỗi văn bản, font, fontScale, 
        # color[, thickness[, lineType[, bottomLeftOrigin]]])   
        # cv2.putText(frame, "Awake", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

