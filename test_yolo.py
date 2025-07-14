from ultralytics import YOLO
import cv2

# Load mô hình YOLOv8 đã huấn luyện
model = YOLO("assets/best.pt")  # hoặc path tới best.pt

# Mở webcam (0 là camera mặc định, nếu không chạy được thì thử 1, 2,...)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Chạy dự đoán trên frame hiện tại
    results = model.predict(source=frame, show=False, conf=0.5, verbose=False)

    # Vẽ khung kết quả lên frame
    annotated_frame = results[0].plot()

    # Hiển thị frame
    cv2.imshow("Drowsiness Detection - Real-Time", annotated_frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
