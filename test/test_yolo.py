import cv2
import numpy as np
import onnxruntime as ort

class YoloDrowsinessONNXDetector:
    def __init__(self, model_path='assets/yolox_nano.onnx', conf_thresh=0.6):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.conf_thresh = conf_thresh
        self.input_size = 416

    def _preprocess(self, frame):
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))[None, ...]
        return img

    def detect(self, frame):
        inp = self._preprocess(frame)
        outputs = self.session.run(None, {self.input_name: inp})
        preds = outputs[0].squeeze(0)

        h, w = frame.shape[:2]
        for det in preds:
            det = det.tolist()
            x1, y1, x2, y2 = det[0:4]
            score = det[4]
            cls_id = int(det[-1])

            if score < self.conf_thresh:
                continue

            # Cast về ảnh gốc (nếu cần)
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            return True, (x1, y1, x2, y2), float(score)

        return False, (0, 0, 0, 0), 0.0


# ----------------------------
# Phần chạy thử webcam
# ----------------------------
detector = YoloDrowsinessONNXDetector(model_path='assets/yolox_nano.onnx')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không mở được webcam!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    found, bbox, score = detector.detect(frame)

    if found:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Drowsiness Detection - ONNX", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
