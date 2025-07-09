from ultralytics import YOLO

class YoloDrowsinessDetector:
    def __init__(self, model_path, conf_thresh):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = box.conf[0].item()
                if self.model.names[cls] == "drowsy" and conf > self.conf_thresh:
                    coords = list(map(int, box.xyxy[0]))
                    return True, coords, conf
        return False, None, None