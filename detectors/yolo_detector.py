'''
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
'''


import torch
import cv2
import platform
import pathlib

class YoloDrowsinessDetector:
    def __init__(self, model_path, confidence_thresh=0.8, target_class_name='drowsy'):
        self.confidence_thresh = confidence_thresh
        self.target_class_name = target_class_name 

        if platform.system() == 'Windows':
            temp_posix_path = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
        
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
            print("[INFO] Tải mô hình YOLOv5 thành công.")
            print("[INFO] Danh sách nhãn của model:", self.model.names)
            print(f"[INFO] Lớp mục tiêu được thiết lập là: '{self.target_class_name}'")

        except Exception as e:
            print(f"[ERROR] Không thể tải mô hình YOLOv5. Lỗi: {e}")
            print("[INFO] Đang thử tải lại với force_reload=True...")
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
            print("[INFO] Tải lại mô hình YOLOv5 thành công.")
            print("[INFO] Danh sách nhãn của model:", self.model.names)
            print(f"[INFO] Lớp mục tiêu được thiết lập là: '{self.target_class_name}'")
        finally:
            if platform.system() == 'Windows':
                pathlib.PosixPath = temp_posix_path

    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb)
        df = results.pandas().xyxy[0]
        
        detections = df[df['confidence'] >= self.confidence_thresh]

        target_detections = detections[detections['name'] == self.target_class_name]
        
        if not target_detections.empty:
            best_detection = target_detections.loc[target_detections['confidence'].idxmax()]
            
            x1 = int(best_detection['xmin'])
            y1 = int(best_detection['ymin'])
            x2 = int(best_detection['xmax'])
            y2 = int(best_detection['ymax'])
            
            coords = [x1, y1, x2, y2]
            conf = best_detection['confidence']
            
            return True, coords, conf
            
        return False, [], 0.0