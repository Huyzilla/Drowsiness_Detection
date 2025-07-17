import onnxruntime as ort
import numpy as np

class YoloDrowsinessONNXDetector:
    def __init__(self, model_path='best.onnx', conf_thresh=0.92):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.conf_thresh = conf_thresh
        self.input_size = 640

    def _preprocess(self, frame):
        import cv2
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))[None, ...]
        return img
    def detect(self, frame):
        import cv2
        inp = self._preprocess(frame)
        outputs = self.session.run(None, {self.input_name: inp})
        preds = outputs[0].squeeze(0)  # shape [N, ?]

        h, w = frame.shape[:2]
        for det in preds:
            det = det.tolist()
            # 4 giá trị đầu là x1,y1,x2,y2 (pixel)
            x1, y1, x2, y2 = det[0:4]
            score = det[4]             # phần tử thứ 5 luôn là confidence
            cls_id = int(det[-1])      # phần tử cuối cùng là class_id
            
            if score < self.conf_thresh:
                continue

            # cast về int và trả kết quả
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            return True, (x1, y1, x2, y2), float(score)

        return False, (0, 0, 0, 0), 0.0


    

