import numpy as np

class EARCalibrator:
    def __init__(self, calibration_frames, factor):
        self.calibration_frames = calibration_frames
        self.factor = factor
        self.ear_values = []

    def update(self, ear):
        self.ear_values.append(ear)
        return len(self.ear_values) >= self.calibration_frames

    def get_threshold(self):
        return np.mean(self.ear_values) * self.factor