import cv2
import mediapipe as mp
import numpy as np
import math
import time
import pygame
from my_utils.config_loader import load_config
from detectors.eye_detector import eye_aspect_ratio
from detectors.mouth_detector import mouth_aspect_ratio
from detectors.yolox_onnx_detector import YoloDrowsinessONNXDetector
from calibration.calibrator import EARCalibrator
from my_utils.draw_utils import draw_text, draw_contour

config = {
    'mediapipe_settings': {'eye_ar_consec_frames': 20, 'mouth_ar_thresh': 0.6, 'mouth_ar_consec_frames': 20},
    'head_pose_settings': {
        'y_deviation_ratio_thresh': 0.18, # Tỷ lệ lệch y so với chiều cao khung hình
        'angle_deviation_thresh': 20, 
        'score_thresh': 45,   # Ngưỡng điểm cho tư thế đầu 
        'score_increment': 1,
        'score_decrement': 1,
        'consec_frames': 20 
    },
    'yolo_settings': {'confidence_thresh': 0.9, 'score_thresh': 45, 'score_increment': 2, 'score_decrement': 1},
    'calibration_settings': {'calibration_frames': 30, 'ear_calibration_factor': 0.85},
    'sound_settings': {'sound_path': 'sound/TrinhAiCham.wav'}
}

# MediaPipe
EYE_AR_CONSEC_FRAMES = config['mediapipe_settings']['eye_ar_consec_frames']
MOUTH_AR_THRESH = config['mediapipe_settings']['mouth_ar_thresh']
MOUTH_AR_CONSEC_FRAMES = config['mediapipe_settings']['mouth_ar_consec_frames']

# Head Pose 
HEAD_Y_RATIO_THRESH = config['head_pose_settings']['y_deviation_ratio_thresh']
HEAD_ANGLE_THRESH = config['head_pose_settings']['angle_deviation_thresh']
HEAD_SCORE_THRESH = config['head_pose_settings']['score_thresh']
HEAD_SCORE_INCREMENT = config['head_pose_settings']['score_increment']
HEAD_SCORE_DECREMENT = config['head_pose_settings']['score_decrement']

# YOLO
YOLO_CONF_THRESH = config['yolo_settings']['confidence_thresh']
YOLO_SCORE_THRESH = config['yolo_settings']['score_thresh']
YOLO_SCORE_INCREMENT = config['yolo_settings']['score_increment']
YOLO_SCORE_DECREMENT = config['yolo_settings']['score_decrement']

CALIBRATION_FRAMES = config['calibration_settings']['calibration_frames']
EAR_CALIBRATION_FACTOR = config['calibration_settings']['ear_calibration_factor']
SOUND_PATH = config['sound_settings']['sound_path']

EYE_COUNTER = 0; MOUTH_COUNTER = 0; yolo_score = 0
NOD_SCORE = 0; TILT_SCORE = 0;

