import cv2
import numpy as np

def draw_text(frame, text, position, color=(255,0,0), scale=0.7, thickness=2):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def draw_contour(frame, points, color=(0,255,0)):
    import numpy as np
    contour = cv2.convexHull(np.array(points))
    cv2.drawContours(frame, [contour], -1, color, 1)