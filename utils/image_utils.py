import cv2
import numpy as np

def warp_perspective(plate_crop):
    h, w = plate_crop.shape[:2]
    src_pts = np.float32([[0, 30], [w - 1, -30], [w - 1, h - 1 - 30], [0, h - 1 + 30]])
    dst_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(plate_crop, M, (w, h))
