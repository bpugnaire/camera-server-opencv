import cv2
import numpy as np

def tracking(frame):
    tracker = cv2.TrackerKCF.create()
    success, box = tracker.update(frame)
    if success:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)
    return frame