import numpy as np
import cv2
from imutils.video import VideoStream
import time

cap = VideoStream(src=0).start()
time.sleep(1.0)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
bbox = cv2.selectROI(frame, False)
x, y, w, h = bbox
track_window = (x, y, w, h)
roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        ret, track_window = cv2.CamShift(dst, (x, y, w, h), term_crit)

        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame, [pts], True, 255, 2)

        cv2.imshow('Camshift Rastreado', img2)

        if cv2.waitKey(1) == 13:
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()