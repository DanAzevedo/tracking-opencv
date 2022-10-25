import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

parameters_lucas_kanade = dict(winSize=(15, 15),
                               maxLevel=4,
                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def select_point(event, x, y, flags, params):
    global point, selected_point, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        selected_point = True
        old_points = np.array([[x, y]], dtype=np.float32)
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', select_point)

selected_point = False
point = ()
old_points = np.array([[]])

mask = np.zeros_like(frame)

while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if selected_point is True:
        cv2.circle(frame, point, 5, (0, 0, 255), 2)

        new_points, status, errors = cv2.calcOpticalFlowPyrLK(frame_gray_init,
                                                              frame_gray,
                                                              old_points,
                                                              None,
                                                              **parameters_lucas_kanade)
        frame_gray_init = frame_gray.copy()
        old_points = new_points

        x, y = new_points.ravel()
        j, k = old_points.ravel()

        mask = cv2.line(mask, (x, y), (j, k), (0, 255, 255), 2)
        frame = cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    img = cv2.add(frame, mask)

    cv2.imshow("Frame", frame)
    cv2.imshow("Frame 2", mask)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()