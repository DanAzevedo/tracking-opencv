import cv2
import sys
from random import randint

tracker = cv2.TrackerCSRT_create()

video = cv2.VideoCapture("videos/walking.avi")

if not video.isOpened():
    print("Não foi possível abrir o vídeo")
    sys.exit()

ok, frame = video.read()
if not ok:
    print('Não é possível ler o arquivo de vídeo')
    sys.exit()

cascade = cv2.CascadeClassifier('cascade/fullbody.xml')

def detectar():
    while True:
        ok, frame = video.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detection = cascade.detectMultiScale(frame_gray)
        for (x, y, l, a) in detection:
            cv2.rectangle(frame, (x, y), (x + l, y + a), (0,0,255), 2)
            cv2.imshow("Deteção", frame)

            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            if x > 0:
                print('Detecção efetuada pelo haarcascade')
                return x, y, l, a

bbox = detectar()
#print(bbox)

ok = tracker.init(frame, bbox)
colors = (randint(0, 255), randint(0, 255), randint(0, 255))

while True:
    ok, frame = video.read()
    if not ok:
        break

    ok, bbox = tracker.update(frame)
    if ok:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors, 2, 1)
    else:
        print('Falha no rastreamento. Será executado o detector haarcascade')
        bbox = detectar()
        tracker = cv2.TrackerMOSSE_create()
        tracker.init(frame, bbox)

    cv2.imshow("Tracking", frame)
    k = cv2.waitKey(1) & 0XFF
    if k == 27:
        break

















