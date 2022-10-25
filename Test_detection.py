import cv2

image = cv2.imread('imagens/pessoas.jpg')

detector = cv2.CascadeClassifier('cascade/fullbody.xml')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Pessoas", image_gray)

detections = detector.detectMultiScale(image_gray)

print(detections)
print(len(detections))

for (x, y, l, a) in detections:
    cv2.rectangle(image, (x, y), (x + l, y + a), (0,255,0), 2)

cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()