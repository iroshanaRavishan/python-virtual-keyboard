import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    img, bbox = cvzone.putTextRect(img, "How Are You...?", [100, 100], 2, 2, offset=50, border=5)
    cv2.imshow("Img", img)
    cv2.waitKey(1)
