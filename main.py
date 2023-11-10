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

    if hands:
        # Get landmarks of the first hand
        lmList = hands[0]["lmList"]
        cursor = lmList[8]

        # Get the coordinates of the index finger tip
        x1, y1 = lmList[8][0], lmList[8][1] # Index 8 corresponds to the index fingertip
        x2, y2 = lmList[4][0], lmList[4][1] # Index 4 corresponds to the thumb fingertip

        # Display the connecting lines on the image
        length, info, img = detector.findDistance((x1, y1), (x2, y2), img, (255, 255, 255), 5)

        # Check if the fingers are close enough to complete a CLICK action
        if length < 50:
            print("Clicked")
        else:
            print("Not")


    cv2.imshow("Img", img)
    cv2.waitKey(1)
