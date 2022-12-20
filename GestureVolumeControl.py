import numpy as np
import cv2
import time
import mediapipe as mp

captureDevice = cv2.VideoCapture(0) #captureDevice = camera

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0 
while True:
    ret, frame = captureDevice.read() 
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame,f'FPS:{int(fps)}',(40,50),cv2.FONT_HERSHEY_COMPLEX,1, \
           (255,0,0),3 )

    cv2.imshow('my frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captureDevice.release()
cv2.destroyAllWindows()