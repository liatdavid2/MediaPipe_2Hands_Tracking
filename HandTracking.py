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

# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(captureDevice.get(3))
frame_height = int(captureDevice.get(4))
   
size = (frame_width, frame_height)
# Below VideoWriter object will create
# a frame of above defined The output 
# is stored in 'filename.avi' file.
result = cv2.VideoWriter('HandTracking.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
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

    # Write the frame into the
    # file 'filename.avi'
    result.write(frame)

    cv2.imshow('my frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captureDevice.release()
result.release()
cv2.destroyAllWindows()