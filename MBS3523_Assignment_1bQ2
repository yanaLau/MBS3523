import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    width = cam.get(3)
    height = cam.get(4)
    print(height,width)

    Canny  = cv2.Canny(frame,100,100)
    HSV= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    Blurred = cv2.GaussianBlur(frame, (15, 15), 0)

    cv2.imshow('cam', frame)
    cv2.imshow('canny', Canny)
    cv2.imshow('HSV', HSV)
    cv2.imshow('Blurred', Blurred)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()