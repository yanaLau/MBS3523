import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    flipped_h = cv2.flip(frame, 1)
    flipped_v = cv2.flip(frame, 0)
    flipped_both = cv2.flip(frame, -1)

    top_row = np.hstack((frame, flipped_h))
    bottom_row = np.hstack((flipped_v, flipped_both))
    combined = np.vstack((top_row, bottom_row))

    cv2.imshow('Webcam Frame Orientations', combined)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()