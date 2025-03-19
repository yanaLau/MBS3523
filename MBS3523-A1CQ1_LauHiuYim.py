import serial
import cv2
import numpy as np

ser = serial.Serial('COM8', 9600)

#initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #read the temperature data from Arduino
    if ser.in_waiting:
        temperature = ser.readline().decode('utf-8').strip()

    #displaying the temperature on the frame
    cv2.putText(frame, f'Indoor temperature: {temperature} degree C', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #show the webcam frame
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()