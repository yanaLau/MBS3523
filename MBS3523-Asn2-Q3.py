# YOLO v3

import cv2
import numpy as np
from tensorboard.compat.tensorflow_stub.dtypes import double

confThreshold = 0.8

cam = cv2.VideoCapture(0)

# Create an empty list - classes[] and point the classesFile to 'coco80.names'
classesFile = 'coco80.names'
classes = []
# Load all classes in coco80.names into classes[]
with open(classesFile, 'r') as f:
    classes = f.read().splitlines()
    print(classes)
    print(len(classes))

# Load the configuration and weights file
# You need to download the weights and cfg files from https://pjreddie.com/darknet/yolo/
net = cv2.dnn.readNetFromDarknet('yolov3-608.cfg','yolov3-608.weights')
# Use OpenCV as backend and use CPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

fruit_counts = {'banana': 0, 'apple': 0, 'orange': 0}
fruit_prices = {'banana': 1.0, 'apple': 1.5, 'orange': 0.8}

while True:
    # success , img = cam.read()
    img = cv2.imread("fruit.png")

    height, width, ch = img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    print(layerNames)

    output_layers_names = net.getUnconnectedOutLayersNames()
    print(output_layers_names)

    LayerOutputs = net.forward(output_layers_names)
    print(len(LayerOutputs))
    # print(LayerOutputs[0].shape)
    # print(LayerOutputs[1].shape)
    # print(LayerOutputs[2].shape)
    # print(LayerOutputs[0][0])


    bboxes = [] # array for all bounding boxes of detected classes
    confidences = [] # array for all confidence values of matching detected classes
    class_ids = [] # array for all class IDs of matching detected classes

    for output in LayerOutputs:
        for detection in output:
            scores = detection[5:] # omit the first 5 values
            class_id = np.argmax(scores) # find the highest score ID out of 80 values which has the highest confidence value
            confidence = scores[class_id]
            if confidence > confThreshold:
                center_x = int(detection[0]*width) #YOLO predicts centers of image
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                bboxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
                # cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)

    # print(len(bboxes))
    indexes = cv2.dnn.NMSBoxes(bboxes, confidences, confThreshold, 0.4) #Non-maximum suppresion
    # print(indexes)
    # print(indexes.flatten())

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255,size=(len(bboxes),3))
    n=0
    z=0
    p=0
    if len(indexes) > 0:
        for i in indexes.flatten():
            if class_ids[i] == 46:
                x, y, w, h = bboxes[i]
                label1 = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                if label1 == 'banana':
                    n = n + 1
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label1 + " " + "[" + str(n) + "]  " + confidence, (x, y + 20), font, 1,
                                (255, 255, 255), 2)
        for a in indexes.flatten():
            if class_ids[a] == 47:
                x, y, w, h = bboxes[a]
                label2 = str(classes[class_ids[a]])
                confidence = str(round(confidences[a], 2))
                color = colors[a]
                if label2 == 'apple':
                    z = z + 1
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label2 + " " + "[" + str(z) + "]  " + confidence, (x, y + 20), font, 1,
                                (255, 255, 255), 2)
        for o in indexes.flatten():
            if class_ids[o] == 49:
                x, y, w, h = bboxes[o]
                label3 = str(classes[class_ids[o]])
                confidence = str(round(confidences[o], 2))
                color = colors[o]
                if label3 == 'orange':
                    p = p + 1
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label3 + " " + "[" + str(p) + "]  " + confidence, (x, y + 20), font, 1,
                                (255, 255, 255), 2)
        cv2.putText(img, "There are: " + str(n) + " " + label1, (200, 20), font, 2, (0, 0, 255), 2)
        cv2.putText(img, "There are: " + str(z) + " " + label2, (200, 40), font, 2, (0, 0, 255), 2)
        cv2.putText(img, "There are: " + str(p) + " " + label3, (200, 60), font, 2, (0, 0, 255), 2)
        f= n * 2
        g = p * 4
        h = z * 3
        total_price = f+g+h
        cv2.putText(img, "Total price " + str(total_price), (200, 80), font, 2, (0, 0, 255), 2)

    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xff == 27:
        break

cam.release()
cv2.destroyAllWindows()

