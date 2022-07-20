import cv2
import numpy as np


confid = 0.5
thresh = 0.5




def detect_pedestrians(frame, net,ln1,H,W):
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
   # start = time.time()
    layerOutputs = net.forward(ln1)
   # end = time.time()
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filtering out only persons in frame
            if classID == 0:

                if confidence > confid:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)
    font = cv2.FONT_HERSHEY_PLAIN
    boxes1 = []
    for i in range(len(boxes)):
        if i in idxs:
            boxes1.append(boxes[i])
            x, y, w, h = boxes[i]
    return boxes1