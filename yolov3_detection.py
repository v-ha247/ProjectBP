import cv2
import numpy as np

net = cv2.dnn.readNetFromDarknet("models/yolov3-tiny.cfg", "models/yolov3-tiny.weights")
# slower
# net = cv2.dnn.readNetFromDarknet("models/yolov3.cfg", "models/yolov3.weights")
classes = []
with open("models/coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]
person_id = 0

layer_names = net.getLayerNames()
output_layers = []
for i in net.getUnconnectedOutLayers():
    output_layers.append(layer_names[i - 1])

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    if ret != True:
        print('no frame captured')
        continue

    Height = frame.shape[0]
    Width = frame.shape[1]
    
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
 
    class_IDs = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:

                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)

                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2

                class_IDs.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.6)

    for i in indices:
        box = boxes[i]
        if class_IDs[i] == person_id:
            label = str(classes[class_IDs[i]]) 
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), (0, 0, 0), 2)
            cv2.putText(frame, label, (int(box[0]) - 10, int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imshow('frame', cv2.resize(frame, (1280, 720)))
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

