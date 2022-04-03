import numpy as np
import cv2
import time
import camControl
import pantiltControl

def check_time(duration, old_time):
    current_time = time.time()
    if current_time - old_time > duration:
        old_time = time.time()
        return True, old_time
    return False, old_time

net = cv2.dnn.readNetFromDarknet("models/yolov3-tiny.cfg", "models/yolov3-tiny.weights")
classes = []
with open("models/coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]
person_id = 0

layer_names = net.getLayerNames()
output_layers = []
for i in net.getUnconnectedOutLayers():
    output_layers.append(layer_names[i - 1])

pan_pin = 17
tilt_pin = 27
cam_width = 320
cam_height = 208
pan_angle = 90
tilt_angle = 50

servos = pantiltControl.Pantilt(pan_pin, tilt_pin, cam_width, cam_height, pan_angle, tilt_angle)

cam = camControl.Camera((cam_width, cam_height))
cam.vertical_flip()
cam.start()
time.sleep(2)

old_time = time.time()
while(True):
    frame = cam.get_frame()
    
    run, old_time = check_time(0.5, old_time)
    if run:
        # Height = frame.shape[0]
        # Width = frame.shape[1]
        
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

                    center_x = int(detection[0] * cam_width)
                    center_y = int(detection[1] * cam_height)

                    w = int(detection[2] * cam_width)
                    h = int(detection[3] * cam_height)
                    x = center_x - w / 2
                    y = center_y - h / 2

                    class_IDs.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.6)
        
        box = None
        person = False
        for i in indices:
            if class_IDs[i] == person_id:
                box = boxes[i]
                print('person')
                person = True
        if person:
            (x, y, w, h) = box
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 0), 2)
            servos.move(int(x), int(y), int(w), int(h))

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    
cam.stop()
cv2.destroyAllWindows()

