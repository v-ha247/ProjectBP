import numpy as np
import cv2
import time
import camControl as camControl
import pantiltControl as pantiltControl

def check_time(duration, old_time):
    current_time = time.time()
    if current_time - old_time > duration:
        old_time = time.time()
        return True, old_time
    return False, old_time

net = cv2.dnn.readNetFromCaffe('../models/MobileNetSSD_deploy.prototxt', '../models/MobileNetSSD_deploy.caffemodel')

classes =  ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", 
            "diningtable",  "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
person_id = 15

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
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        
        net.setInput(blob)
        detections = net.forward()
        
        bbox = None
        person = False
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                id = int(detections[0, 0, i, 1])

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                               
                if id == person_id:
                    print('person')
                    bbox = box.astype("int")
                    person = True
                
        if person:
            (x1, y1, x2, y2) = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            servos.move(x1, y1, x2 - x1, y2 - y1)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()
