import numpy as np
import cv2
import time
import camControl
import pantiltControl

# haar descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def check_time(duration, old_time):
    current_time = time.time()
    if current_time - old_time > duration:
        old_time = time.time()
        return True, old_time
    return False, old_time

def person_detect_hog(img):
    bounding_box, weights = hog.detectMultiScale(img, winStride = (4, 4))
    return bounding_box

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
    
    run, old_time = check_time(0.1, old_time)
    if run:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bounding_box = person_detect_hog(frame)
        
        person = False
        for (x, y, w, h) in bounding_box:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
            fx = x
            fy = y
            fw = w
            fh = h
            person = True
        if person:
            servos.move(fx, fy, fw, fh)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()
