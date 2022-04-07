import numpy as np
import cv2
import time
import camControl as camControl
import pantiltControl as pantiltControl

def person_detect_haar_body(img, scaleFactor=1.4, minNeighbors=2):
    ffd_faces = frontalface_default_cascade.detectMultiScale(img, scaleFactor, minNeighbors)
    if ffd_faces == ():
        ffa2_faces = frontalface_alt2_cascade.detectMultiScale(img, scaleFactor, minNeighbors)
        if ffa2_faces == ():
            pf_faces = profileface_cascade.detectMultiScale(img, scaleFactor, minNeighbors)
            if pf_faces == ():
                ub_bodys = upperbody_cascade.detectMultiScale(img, scaleFactor, minNeighbors)
                if ub_bodys == ():
                    fb_bodys = fullbody_cascade.detectMultiScale(img, scaleFactor, minNeighbors)
                    if fb_bodys != ():
                        print('fullbody_cascade')
                    return fb_bodys
                else:
                    print('upperbody_cascade')
                    return ub_bodys
            else:
                print('profileface_cascade')
                return pf_faces
        else:
            print('frontalface_alt2_cascade')
            return ffa2_faces
    else:
        print('frontalface_default_cascade')
        return ffd_faces

def person_detect_haar(img, scaleFactor=1.4, minNeighbors=2):
    ffd_faces = frontalface_default_cascade.detectMultiScale(img, scaleFactor, minNeighbors)
    if ffd_faces == ():
        ffa2_faces = frontalface_alt2_cascade.detectMultiScale(img, scaleFactor, minNeighbors)
        if ffa2_faces == ():
            pf_faces = profileface_cascade.detectMultiScale(img, scaleFactor, minNeighbors)
            if pf_faces != ():
                print('profileface_cascade')
            return pf_faces
        else:
            print('frontalface_alt2_cascade')
            return ffa2_faces
    else:
        print('frontalface_default_cascade')
        return ffd_faces


def check_time(duration, old_time):
    current_time = time.time()
    if current_time - old_time > duration:
        old_time = time.time()
        return True, old_time
    return False, old_time


# haar cascades
frontalface_default_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')
frontalface_alt2_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_alt2.xml')
profileface_cascade = cv2.CascadeClassifier('../data/haarcascade_profileface.xml')
upperbody_cascade = cv2.CascadeClassifier('../data/haarcascade_upperbody.xml')
fullbody_cascade = cv2.CascadeClassifier('../data/haarcascade_fullbody.xml')

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
        persons = person_detect_haar_body(gray)
    
        if persons != ():
            for (x, y, w, h) in persons:        
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
                fx = x
                fy = y
                fw = w
                fh = h
            servos.move(fx, fy, fw, fh)
        
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    

cam.stop()
cv2.destroyAllWindows()
