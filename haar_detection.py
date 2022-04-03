import numpy as np
import cv2

# haar cascades
frontalface_default_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
frontalface_alt2_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
profileface_cascade = cv2.CascadeClassifier('data/haarcascade_profileface.xml')
upperbody_cascade = cv2.CascadeClassifier('data/haarcascade_upperbody.xml')
fullbody_cascade = cv2.CascadeClassifier('data/haarcascade_fullbody.xml')

cap = cv2.VideoCapture(0)

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


while(True):
    ret, frame = cap.read()
    if ret != True:
        print('no frame captured')
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    persons = person_detect_haar(gray)

    for (x, y, w, h) in persons:        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2) # color in BGR not RGB, idk y

    cv2.imshow('frame', cv2.resize(frame, (1280, 720)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()