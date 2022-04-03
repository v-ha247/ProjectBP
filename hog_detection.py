import numpy as np
import cv2
import imutils

# haar descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture("D:/System/OneDrive/Obrázky/silvestr/20181231_205201.mp4")
# cap = cv2.VideoCapture(0)

def person_detect_hog(img):
    bounding_box, weights = hog.detectMultiScale(img, winStride = (4, 4))
    return bounding_box

while(True):
    ret, frame = cap.read()
    if ret != True:
        print('no frame captured')
        continue
    
    # frame = cv2.resize(frame, (320, 208))   
    frame = imutils.resize(frame, width=min(400, frame.shape[1]))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bounding_box = person_detect_hog(frame)
    
    for (x, y, w, h) in bounding_box:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)

    cv2.imshow('frame', cv2.resize(frame, (1280, 720)))
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()