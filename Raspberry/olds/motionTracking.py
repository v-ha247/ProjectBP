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

frame2 = cam.get_frame()
old_time = time.time()
while True:
    frame1 = cam.get_frame()
    
    run, old_time = check_time(0.2, old_time)
    if run:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray1, gray2)
        blur = cv2.GaussianBlur(diff, (5, 5), 0) 
        _ , thresholdimg = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        
        dilated = cv2.dilate(thresholdimg, None, iterations=3)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion = False
        area = ()
        if contours != ():
            biggest_motion = 2000
            for c in contours:
                found_area = cv2.contourArea(c)   
                if found_area > biggest_motion:
                    biggest_motion = found_area
                    area = cv2.boundingRect(c)
                    motion = True       
            else:
                motion_center = ()
        if motion:
            (x, y, w, h) = area
            frame = frame1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
            servos.move(x, y, w, h)
    
    frame2 = frame1
    cv2.imshow("Frame", frame1)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()
