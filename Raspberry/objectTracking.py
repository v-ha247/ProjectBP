import cv2
import time
import objectDetection
import controlCam
import controlServos

def check_time(duration, old_time):
    current_time = time.time()
    if current_time - old_time > duration:
        return True, time.time()
    return False, old_time

detection_types = ['VIOLAJONES', 'HOGDESCRIPTOR','YOLO', 'SSD']
detection_type = detection_types[3]

tracker_types = ['BOOSTING', 'TLD', 'MEDIANFLOW', 'MOSSE']
tracker_type = tracker_types[2]

def init_detector(detection_type):
    if detection_type == 'VIOLAJONES':
        detector = objectDetection.ViolaJones()
    if detection_type == 'HOGDESCRIPTOR':
        detector = objectDetection.HogDescriptor()
    if detection_type == 'YOLO':
        detector = objectDetection.Yolo()
    if detection_type == 'SSD':
        detector = objectDetection.SSD()
    return detector

def init_tracker(tracker_type):
    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    if tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create() 
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create() 
    if tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    return tracker

pan_pin = 17
tilt_pin = 27
cam_width = 640
cam_height = 480
pan_angle = 90
tilt_angle = 50

servos = controlServos.Pantilt(pan_pin, tilt_pin, cam_width, cam_height, pan_angle, tilt_angle)

cam = controlCam.Camera((cam_width, cam_height))
cam.vertical_flip()
cam.start()

detecting = True
tracking = False
fps = 0

print("\nGet ready")
time.sleep(3)

last_tracking = time.time()
detector = init_detector(detection_type)
tracker = init_tracker(tracker_type)
move_time = time.time()
while True:
    frame = cam.get_frame()
    
    if detecting:
        timer = cv2.getTickCount()
        roi = detector.detect(frame)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        if roi[0] != -1:
            (x, y, w, h) = roi
            detecting = False

    if not detecting and not tracking:
        (x, y, w, h) = roi       
        ret = tracker.init(frame, (x, y, w, h))
        if ret:
            tracking = True
            time.sleep(1)
        else:
            detecting = True
    
    if tracking:
        timer = cv2.getTickCount()
        ret, bbox = tracker.update(frame)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        if ret:
            (x, y, w, h) = bbox
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255,192,203), 2)
            if not servos.get_status():
                servos.move(int(x), int(y), int(w), int(h))
            last_tracking = time.time()
        else:
            cv2.putText(frame, "Tracking failure detected", (10,140), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 2)
            reset, _ = check_time(2, last_tracking)
            if reset:
                servos.move_default()
                detecting = True
                tracking = False
                tracker = init_tracker(tracker_type)

    cv2.putText(frame, detection_type + " Detector", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 2)
    cv2.putText(frame, tracker_type + " Tracker", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 2)
    cv2.putText(frame, "FPS : " + str(int(fps)), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 2)
    if detecting:
        cv2.putText(frame, "Detecting", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 2)
    if tracking:
        cv2.putText(frame, "Tracking", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()
