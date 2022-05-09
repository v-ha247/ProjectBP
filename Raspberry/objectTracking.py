import cv2
import time
import objectDetection
import controlCam
import controlServos

DEBUG = True

def check_time(duration, old_time):
    current_time = time.time()
    if current_time - old_time > duration:
        return True, time.time()
    return False, old_time

detection_types = ['VIOLAJONES', 'MASKRCNN','YOLO', 'SSD']
tracker_types = ['BOOSTING', 'MEDIANFLOW', 'TLD', 'MOSSE']

while True:
    num = int(input("Choose detection algorithm:\n1: Viola Jones\n2: Mask R-CNN\n3: YOLO\n4: SSD\n")) - 1
    if num in [0, 1, 2, 3]:
        detection_type = detection_types[num]
        break
    print("Invalid input")
while True:
    num = int(input("Choose tracking algorithm:\n1: BOOSTING\n2: MEDIANFLOW\n3: TLD\n4: MOSSE\n")) - 1
    if num in [0, 1, 2, 3]:
        tracker_type = tracker_types[num]
        break
    print("Invalid input")

def init_detector(detection_type):
    if detection_type == 'VIOLAJONES':
        detector = objectDetection.ViolaJones()
    if detection_type == 'MASKRCNN':
        detector = objectDetection.MaskRCNN()
    if detection_type == 'YOLO':
        detector = objectDetection.Yolo()
    if detection_type == 'SSD':
        detector = objectDetection.SSD()
    return detector

def init_tracker(tracker_type):
    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create() 
    if tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create() 
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

detector = init_detector(detection_type)
tracker = init_tracker(tracker_type)

print("\nGet ready")
time.sleep(3)

last_tracking = time.time()
move_time = time.time()
while True:
    frame = cam.get_frame()
    
    if detecting:
        timer = cv2.getTickCount()
        print("Detecting")
        roi = detector.detect_object(frame)
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
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0,0,0), 2)
            if not servos.moving():
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
    
    if DEBUG:
        cv2.putText(frame, detection_type + " Detector", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        cv2.putText(frame, tracker_type + " Tracker", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        cv2.putText(frame, "FPS : " + str(int(fps)), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        if detecting:
            cv2.putText(frame, "Detecting", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        if tracking:
            cv2.putText(frame, "Tracking", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()

