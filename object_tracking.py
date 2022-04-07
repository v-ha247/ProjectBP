import cv2
import object_detection
import time

def check_time(duration, old_time):
    current_time = time.time()
    if current_time - old_time > duration:
        return True
    return False

detection_types = ['VIOLAJONES', 'HOGDESCRIPTOR','YOLO', 'SSD']
detection_type = detection_types[3]

# tracker_types = ['BOOSTING', 'TLD', 'MEDIANFLOW', 'MOSSE']
# tracker_type = tracker_types[3]
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[6]

def init_detector(detection_type):
    if detection_type == 'VIOLAJONES':
        detector = object_detection.ViolaJones()
    if detection_type == 'HOGDESCRIPTOR':
        detector = object_detection.HogDescriptor()
    if detection_type == 'YOLO':
        detector = object_detection.Yolo()
    if detection_type == 'SSD':
        detector = object_detection.SSD()
    return detector

def init_tracker(tracker_type):
    # if tracker_type == 'BOOSTING':
    #     tracker = cv2.legacy.TrackerBoosting_create()
    # if tracker_type == 'TLD':
    #     tracker = cv2.legacy.TrackerTLD_create() 
    # if tracker_type == 'MEDIANFLOW':
    #     tracker = cv2.legacy.TrackerMedianFlow_create() 
    # if tracker_type == 'MOSSE':
    #     tracker = cv2.legacy.TrackerMOSSE_create()
    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create() 
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create() 
    if tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create() 
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create() 
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    return tracker

cap = cv2.VideoCapture(0)

detecting = True
tracking = False
# reset = False
fps = 0

time.sleep(3)
print("Get ready")

last_tracking = time.time()
detector = init_detector(detection_type)
tracker = init_tracker(tracker_type)
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    if detecting:
        timer = cv2.getTickCount()
        roi = detector.detect(frame)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        if roi[0] != -1:
            (x, y, w, h) = roi
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255,192,203), 2)
            detecting = False

    if not detecting and not tracking:
        (x, y, w, h) = roi       
        ret = tracker.init(frame, (x, y, w, h))
        if ret:
            tracking = True
        else:
            detecting = True

    if tracking:
        timer = cv2.getTickCount()
        ret, bbox = tracker.update(frame)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        if ret:
            (x, y, w, h) = bbox
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255,192,203), 2)
            last_tracking = time.time()
        else:
            cv2.putText(frame, "Tracking failure detected", (10,140), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 2)
            reset = check_time(5, last_tracking)
            if reset:
                detecting = True
                tracking = False
                tracker = init_tracker(tracker_type)

    cv2.putText(frame, detection_type + " Detector", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 1)
    cv2.putText(frame, tracker_type + " Tracker", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 1)
    cv2.putText(frame, "FPS : " + str(int(fps)), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 1)
    if detecting:
        cv2.putText(frame, "Detecting", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 1)
    if tracking:
        cv2.putText(frame, "Tracking", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 1)
    cv2.imshow("Frame", cv2.resize(frame, (1280, 720)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('r'):
        old_time = time.time()
        print("restarting...")
        while True:
            reset = check_time(3, old_time)
            if reset:
                detecting = True
                tracking = False
                tracker = init_tracker(tracker_type)
                break

cap.release()
