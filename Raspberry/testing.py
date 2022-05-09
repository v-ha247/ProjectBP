import cv2
import time
import objectDetection
import numpy as np

class ViolaJones:
    def __init__(self):
        print("Viola Jones Detector")
        self.frontalface_default_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
        self.frontalface_alt2_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
        self.profileface_cascade = cv2.CascadeClassifier('data/haarcascade_profileface.xml')
        self.upperbody_cascade = cv2.CascadeClassifier('data/haarcascade_upperbody.xml')
        self.fullbody_cascade = cv2.CascadeClassifier('data/haarcascade_fullbody.xml')

    def person_detect_haar(self, img, scaleFactor=1.5, minNeighbors=5):
        fb_bodys = self.fullbody_cascade.detectMultiScale(img, scaleFactor, minNeighbors)
        if fb_bodys == ():
            ub_bodys = self.upperbody_cascade.detectMultiScale(img, scaleFactor, minNeighbors)
            if ub_bodys == ():
                pf_faces = self.profileface_cascade.detectMultiScale(img, scaleFactor, minNeighbors)
                if pf_faces == ():
                    ffa2_faces = self.frontalface_alt2_cascade.detectMultiScale(img, scaleFactor, minNeighbors)
                    if ffa2_faces == ():
                        ffd_faces = self.frontalface_default_cascade.detectMultiScale(img, scaleFactor, minNeighbors)
                        return ffd_faces
                    else:
                        return ffa2_faces
                else:
                    return pf_faces
            else:
                return ub_bodys
        else:
            return fb_bodys

    def detect_object(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.person_detect_haar(gray)

        return detections


class MaskRCNN:
    def __init__(self):
        print("MaskRCNN Detector")
        self.net = cv2.dnn.readNetFromTensorflow("models/frozen_inference_graph.pb", "models/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
        self.classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
         'traffic' 'light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 
         'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 
         'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball' 'glove', 
         'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 
         'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 
         'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 
         'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 
         'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.person_id = 0


    def detect_object(self, frame):
        Height = frame.shape[0]
        Width = frame.shape[1]
        
        self.net.setInput(cv2.dnn.blobFromImage(frame, swapRB=True, crop=False))
        outs, mask = self.net.forward(["detection_out_final","detection_masks"])

        detections = []
        for i in range(0, outs.shape[2]):
            class_id = int(outs[0, 0, i, 1])
            confidence = outs[0, 0, i, 2]
            if confidence > 0.5 and class_id == self.person_id:
                box = outs[0, 0, i, 3:7] * np.array([Width, Height, Width, Height])
                (x1, y1, x2, y2) = box.astype("int")
                detections.append([x1, y1, x2 - x1, y2 - y1])
       
        return detections

class Yolo:
    def __init__(self):
        print("Yolo Detector")
        self.net = cv2.dnn.readNet("models/yolov4-tiny.weights", "models/yolov4-tiny.cfg")
        self.classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
         'traffic' 'light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 
         'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 
         'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball' 'glove', 
         'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 
         'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 
         'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 
         'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 
         'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.person_id = 0
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.5

        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1/float(255.0), swapRB=True)

    def detect_object(self, frame):
        class_IDs, scores, boxes = self.model.detect(frame, self.confidence_threshold, self.nms_threshold)
    
        detections = []
        for (class_id, score, box) in zip(class_IDs, scores, boxes):
            if class_id == self.person_id:
                detections.append(box)

        return detections

class SSD:
    def __init__(self):
        print("SSD Detector")
        self.net = cv2.dnn.readNetFromCaffe('models/MobileNetSSD_deploy.prototxt', 'models/MobileNetSSD_deploy.caffemodel')
        self.classes =  ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", 
         "diningtable",  "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        self.person_id = 15

    def detect_object(self, frame):
        Height = frame.shape[0]
        Width = frame.shape[1]

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        outs = self.net.forward()

        detections = []
        for i in range(0, outs.shape[2]):
            confidence = outs[0, 0, i, 2]
            if confidence > 0.5:
                id = int(outs[0, 0, i, 1])
                box = outs[0, 0, i, 3:7] * np.array([Width, Height, Width, Height])     
                if id == self.person_id:
                    (x1, y1, x2, y2) = box.astype("int")
                    detections.append([x1, y1, x2 - x1, y2 - y1])

        return detections

def test_detection_time(image, imgname, detection_type, iterations=1):
    if detection_type == 'VIOLAJONES':
        detector = ViolaJones()
    elif detection_type == 'MASKRCNN':
        detector = MaskRCNN()
    elif detection_type == 'YOLO':
        detector = Yolo()
    elif detection_type == 'SSD':
        detector = SSD()
    else:
        return

    for i in range(iterations):
        frame = cv2.imread(image)        
        start = time.time()
        detections = detector.detect_object(frame)
        duration = time.time() - start

        for roi in detections:
            (x, y, w, h) = roi
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0,0,0), 2)
        else:
            cv2.putText(frame, detection_type + " Detector", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        cv2.putText(frame, detection_type + " Detector", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.imshow("Frame", cv2.resize(frame, (1280, 720)))
        duration = round(duration, 3)
        print(f'{detection_type} = {i} - time: {duration} seconds')

        #cv2.imwrite(f'{detection_type}_{imgname}-{i}.jpg', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def test_tracking(video, videoname, tracker_type):
    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create() 
    elif tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create() 
    elif tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    else:
        return 
    detector = objectDetection.SSD()

    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter(f'{tracker_type}_{videoname}-test.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

    detecting = True
    tracking = False
    frames = []
    while True:
        ret, frame = cap.read()
        if ret:             
            if detecting:
                print("Detecting")
                roi = detector.detect_object(frame)
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
                    cv2.putText(frame, tracker_type + " Tracker", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                else:
                    cv2.putText(frame, "Tracking failure detected", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 2)
                frames.append(int(fps))
                                                              
            out.write(frame)
            cv2.imshow('frame',frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break            
        else:
            break
    
    allframes = len(frames)
    avg = sum(frames) / allframes

    print(f'video:{videoname}, avg:{avg}, frames:{allframes}')

    cap.release()
    out.release()

    cv2.destroyAllWindows()



image1 = 'D:/System/OneDrive/school/BP/ProjectBP/ProjectBP/Raspberry/test/Ja-polariod-664x927.jpg'
image2 = 'D:/System/OneDrive/school/BP/ProjectBP/ProjectBP/Raspberry/test/lidi-839x229.jpg'
image3 = 'D:/System/OneDrive/school/BP/ProjectBP/ProjectBP/Raspberry/test/omar-lopez-unsplash-1037x657.jpg'
image4 = 'D:/System/OneDrive/school/BP/ProjectBP/ProjectBP/Raspberry/test/selfie-640x853.jpg'

video1 = 'D:/System/OneDrive/school/BP/ProjectBP/ProjectBP/Raspberry/test/vid1.mp4'
video2 = 'D:/System/OneDrive/school/BP/ProjectBP/ProjectBP/Raspberry/test/vid2.mp4'
video3 = 'D:/System/OneDrive/school/BP/ProjectBP/ProjectBP/Raspberry/test/vid3.mp4'
video4 = 'D:/System/OneDrive/school/BP/ProjectBP/ProjectBP/Raspberry/test/vid4.mp4'

test_tracking(video1, 'video1', 'BOOSTING')
test_tracking(video1, 'video1', 'MEDIANFLOW')
test_tracking(video1, 'video1', 'TLD')
test_tracking(video1, 'video1', 'MOSSE')

test_tracking(video2, 'video2', 'BOOSTING')
test_tracking(video2, 'video2', 'MEDIANFLOW')
test_tracking(video2, 'video2', 'TLD')
test_tracking(video2, 'video2', 'MOSSE')

test_tracking(video3, 'video3', 'BOOSTING')
test_tracking(video3, 'video3', 'MEDIANFLOW')
test_tracking(video3, 'video3', 'TLD')
test_tracking(video3, 'video3', 'MOSSE')

test_tracking(video4, 'video4', 'BOOSTING')
test_tracking(video4, 'video4', 'MEDIANFLOW')
test_tracking(video4, 'video4', 'TLD')
test_tracking(video4, 'video4', 'MOSSE')

test_detection_time(image1, 'Ja-polariod-664x927', 'VIOLAJONES', 10)
test_detection_time(image2, 'lidi-839x229', 'VIOLAJONES', 10)
test_detection_time(image3, 'omar-lopez-unsplash-1037x657', 'VIOLAJONES', 10)
test_detection_time(image4, 'selfie-640x853', 'VIOLAJONES', 10)

test_detection_time(image1, 'Ja-polariod-664x927', 'MASKRCNN', 10)
test_detection_time(image2, 'lidi-839x229', 'MASKRCNN', 10)
test_detection_time(image3, 'omar-lopez-unsplash-1037x657', 'MASKRCNN', 10)
test_detection_time(image4, 'selfie-640x853', 'MASKRCNN', 10)

test_detection_time(image1, 'Ja-polariod-664x927', 'YOLO', 10)
test_detection_time(image2, 'lidi-839x229', 'YOLO', 10)
test_detection_time(image3, 'omar-lopez-unsplash-1037x657', 'YOLO', 10)
test_detection_time(image4, 'selfie-640x853', 'YOLO', 10)

test_detection_time(image1, 'Ja-polariod-664x927', 'SSD', 10)
test_detection_time(image2, 'lidi-839x229', 'SSD', 10)
test_detection_time(image3, 'omar-lopez-unsplash-1037x657', 'SSD', 10)
test_detection_time(image4, 'selfie-640x853', 'SSD', 10)
