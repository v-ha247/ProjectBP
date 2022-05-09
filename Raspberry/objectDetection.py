import numpy as np
import cv2
import math

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

        center_res_x = frame.shape[0] / 2
        center_res_y = frame.shape[1] / 2

        roi = [-1]
        if detections != ():
            roi = detections[0]
            x = abs(center_res_x - ((roi[0] + roi[2]) / 2))
            y = abs(center_res_y - ((roi[1] + roi[3]) / 2))
            closest = math.sqrt(x*x + y*y)        
            for (x, y, w, h) in detections:
                x = abs(center_res_x - ((x + w) / 2))
                y = abs(center_res_y - ((y + h) / 2))
                distance = math.sqrt(x*x + y*y)
                if closest > distance:
                    closest = distance
                    roi = [int(x), int(y), int(w), int(h)]

        return roi


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
        center_res_x = Height / 2
        center_res_y = Width / 2
        
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
       
        roi = [-1]
        if detections:
            roi = detections[0]
            x = abs(center_res_x - ((roi[0] + roi[2]) / 2))
            y = abs(center_res_y - ((roi[1] + roi[3]) / 2))
            closest = math.sqrt(x*x + y*y)        
            for (x, y, w, h) in detections:
                x = abs(center_res_x - ((x + w) / 2))
                y = abs(center_res_y - ((y + h) / 2))
                distance = math.sqrt(x*x + y*y)
                if closest > distance:
                    closest = distance
                    roi = [int(x), int(y), int(w), int(h)]

        return roi

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
        Height = frame.shape[0]
        Width = frame.shape[1]
        center_res_x = Height / 2
        center_res_y = Width / 2

        class_IDs, scores, boxes = self.model.detect(frame, self.confidence_threshold, self.nms_threshold)
    
        detections = []
        for (class_id, score, box) in zip(class_IDs, scores, boxes):
            if class_id == self.person_id:
                detections.append(box)
        
        roi = [-1]
        if detections:
            roi = detections[0]
            x = abs(center_res_x - ((roi[0] + roi[2]) / 2))
            y = abs(center_res_y - ((roi[1] + roi[3]) / 2))
            closest = math.sqrt(x*x + y*y)        
            for (x, y, w, h) in detections:
                x = abs(center_res_x - ((x + w) / 2))
                y = abs(center_res_y - ((y + h) / 2))
                distance = math.sqrt(x*x + y*y)
                if closest > distance:
                    closest = distance
                    roi = [int(x), int(y), int(w), int(h)]

        return roi

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
        center_res_x = Height / 2
        center_res_y = Width / 2

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

        roi = [-1]
        if detections:
            roi = detections[0]
            x = abs(center_res_x - ((roi[0] + roi[2]) / 2))
            y = abs(center_res_y - ((roi[1] + roi[3]) / 2))
            closest = math.sqrt(x*x + y*y)        
            for (x, y, w, h) in detections:
                x = abs(center_res_x - ((x + w) / 2))
                y = abs(center_res_y - ((y + h) / 2))
                distance = math.sqrt(x*x + y*y)
                if closest > distance:
                    closest = distance
                    roi = [int(x), int(y), int(w), int(h)]

        return roi

