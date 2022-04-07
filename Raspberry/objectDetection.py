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

    def detect(self, frame):
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


class HogDescriptor:
    def __init__(self):
        print("HOG Descriptor Detector")
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def person_detect_hog(self, frame):
        bounding_box, weights = self.hog.detectMultiScale(frame, winStride = (4, 4))
        return bounding_box

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.person_detect_hog(gray)

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


class Yolo:
    def __init__(self):
        print("Yolo Detector")
        self.net = cv2.dnn.readNetFromDarknet("models/yolov3-tiny.cfg", "models/yolov3-tiny.weights")
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

        self.layer_names = self.net.getLayerNames()
        self.output_layers = []
        for i in self.net.getUnconnectedOutLayers():
            self.output_layers.append(self.layer_names[i - 1])

    def detect(self, frame):
        Height = frame.shape[0]
        Width = frame.shape[1]
        center_res_x = Height / 2
        center_res_y = Width / 2

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416), (0,0,0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
    
        class_IDs = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:

                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)

                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2

                    class_IDs.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.6)

        detections = []
        for i in indices:
            box = boxes[i]
            if class_IDs[i] == self.person_id:
                (x, y, w, h) = box
                detections.append([int(x), int(y), int(w), int(h)]) 

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

    def detect(self, frame):
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