import cv2
import time

CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open("models/coco.names","r") as f:
    class_names = [line.strip() for line in f.readlines()]

vc = cv2.VideoCapture(0)
net = cv2.dnn.readNet("models/yolov4-tiny.weights", "models/yolov4-tiny.cfg")

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/float(255.0), swapRB=True) #float is important for Python version 2!!!

avg_FPS=0; count=0; total_fps=0;
while cv2.waitKey(1) < 1:
    (grabbed, frame) = vc.read()
    if not grabbed:
        break
    start = time.time()
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    end = time.time()

    FPS = 1 / (end - start)
    total_fps = total_fps + FPS; count=count+1;
    avg_FPS = total_fps / float(count)

    start_drawing = time.time()
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid], score*100)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    end_drawing = time.time()
    
    fps_label = "avg FPS: %.2f FPS: %.2f (excluding drawing %.2fms)" % (avg_FPS, 1 / (end - start), (end_drawing - start_drawing) * 1000)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 127), 2)
    cv2.imshow("detections", frame)
print(avg_FPS)