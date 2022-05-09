import numpy as np
import imutils
import cv2

LABELS = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
         'traffic' 'light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 
         'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 
         'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball' 'glove', 
         'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 
         'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 
         'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 
         'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 
         'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

maxConfidence = 0.5
threshold = 0.85

weightsPath = "models/frozen_inference_graph.pb"
configPath = "models/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"

net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

cap = cv2.VideoCapture(0)
while True:
	_,frame = cap.read()
	frame = imutils.resize(frame, width=640, height=480)

	blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
	net.setInput(blob)
	(boxes, masks) = net.forward(["detection_out_final","detection_masks"])

	for i in range(0, boxes.shape[2]):
		classID = int(boxes[0, 0, i, 1])
		confidence = boxes[0, 0, i, 2]
		if confidence > maxConfidence and classID == 0:
			(H, W) = frame.shape[:2]
			box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
			(startX, startY, endX, endY) = box.astype("int")
			boxW = endX - startX
			boxH = endY - startY

			mask = masks[i, classID]
			mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_NEAREST)
			mask = (mask > threshold)

			roi = frame[startY:endY, startX:endX][mask]

			blended = ((0.4 * np.array([255, 0, 0])) + (0.6 * roi))
			frame[startY:endY, startX:endX][mask] = blended

			cv2.rectangle(frame, (startX, startY), (endX, endY), (255,0,0), 2)

			text = "{}: {:.4f}".format(LABELS[classID], confidence)
			cv2.putText(frame, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

	cv2.imshow('Video', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()