import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
GPIO.setup(8, GPIO.OUT)


# Load the pre-trained YOLOv4-tiny model
model = cv2.dnn.readNetFromDarknet('yolov3-tiny.cfg', 'yolov3-tiny.weights')

# Set the mean values for normalization
mean_values = (0, 0, 0)
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Open the camera module
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Perform person detection using the YOLOv4-tiny model
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    output_layers = model.getUnconnectedOutLayersNames()
    outputs = model.forward(output_layers)

    # Process the output to extract detected objects and their bounding boxes
    conf_threshold = 0.8
    nms_threshold = 0.4
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and class_id == 0:
                GPIO.output(8,True)
                time.sleep(5)
                GPIO.output(8,False)
                

