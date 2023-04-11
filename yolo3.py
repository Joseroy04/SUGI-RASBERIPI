import cv2
import time
import RPi.GPIO as GPIO
import numpy as np

GPIO.setmode(GPIO.BOARD)
GPIO.setup(8, GPIO.OUT)

# Load the pre-trained YOLOv4-tiny model
model = cv2.dnn.readNetFromDarknet('yolov3-tiny.cfg', 'yolov3-tiny.weights')

# Set the mean values for normalization
mean_values = (0, 0, 0)
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# Create a VideoCapture object
cap = cv2.VideoCapture(0)


# Get the first frame
ret, frame1 = cap.read()

while True:
    # Get the next frame
    ret, frame = cap.read()

    # Convert the frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between the frames
    diff = cv2.absdiff(gray1, gray2)

    # Threshold the difference image to highlight the moving pixels
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilate the thresholded image to fill in any gaps
    dilated = cv2.dilate(thresh, None, iterations=2)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over the contours and draw a bounding box around any that are large enough
    for contour in contours:
        if cv2.contourArea(contour) > 5000:
            # Perform person detection using the YOLOv4-tiny model
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            model.setInput(blob)
            output_layers = model.getUnconnectedOutLayersNames()
            outputs = model.forward(output_layers)

            # Process the output to extract detected objects and their bounding boxes
            conf_threshold = 0.8
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > conf_threshold and class_id == 0:
                        GPIO.output(8,True)
                        time.sleep(5)
                        GPIO.output(8,False)
            
    # Set frame1 to be frame2 for the next iteration
    frame1 = frame
