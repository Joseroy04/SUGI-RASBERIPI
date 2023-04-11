import cv2
import numpy as np

# Load the pre-trained YOLOv4-tiny model
model = cv2.dnn.readNetFromDarknet('yolov3-tiny.cfg', 'yolov3-tiny.weights')

# Load the class names
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Set the input image size for the model
# model.setInputShape(416, 416)

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
    class_ids = []
    boxes = []
    confidences = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and class_id == 0:
                # Extract the bounding box
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (center_x, center_y, width, height) = box.astype("int")
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        # i = i[0]
        (x, y, w, h) = boxes[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with detected persons
    cv2.imshow('Person detection', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
