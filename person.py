import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

# Adjust these parameters for optimization
resize_factor = 0.5  # Resize factor for input frames
confidence_threshold = 0.5  # Minimum confidence for detections
frame_skip = 3  # Process every nth frame

while cap.isOpened():
    for _ in range(frame_skip):
        ret, frame = cap.read()
        if not ret:
            break
    
    if not ret:
        break

    height, width, _ = frame.shape
    resized_frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)

    blob = cv2.dnn.blobFromImage(resized_frame, scalefactor=1/255, size=(320, 320), swapRB=True, crop=False)

    net.setInput(blob)
    layer_outputs = net.forward(net.getUnconnectedOutLayersNames())

    class_ids = []
    confidences = []
    boxes = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

    for i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f'{label} {confidence}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Person Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
