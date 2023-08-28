import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.saved_model.load('saved_model_dir/saved_model')


# Load COCO class labels
classes = {}
with open('coco_labels.txt', 'r') as f:
    classes = f.read().splitlines()

# Open the video file
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(input_image)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = model(input_tensor)

    for detection in detections['detection_boxes'][0]:
        if detection[2] > 0.5:  # Confidence threshold
            y_min, x_min, y_max, x_max = detection
            y_min = int(y_min * frame.shape[0])
            x_min = int(x_min * frame.shape[1])
            y_max = int(y_max * frame.shape[0])
            x_max = int(x_max * frame.shape[1])
            
            class_id = int(detection[1])
            label = classes[class_id]

            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
