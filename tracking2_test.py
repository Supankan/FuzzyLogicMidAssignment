import cv2
import numpy as np

# Load YOLOv4 Tiny model using OpenCV DNN
config_path = "yolov4-tiny.cfg"  # Path to YOLO config file
weights_path = "yolov4-tiny.weights"  # Path to YOLO weights file
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class names
classes = open("coco.names").read().strip().split("\n")  # Path to COCO names file

# Initialize OpenCV MultiTracker
tracker = cv2.MultiTracker_create()

# Open the video file
video_path = "files/bolt_short.mp4"
cap = cv2.VideoCapture(video_path)

# Define video writer to save the output
output_path = "files/bolt_short_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Parameters
input_width, input_height = 416, 416  # YOLO input size
conf_threshold = 0.5
nms_threshold = 0.4

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Use YOLO to detect objects in the first frame or periodically
    if tracker.getObjects().empty():
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (input_width, input_height), [0, 0, 0], swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        detections = net.forward(output_layers)

        bboxes = []
        confidences = []
        class_ids = []

        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x, center_y, w, h = (
                        obj[0] * frame.shape[1],
                        obj[1] * frame.shape[0],
                        obj[2] * frame.shape[1],
                        obj[3] * frame.shape[0],
                    )
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    bboxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(bboxes, confidences, conf_threshold, nms_threshold)
        for i in indices:
            i = i[0]
            box = bboxes[i]
            tracker.add(cv2.TrackerKCF_create(), frame, tuple(box))

    # Update trackers
    success, objects = tracker.update(frame)
    for i, box in enumerate(objects):
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = classes[class_ids[i]] if i < len(class_ids) else "Object"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Write the annotated frame to the output video
    out.write(frame)

    # Optionally: Display the frame
    cv2.imshow("YOLO + OpenCV Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved at {output_path}")
