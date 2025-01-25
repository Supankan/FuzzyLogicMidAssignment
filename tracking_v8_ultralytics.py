from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the YOLOv8 model
model = YOLO("yolo11n.pt")

# Open the video file
video_path = "files/walking.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Define video writer
output_path = "files/walking_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLOv8 tracking on the frame
    results = model.track(frame, persist=True)

    # Get the boxes and track IDs, ensuring that results are valid
    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Plot the tracks
    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))  # x, y center point
        if len(track) > 30:  # Retain 30 tracks for 30 frames
            track.pop(0)

        # Draw the tracking lines
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

    # Write the annotated frame to the output video
    out.write(annotated_frame)

# Release resources
cap.release()
out.release()

print(f"Output video saved at {output_path}")
