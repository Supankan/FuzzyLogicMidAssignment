from ultralytics import YOLO
import cv2
from sort import Sort  # Import the SORT tracker

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize SORT tracker
tracker = Sort()

# Open the video file
video_path = "path/to/your/video.mp4"
cap = cv2.VideoCapture(video_path)

# Define video writer to save the output
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Get bounding boxes and class labels from YOLO results
    detections = []
    for r in results[0].boxes:  # Extract bounding boxes
        bbox = r.xyxy[0].cpu().numpy()  # Convert to numpy array
        conf = r.conf.cpu().numpy()  # Confidence score
        class_id = r.cls.cpu().numpy()  # Class label (optional, if needed)

        # Add detection only if confidence is high
        if conf > 0.5:  # You can set your own threshold here
            detections.append([bbox[0], bbox[1], bbox[2], bbox[3], conf])  # [x1, y1, x2, y2, conf]

    # Convert to numpy array for SORT
    detections = np.array(detections)

    # Update tracker with the latest detections
    tracked_objects = tracker.update(detections)

    # Loop through tracked objects
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj.astype(int)  # Get bounding box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {int(obj_id)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Write the annotated frame to the output video
    out.write(frame)

    # Optionally: Display the frame
    cv2.imshow("YOLOv8 + SORT Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved at {output_path}")

