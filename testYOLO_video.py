from ultralytics import YOLO
import cv2

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # You can use any pre-trained YOLOv8 model

# Open the video file
video_path = "C:/Users/User/PycharmProjects/Random/RandomML/FuzzyLogicProject/files/input_files/ronaldo2.mp4"
cap = cv2.VideoCapture(video_path)

# Define video writer to save the output
output_path = "C:/Users/User/PycharmProjects/Random/RandomML/FuzzyLogicProject/files/output_files/output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for saving mp4 files
fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second from the input video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get the width of the frame
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get the height of the frame
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        break

    # Perform object detection on the frame
    results = model(frame)

    # Get the annotated frame (with bounding boxes)
    annotated_frame = results[0].plot()  # Plot the results on the frame

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Optionally: Display the frame (remove this if running on headless environments)
    cv2.imshow("YOLOv8 Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved at {output_path}")
