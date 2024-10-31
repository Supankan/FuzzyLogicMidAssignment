from ultralytics import YOLO

# Load the pre-trained YOLOv8 model (no need to train)
model = YOLO("yolov8n.pt")  # 'yolov8n.pt' is a small pre-trained YOLOv8 model

# Perform object detection on an image
results = model("files/chevrolet_camaro_1970.jpg")  # Path to your input image

# Display the results
results[0].show()

# Optionally, save the results to an image file
results[0].save("files/output.jpg")
