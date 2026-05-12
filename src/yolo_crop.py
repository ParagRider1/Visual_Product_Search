from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Load YOLO model
model = YOLO("yolov8n.pt")

# Dataset root
DATASET_ROOT = Path("data/DeepFashion")

# Sample image
img_path = DATASET_ROOT / "Img" / "img" / "WOMEN" / "Dresses" / "id_00000002" / "02_1_front.jpg"

# Read image
image = cv2.imread(str(img_path))

# Run YOLO detection
results = model(str(img_path))

# Extract first detected box
boxes = results[0].boxes

if len(boxes) == 0:
    print("No detections found")
    exit()

# Get coordinates
x1, y1, x2, y2 = boxes.xyxy[0].cpu().numpy().astype(int)

# Crop image
cropped = image[y1:y2, x1:x2]

# Convert BGR -> RGB
cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

# Show cropped image
plt.figure(figsize=(5, 7))
plt.imshow(cropped)
plt.title("YOLO Cropped Product")
plt.axis("off")

plt.show()