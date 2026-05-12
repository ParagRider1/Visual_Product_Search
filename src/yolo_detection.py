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

# Run detection
results = model(str(img_path))

# Plot result
annotated_frame = results[0].plot()

# Convert BGR -> RGB
annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

# Show image
plt.figure(figsize=(8,8))
plt.imshow(annotated_frame)
plt.title("YOLO Detection Result")
plt.axis("off")

plt.show()