from ultralytics import YOLO
from pathlib import Path

architecture = "n"
version = "010923_1"
model = YOLO(f"weights/yolov8{architecture}_{version}.pt")

# Test Data
source = Path(f"track/video.mp4")
project = f"validation"
vidName = f"{version}"

# results would be a generator which is more friendly to memory by setting stream=True
results = model.predict(
    source=source,
    show=True,
    save_txt=True,
    project=project,
    name=f"{vidName}",
    save=True,
    exist_ok=True,
    stream=True,
)

for result in results:
    continue
