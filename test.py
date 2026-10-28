import cv2
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# keremberke/yolov8x-sku110k is gone from Hugging Face (401/404).
# Ultralytics then saved the error page as a .pt file, which caused:
#   _pickle.UnpicklingError: pickle data was truncated
#
# Use a public SKU-110K YOLO checkpoint instead (YOLOv8l, closest available).
HF_REPO = "sbfisher/yolov8l-sku110k"
HF_FILE = "best_imgsz640.pt"
MIN_WEIGHT_BYTES = 10 * 1024 * 1024  # real YOLO weights are tens of MB

WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)

# Remove the leftover 401 error body if it is still sitting in weights/
for leftover in WEIGHTS_DIR.glob("*.pt"):
    if leftover.stat().st_size < MIN_WEIGHT_BYTES:
        leftover.unlink()

print(f"Downloading {HF_REPO}/{HF_FILE} (first run only)...")
weights_path = Path(
    hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, local_dir=WEIGHTS_DIR)
)
if weights_path.stat().st_size < MIN_WEIGHT_BYTES:
    raise RuntimeError(
        f"Downloaded weights look truncated ({weights_path.stat().st_size} bytes): {weights_path}"
    )

model = YOLO(str(weights_path))

# 2. Path to your shelf image
IMAGE_PATH = "dataset/images/1.jpg"  # Replace with your image file path
OUTPUT_DIR = "test/cropped_products"
ANNOTATED_PATH = "test/shelf_with_boxes.jpg"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3. Run inference
# iou=0.5 helps avoid dropping adjacent shelf items
results = model.predict(source=IMAGE_PATH, conf=0.25, iou=0.5, save=False)

# 4. Crop every detected product for Step 2 (Vector Search)
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

result = results[0]
boxes = result.boxes.xyxy.cpu().numpy()
confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else [None] * len(boxes)

print(f"Total products detected on shelf: {len(boxes)}")

# Full shelf with numbered boxes so you can match product_N.jpg and see misses
annotated = image.copy()
h, w = annotated.shape[:2]
thickness = max(2, min(w, h) // 400)
font_scale = max(0.5, min(w, h) / 1400)
font = cv2.FONT_HERSHEY_SIMPLEX
box_color = (0, 220, 0)
text_bg = (0, 140, 0)

for idx, (box, conf) in enumerate(zip(boxes, confs), start=1):
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = image[y1:y2, x1:x2]
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"product_{idx}.jpg"), crop)

    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)
    label = f"{idx}" if conf is None else f"{idx} {conf:.2f}"
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    ty = y1 - 6 if y1 - th - 10 > 0 else y1 + th + 8
    cv2.rectangle(
        annotated,
        (x1, ty - th - 4),
        (x1 + tw + 6, ty + baseline),
        text_bg,
        -1,
    )
    cv2.putText(annotated, label, (x1 + 3, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

os.makedirs(os.path.dirname(ANNOTATED_PATH) or ".", exist_ok=True)
cv2.imwrite(ANNOTATED_PATH, annotated)

print(f"Saved {len(boxes)} cropped products to '{OUTPUT_DIR}/'")
print(f"Full shelf with boxes: '{ANNOTATED_PATH}'")
print("Numbers on the full image match product_N.jpg. Empty regions are misses.")
