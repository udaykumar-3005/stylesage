from ultralytics import YOLO
from PIL import Image

MODEL_PATH = "artifacts/weights/yolov8n.pt"

# We still filter classes so we don't detect cars or traffic lights.
VALID_CLASSES = ['person', 'tie', 'handbag', 'suitcase', 'backpack']

def load_yolo_model():
    print(f"👁️  Loading YOLO model from {MODEL_PATH}...")
    return YOLO(MODEL_PATH)

def detect_objects(model, image_path):
    """
    Scans image and returns ALL valid fashion objects found.
    Sorted by Confidence (Highest confidence first).
    """
    print(f"🔎 Scanning image: {image_path}")
    results = model(image_path, conf=0.5) 
    first_result = results[0]
    
    found_objects = []

    for box in first_result.boxes:
        class_id = int(box.cls[0])
        class_name = first_result.names[class_id]
        confidence = float(box.conf[0])
        
        # Filter: Only keep relevant items
        if class_name in VALID_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            found_objects.append({
                "label": class_name,
                "score": round(confidence, 2),
                "box": (x1, y1, x2, y2)
            })

    # Sort purely by Confidence (The AI's "sureness")
    found_objects.sort(key=lambda x: x['score'], reverse=True)
    
    return found_objects

def crop_and_save(image_path, box, output_path="data/processed/crop_temp.jpg", padding_pct=0.10):
    """
    Crops the object with padding so it doesn't look cramped.
    """
    print(f"✂️  Cropping with {int(padding_pct*100)}% padding...")
    
    with Image.open(image_path) as img:
        img_width, img_height = img.size
        x1, y1, x2, y2 = box

        # 1. Calculate Padding
        pad_x = int((x2 - x1) * padding_pct)
        pad_y = int((y2 - y1) * padding_pct)

        # 2. Apply Padding & Clamp (Ensure we don't go off-screen)
        new_x1 = max(0, x1 - pad_x)
        new_y1 = max(0, y1 - pad_y)
        new_x2 = min(img_width, x2 + pad_x)
        new_y2 = min(img_height, y2 + pad_y)

        # 3. Crop & Save
        cropped_img = img.crop((new_x1, new_y1, new_x2, new_y2))
        cropped_img.save(output_path)
        print(f"✅ Saved padded crop to: {output_path}")
        return output_path