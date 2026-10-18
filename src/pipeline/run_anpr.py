import cv2
import torch
import easyocr
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import logging
import warnings
import re
import difflib

# Suppress some of the noisy EasyOCR warnings
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Includes active and commonly-seen legacy/alternate Indian registration codes.
VALID_INDIAN_STATES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK",
    "UP", "WB"
}

# Try to import your custom formatter, or define a fallback
try:
    import sys
    sys.path.append('src/ocr')
    from post_process import correct_indian_plate
except ImportError:
    logger.warning("Could not load custom post-processing. Using raw OCR output.")
    def correct_indian_plate(text): return text

class ANPRPipeline:
    def __init__(self, yolo_weights: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"🚀 Initializing Production ANPR Pipeline on {self.device.upper()}...")
        
        # 1. Load Custom YOLO Detector
        logger.info("📦 Loading YOLOv8 Plate Detector...")
        self.detector = YOLO(yolo_weights)
        
        # 2. Load EasyOCR (English)
        logger.info("📦 Loading EasyOCR Engine...")
        self.reader = easyocr.Reader(['en'], gpu=(self.device == 'cuda'))
        
    def _calculate_format_score(self, text: str) -> float:
        """Scores how well the text matches standard Indian HSRP rules."""
        if not text: 
            return 0.0

        # Penalize invalid state prefixes early to force manual review.
        if len(text) >= 2:
            state_code = text[:2]
            if state_code.isalpha() and state_code not in VALID_INDIAN_STATES:
                return 0.0
            
        # Perfect match: e.g., MH20DV2366
        pattern = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$')
        if pattern.match(text):
            return 1.0
            
        # Close match
        if 9 <= len(text) <= 11 and text.isalnum():
            return 0.5 
            
        return 0.0
        
    def process_image(self, image_path: str):
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"❌ Could not load image at {image_path}")
            return
            
        logger.info(f"🔎 Processing: {Path(image_path).name}")
        
        # Phase 1: Detection with improved conf threshold
        results = self.detector(img, verbose=False, conf=0.45)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        
        if len(boxes) == 0:
            logger.warning("⚠️ No license plates detected in this image.")
            return
        
        height, width = img.shape[:2]
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Dynamic padding based on box size
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)
            pad_x = max(2, int(box_w * 0.05))
            pad_y = max(2, int(box_h * 0.05))
            y1_p, y2_p = max(0, y1 - pad_y), min(height, y2 + pad_y)
            x1_p, x2_p = max(0, x1 - pad_x), min(width, x2 + pad_x)
            
            plate_crop = img[y1_p:y2_p, x1_p:x2_p]
            if plate_crop.size == 0:
                continue
            
            # Basic Image Preprocessing for OCR
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
            
            # Phase 2: OCR with improved allowlist (includes dash and dot)
            ocr_results = self.reader.readtext(
                gray, 
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.',
                detail=1, 
                paragraph=False 
            )
            
            final_text = ""
            avg_confidence = 0.0
            format_score = 0.0
            
            if ocr_results:
                # Parse all OCR blocks with spatial coordinates
                parsed_blocks = []
                for bbox, text, conf in ocr_results:
                    if not text or conf <= 0:
                        continue
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    parsed_blocks.append({
                        "text": text,
                        "conf": float(conf),
                        "height": max(1.0, float(y_max - y_min)),
                        "x_center": float((x_min + x_max) / 2.0),
                        "y_center": float((y_min + y_max) / 2.0),
                    })

                if parsed_blocks:
                    # Spatial OCR ordering: group by Y, sort by X
                    heights = sorted([b["height"] for b in parsed_blocks])
                    median_h = heights[len(heights) // 2]
                    valid_blocks = [b for b in parsed_blocks if b["height"] >= median_h * 0.5]

                    if valid_blocks:
                        row_tol = max(5.0, median_h * 0.6)
                        by_y = sorted(valid_blocks, key=lambda b: b["y_center"])
                        rows = []

                        for block in by_y:
                            if not rows:
                                rows.append([block])
                                continue
                            last_row = rows[-1]
                            row_y = sum(j["y_center"] for j in last_row) / len(last_row)
                            if abs(block["y_center"] - row_y) <= row_tol:
                                last_row.append(block)
                            else:
                                rows.append([block])

                        plate_parts = []
                        ordered_blocks = []
                        for row in rows:
                            for block in sorted(row, key=lambda b: b["x_center"]):
                                plate_parts.append(block["text"])
                                ordered_blocks.append(block)

                        raw_text = "".join(plate_parts).replace(" ", "")
                        # Strip non-alphanumeric (removes dashes/dots)
                        clean_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())

                        # Phase 3: Fuzzy state correction
                        if len(clean_text) >= 4:
                            state_code = clean_text[:2]
                            if state_code.isalpha() and state_code not in VALID_INDIAN_STATES:
                                match = difflib.get_close_matches(
                                    state_code,
                                    sorted(VALID_INDIAN_STATES),
                                    n=1,
                                    cutoff=0.5,
                                )
                                if match:
                                    logger.info(
                                        "Auto-corrected state code from %s to %s",
                                        state_code,
                                        match[0],
                                    )
                                    clean_text = match[0] + clean_text[2:]

                        avg_confidence = sum(b["conf"] for b in ordered_blocks) / len(ordered_blocks)
                        final_text = correct_indian_plate(clean_text)
                        format_score = self._calculate_format_score(final_text)
            
            format_emoji = "✅" if format_score == 1.0 else "⚠️" if format_score > 0 else "❌"
            print(f"  [{i+1}] {format_emoji} OCR: \033[92m{final_text}\033[0m (conf: {avg_confidence:.3f}, format: {format_score:.1f})")

if __name__ == "__main__":
    # Point this to the absolute best YOLO model you trained in Phase 2
    # It is likely in your 'runs/detect/train/weights/best.pt' folder
    YOLO_MODEL_PATH = "models/checkpoints/plate_detector_v1.pt" 
    
    pipeline = ANPRPipeline(YOLO_MODEL_PATH)
    
    # Let's test it on a raw, uncropped photo of a car from your dataset
    # Replace this with an actual path to a full car image!
    TEST_IMAGES = [
        "data/raw/dataset_archive_2/train/1.jpg",
        "data/raw/dataset_archive_2/train/155.jpg",
        "data/raw/dataset_archive_2/train/145.jpg",
        "data/raw/dataset_archive_2/train/83.jpg",
        "data/raw/dataset_archive_2/train/28.jpg",
        "data/raw/indian_plates/number_plate_images_ocr/number_plate_images_ocr/dc_license_plates_VYA8KOAVKLW6PJYK.jpg",
    ]

    for img_path in TEST_IMAGES:
        if Path(img_path).exists():
            pipeline.process_image(img_path)
        else:
            logger.error(f"Please update TEST_IMAGE path. Could not find: {img_path}")
