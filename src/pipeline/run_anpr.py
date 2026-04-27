import cv2
import torch
import easyocr
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import logging
import warnings

# Suppress some of the noisy EasyOCR warnings
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

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
        
    def process_image(self, image_path: str):
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"❌ Could not load image at {image_path}")
            return
            
        logger.info(f"🔎 Processing: {Path(image_path).name}")
        
        # Phase 1: Detection
        results = self.detector(img, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        
        if len(boxes) == 0:
            logger.warning("⚠️ No license plates detected in this image.")
            return
            
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Add a slight padding to the bounding box
            pad = 5
            h, w = img.shape[:2]
            y1, y2 = max(0, y1-pad), min(h, y2+pad)
            x1, x2 = max(0, x1-pad), min(w, x2+pad)
            
            plate_crop = img[y1:y2, x1:x2]
            
            # Basic Image Preprocessing for OCR
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            # Increase contrast slightly
            gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
            
            # Phase 2: OCR (Notice detail=1 and paragraph=False)
            ocr_results = self.reader.readtext(
                gray, 
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                detail=1, 
                paragraph=False 
            )
            
            if ocr_results:
                # Filter and sort by text height
                valid_texts = []
                for bbox, text, conf in ocr_results:
                    # Bbox format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    # Calculate the height of this specific piece of text
                    height = bbox[2][1] - bbox[0][1]
                    valid_texts.append({'text': text, 'height': height, 'conf': conf})
                
                # Sort the detected text blocks by their physical height on the plate
                valid_texts = sorted(valid_texts, key=lambda x: x['height'], reverse=True)
                
                # The actual license plate number is almost always the largest text
                # If it's a stacked plate, it might be the top 2 largest
                # Let's take the largest pieces of text and combine them
                plate_parts = []
                max_height = valid_texts[0]['height']
                
                for item in valid_texts:
                    # If this text is at least 60% as tall as the biggest text, keep it
                    if item['height'] > max_height * 0.60:
                        plate_parts.append(item['text'])
                
                # Join the valid parts (handles 2-line stacked plates nicely)
                raw_text = "".join(plate_parts).replace(" ", "")
                
                # Phase 3: Post-Processing
                final_text = correct_indian_plate(raw_text)
                
                print(f"  [{i+1}] Raw OCR: {raw_text} --> Final: \033[92m{final_text}\033[0m")
            else:
                print(f"  [{i+1}] ⚠️ Plate detected, but OCR failed to read text.")

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
