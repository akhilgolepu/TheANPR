import cv2
import torch
import easyocr
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import logging
import warnings
import time
import re
import difflib

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

try:
    import sys
    sys.path.append('src/ocr')
    from post_process import correct_indian_plate
except ImportError:
    logger.warning("Could not load custom post-processing. Using raw OCR output.")
    def correct_indian_plate(text): return text

class ANPRVideoPipeline:
    def __init__(self, yolo_weights: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"🚀 Initializing Video ANPR Pipeline on {self.device.upper()}...")
        
        # Load Models
        logger.info("📦 Loading YOLOv8 Plate Detector...")
        self.detector = YOLO(yolo_weights)
        
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
        
    def process_video(self, input_path: str, output_path: str = "runs/output_video.mp4"):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"❌ Could not open video file: {input_path}")
            return
            
        # Get video properties for the writer
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_target = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps_target, (width, height))
        
        logger.info(f"🎥 Starting Video Processing: {total_frames} frames to process.")
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            start_time = time.time()
            
            # Phase 1: Detection with improved conf threshold
            results = self.detector(frame, verbose=False, conf=0.45)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                
                # Dynamic padding based on box size
                box_w = max(1, x2 - x1)
                box_h = max(1, y2 - y1)
                pad_x = max(2, int(box_w * 0.05))
                pad_y = max(2, int(box_h * 0.05))
                y1_pad, y2_pad = max(0, y1 - pad_y), min(height, y2 + pad_y)
                x1_pad, x2_pad = max(0, x1 - pad_x), min(width, x2 + pad_x)
                
                plate_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if plate_crop.size == 0:
                    continue
                    
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
                    for bbox_ocr, text, conf in ocr_results:
                        if not text or conf <= 0:
                            continue
                        xs = [p[0] for p in bbox_ocr]
                        ys = [p[1] for p in bbox_ocr]
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
                
                # Visuals: Draw the bounding box and the text on the main frame
                color = (0, 255, 0) if format_score == 1.0 else (255, 165, 0) if format_score > 0 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                if final_text:
                    # Draw a background rectangle for the text to make it readable
                    (text_w, text_h), _ = cv2.getTextSize(final_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                    cv2.putText(frame, final_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Calculate and draw FPS
            processing_time = time.time() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0.0
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Write the frame
            writer.write(frame)
                
        cap.release()
        writer.release()
        logger.info(f"✅ Video processing complete. Saved to {output_path}")

if __name__ == "__main__":
    YOLO_MODEL_PATH = "models/checkpoints/plate_detector_v1.pt"
    
    # Grab a .mp4 of Indian traffic (from YouTube, phone, or dashcam dataset)
    # Put it in your raw directory and update this path:
    INPUT_VIDEO = "data/processed/2103099-uhd_3840_2160_30fps.mp4" 
    
    pipeline = ANPRVideoPipeline(YOLO_MODEL_PATH)
    
    if Path(INPUT_VIDEO).exists():
        pipeline.process_video(INPUT_VIDEO)
    else:
        logger.error(f"Please place a video file at {INPUT_VIDEO}")