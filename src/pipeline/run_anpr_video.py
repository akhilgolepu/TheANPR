import cv2
import torch
import easyocr
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import logging
import warnings
import time

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

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
            
            # Phase 1: Detection
            results = self.detector(frame, verbose=False)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                
                # Bounding box padding
                pad = 5
                y1_pad, y2_pad = max(0, y1-pad), min(height, y2+pad)
                x1_pad, x2_pad = max(0, x1-pad), min(width, x2+pad)
                
                plate_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if plate_crop.size == 0:
                    continue
                    
                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
                
                # Phase 2: OCR
                ocr_results = self.reader.readtext(
                    gray, 
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    detail=1, 
                    paragraph=False 
                )
                
                final_text = ""
                if ocr_results:
                    valid_texts = []
                    for bbox_ocr, text, conf in ocr_results:
                        h_text = bbox_ocr[2][1] - bbox_ocr[0][1]
                        valid_texts.append({'text': text, 'height': h_text})
                    
                    if valid_texts:
                        valid_texts = sorted(valid_texts, key=lambda x: x['height'], reverse=True)
                        max_height = valid_texts[0]['height']
                        
                        plate_parts = [item['text'] for item in valid_texts if item['height'] > max_height * 0.60]
                        raw_text = "".join(plate_parts).replace(" ", "")
                        
                        # Phase 3: Post-Processing
                        final_text = correct_indian_plate(raw_text)
                
                # Visuals: Draw the bounding box and the text on the main frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if final_text:
                    # Draw a background rectangle for the text to make it readable
                    (text_w, text_h), _ = cv2.getTextSize(final_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)
                    cv2.putText(frame, final_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Calculate and draw FPS
            processing_time = time.time() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0.0
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
        cap.release()
        writer.release()
        # cv2.destroyAllWindows()
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