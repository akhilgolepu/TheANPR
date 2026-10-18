"""
Active Learning Pipeline: Extract Corrections for Retraining

This script exports all human-corrected plate detections from the database,
generates YOLO-compatible annotation files, and prepares them for retraining.

Usage:
    python src/active_learning/export_corrections.py
    
Output:
    - data/active_learning/images/ (copies of images with corrections)
    - data/active_learning/labels/ (YOLO .txt annotation files)
    - data/active_learning/corrections_manifest.json (metadata)
"""

import json
import logging
from pathlib import Path
from datetime import datetime
import shutil
from typing import List, Dict

import numpy as np
import cv2
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import database models
import sys
sys.path.append('src/api')
from database import DetectionRecord

# Configuration
DB_PATH = "data/db/anpr.db"
OUTPUT_DIR = Path("data/active_learning")
IMAGES_DIR = OUTPUT_DIR / "images"
LABELS_DIR = OUTPUT_DIR / "labels"
MANIFEST_FILE = OUTPUT_DIR / "corrections_manifest.json"

# Indian license plate format: SS DD LL NNNN
# Bounding box format for YOLO: <class_id> <x_center> <y_center> <width> <height>
# For plates, we use class_id = 0 (license plate)
PLATE_CLASS_ID = 0


class ActiveLearningExporter:
    """Exports corrected detections for active learning retraining."""
    
    def __init__(self, db_path: str):
        """Initialize database connection."""
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Session = sessionmaker(bind=engine)
        self.session = Session()
        logger.info(f"Connected to database: {db_path}")
    
    def prepare_directories(self):
        """Create output directories if they don't exist."""
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        LABELS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Prepared output directories: {IMAGES_DIR}, {LABELS_DIR}")
    
    def get_corrected_detections(self) -> List[DetectionRecord]:
        """
        Fetch all detections where human has provided a correction.
        
        Returns:
            List of DetectionRecord objects with corrected_plate != null
        """
        query = self.session.query(DetectionRecord).filter(
            DetectionRecord.corrected_plate != None,
            DetectionRecord.corrected_plate != ""
        )
        
        records = query.all()
        logger.info(f"Found {len(records)} corrected detections in database")
        return records
    
    def bbox_to_yolo_format(self, bbox: List[int], image_width: int, image_height: int) -> str:
        """
        Convert bounding box from [x1, y1, x2, y2] to YOLO format.
        
        YOLO format: <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
        
        Args:
            bbox: [x1, y1, x2, y2] in pixel coordinates
            image_width: Width of the image
            image_height: Height of the image
        
        Returns:
            YOLO format string
        """
        x1, y1, x2, y2 = bbox
        
        # Calculate center and dimensions
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        
        # Normalize to 0-1 range
        x_center_norm = x_center / image_width
        y_center_norm = y_center / image_height
        width_norm = width / image_width
        height_norm = height / image_height
        
        return f"{PLATE_CLASS_ID} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
    
    def export_correction(self, record: DetectionRecord, index: int) -> bool:
        """
        Export a single corrected detection as image + YOLO label.
        
        Args:
            record: DetectionRecord with correction
            index: Index for naming output files
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if image exists
            image_path = Path(record.image_path)
            if not image_path.exists():
                logger.warning(f"Image not found: {record.image_path}")
                return False
            
            # Read and copy image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Could not read image: {record.image_path}")
                return False
            
            height, width = image.shape[:2]
            
            # Generate output filename
            output_base = f"correction_{index:05d}"
            output_image = IMAGES_DIR / f"{output_base}.jpg"
            output_label = LABELS_DIR / f"{output_base}.txt"
            
            # Copy image
            shutil.copy2(str(image_path), str(output_image))
            
            # Parse bbox (stored as "[x1, y1, x2, y2]" or similar)
            try:
                # The bbox is likely stored as a list or JSON string
                bbox = record.bbox if isinstance(record.bbox, (list, tuple)) else json.loads(str(record.bbox))
                bbox = [int(x) for x in bbox]
            except Exception as e:
                logger.warning(f"Could not parse bbox for record {record.id}: {e}")
                return False
            
            # Convert to YOLO format
            yolo_line = self.bbox_to_yolo_format(bbox, width, height)
            
            # Write label file
            with open(output_label, 'w') as f:
                f.write(yolo_line + "\n")
            
            logger.debug(
                f"Exported correction {index}: "
                f"{output_image.name} <- {record.corrected_plate} "
                f"(predicted: {record.predicted_plate})"
            )
            return True
        
        except Exception as e:
            logger.error(f"Error exporting correction {index}: {e}")
            return False
    
    def generate_manifest(self, records: List[DetectionRecord], exported_count: int) -> Dict:
        """
        Generate metadata manifest for the exported corrections.
        
        Args:
            records: Original DetectionRecords
            exported_count: Number successfully exported
        
        Returns:
            Manifest dictionary
        """
        manifest = {
            "timestamp": datetime.now().isoformat(),
            "total_corrections": len(records),
            "successfully_exported": exported_count,
            "output_directory": str(OUTPUT_DIR),
            "images_directory": str(IMAGES_DIR),
            "labels_directory": str(LABELS_DIR),
            "sample_records": [
                {
                    "record_id": r.id,
                    "predicted_plate": r.predicted_plate,
                    "corrected_plate": r.corrected_plate,
                    "confidence_score": float(r.confidence_score) if r.confidence_score else 0.0,
                    "correction_timestamp": r.created_at.isoformat() if r.created_at else None
                }
                for r in records[:5]  # First 5 records as samples
            ]
        }
        
        return manifest
    
    def run(self):
        """Execute the full active learning export pipeline."""
        logger.info("=" * 70)
        logger.info("Starting Active Learning Export Pipeline")
        logger.info("=" * 70)
        
        try:
            # Prepare directories
            self.prepare_directories()
            
            # Fetch corrected detections
            records = self.get_corrected_detections()
            
            if len(records) == 0:
                logger.info("No corrected detections found. Nothing to export.")
                return
            
            # Export each correction
            exported_count = 0
            for i, record in enumerate(records, 1):
                if self.export_correction(record, i):
                    exported_count += 1
                
                if i % 50 == 0:
                    logger.info(f"Progress: {i}/{len(records)} corrections processed")
            
            # Generate and save manifest
            manifest = self.generate_manifest(records, exported_count)
            
            with open(MANIFEST_FILE, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info("=" * 70)
            logger.info("Active Learning Export Complete")
            logger.info(f"✓ Exported {exported_count}/{len(records)} corrections")
            logger.info(f"✓ Images saved to: {IMAGES_DIR}")
            logger.info(f"✓ Labels saved to: {LABELS_DIR}")
            logger.info(f"✓ Manifest saved to: {MANIFEST_FILE}")
            logger.info("=" * 70)
            logger.info("\nNext steps:")
            logger.info("1. Review the exported corrections in data/active_learning/")
            logger.info("2. Run: python src/active_learning/retrain_detector.py")
            logger.info("3. Deploy the new model: python src/active_learning/deploy_model.py")
            
        finally:
            self.session.close()


def main():
    """Main entry point."""
    exporter = ActiveLearningExporter(DB_PATH)
    exporter.run()


if __name__ == "__main__":
    main()
