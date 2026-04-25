import cv2
import numpy as np
import yaml
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class SRDatasetGenerator:
    """
    Handles the mathematical degradation of High-Resolution (HR) images 
    to create Low-Resolution (LR) pairs for Super-Resolution training.
    """
    def __init__(self, config_path: str = "configs/sr_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.hr_path = Path(self.config['paths']['hr_data'])
        self.lr_path = Path(self.config['paths']['lr_data'])
        self.scale = self.config['params']['scale_factor']
        
        self.lr_path.mkdir(parents=True, exist_ok=True)

    def _apply_degradation(self, img: np.ndarray) -> np.ndarray:
        """
        Applies a degradation pipeline: Blur -> Downsample -> Noise.
        This simulates the limitations of real-world CCTV/Traffic cameras.
        """
        # 1. Simulate Lens Blur (Gaussian)
        k_size = self.config['params']['blur_kernel']
        img_blur = cv2.GaussianBlur(img, (k_size, k_size), 0)
        
        # 2. Simulate Distance (Bicubic Downsampling)
        h, w = img.shape[:2]
        img_lr = cv2.resize(img_blur, (w // self.scale, h // self.scale), 
                            interpolation=cv2.INTER_CUBIC)
        
        # 3. Simulate Sensor Grain (Gaussian Noise)
        sigma = self.config['params']['noise_sigma']
        noise = np.random.normal(0, sigma, img_lr.shape).astype(np.float32)
        img_lr = np.clip(img_lr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        return img_lr

    def generate(self):
        """Orchestrates the batch degradation process."""
        valid_extensions = ('.png', '.jpg', '.jpeg')
        image_paths = [p for p in self.hr_path.glob("*") if p.suffix.lower() in valid_extensions]
        
        if not image_paths:
            logger.error(f"❌ No images found in {self.hr_path}")
            return

        logger.info(f"🛠️  Generating {len(image_paths)} LR samples (Scale: {self.scale}x)...")
        
        for img_p in tqdm(image_paths, desc="Processing Images"):
            img_hr = cv2.imread(str(img_p))
            if img_hr is None:
                continue
                
            img_lr = self._apply_degradation(img_hr)
            
            # Use the same filename to maintain the mapping for training
            cv2.imwrite(str(self.lr_path / img_p.name), img_lr)

        logger.info(f"✅ Dataset generation complete. Files saved to: {self.lr_path}")

if __name__ == "__main__":
    generator = SRDatasetGenerator()
    generator.generate()