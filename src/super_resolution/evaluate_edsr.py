import os
import cv2
import yaml
import torch
import numpy as np
import math
import logging
from pathlib import Path
from model_edsr import EDSR

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class SREvaluator:
    def __init__(self, config_path="configs/sr_config.yaml", epoch=50):
        project_root = Path(__file__).resolve().parent.parent.parent
        with open(project_root / config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scale = self.config['params']['scale_factor']
        
        # Load the trained weights
        self.model = EDSR(scale_factor=self.scale).to(self.device)
        weights_path = project_root / self.config['training']['save_dir'] / f"edsr_epoch_{epoch}.pt"
        
        if not weights_path.exists():
            logger.error(f"❌ Weights not found at {weights_path}")
            return
            
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        logger.info(f"✅ Loaded EDSR model from Epoch {epoch}")

        self.lr_dir = project_root / self.config['paths']['lr_data']
        self.hr_dir = project_root / self.config['paths']['hr_data']
        self.output_dir = project_root / "reports" / "sr_visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def calculate_psnr(self, img1, img2):
        """Calculates Peak Signal-to-Noise Ratio (PSNR)."""
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        pixel_max = 255.0
        return 20 * math.log10(pixel_max / math.sqrt(mse))

    def evaluate_sample(self, filename: str):
        """Runs inference on a single image and generates a comparison."""
        lr_path = self.lr_dir / filename
        hr_path = self.hr_dir / filename
        
        if not lr_path.exists() or not hr_path.exists():
            logger.error(f"❌ Missing LR or HR image for {filename}")
            return

        # Load images (OpenCV uses BGR)
        lr_img = cv2.imread(str(lr_path))
        hr_img = cv2.imread(str(hr_path))
        
        # Prepare LR image for PyTorch (BGR -> RGB -> Tensor -> CHW -> Normalize -> Add Batch Dim)
        lr_rgb = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        lr_tensor = torch.from_numpy(lr_rgb.transpose(2, 0, 1)).float() / 255.0
        lr_tensor = lr_tensor.unsqueeze(0).to(self.device)

        # Run Inference
        logger.info(f"🔍 Enhancing {filename}...")
        with torch.no_grad():
            sr_tensor = self.model(lr_tensor)

        # Convert back to NumPy (Remove Batch Dim -> CHW to HWC -> De-normalize -> RGB to BGR)
        sr_img = sr_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        sr_img = (sr_img * 255.0).clip(0, 255).astype(np.uint8)
        sr_img = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)

        # Calculate PSNR
        # Note: We compare the generated SR image directly to the Ground Truth HR image
        psnr_val = self.calculate_psnr(hr_img, sr_img)
        logger.info(f"📊 PSNR Score: {psnr_val:.2f} dB")

        # Create Visual Comparison (Stitch them side-by-side)
        # 1. Resize LR up to HR size just using basic interpolation so we can compare sizes
        lr_upscaled = cv2.resize(lr_img, (hr_img.shape[1], hr_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # 2. Add text labels
        cv2.putText(lr_upscaled, "Input (Blurry/Noisy)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(sr_img, f"EDSR Output ({psnr_val:.2f} dB)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(hr_img, "Ground Truth (Original)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # 3. Concatenate horizontally
        comparison = np.hstack((lr_upscaled, sr_img, hr_img))
        
        output_file = self.output_dir / f"compare_{filename}"
        cv2.imwrite(str(output_file), comparison)
        logger.info(f"📸 Visual comparison saved to {output_file}")

if __name__ == "__main__":
    evaluator = SREvaluator()
    
    # The first image in the LR directory will be evaluated and visualized
    sample_file = next(evaluator.lr_dir.iterdir()).name
    evaluator.evaluate_sample(sample_file)