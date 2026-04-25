import os 
import cv2
import random 
import torch 
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class SRPatchDataset(Dataset):
    """
    PyTorch Dataset for Super-Resolution.
    Dynamically extracts synchronous random patches from LR and HR image pairs.
    """

    def __init__(self, lr_dir: str, hr_dir, scale: int=4, lr_patch_size: int=32):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.scale = scale
        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = lr_patch_size * scale

        valid_exts = ('.png', '.jpg', '.jpeg')
        self.image_names = [f.name for f in self.lr_dir.glob("*") if f.suffix.lower() in valid_exts]

        if len(self.image_names) == 0:
            raise ValueError(f"No images found in {self.lr_dir}")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        
        # Load images
        lr_img = cv2.imread(str(self.lr_dir / img_name))
        hr_img = cv2.imread(str(self.hr_dir / img_name))
        
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        
        h, w = lr_img.shape[:2]
        
        # --- NEW: Dynamic Padding for tiny real-world crops ---
        pad_h = max(0, self.lr_patch_size - h)
        pad_w = max(0, self.lr_patch_size - w)
        
        if pad_h > 0 or pad_w > 0:
            # Pad the LR image using edge reflection
            lr_img = cv2.copyMakeBorder(lr_img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            # Pad the HR image by the exact same scaled amount
            hr_img = cv2.copyMakeBorder(hr_img, 0, pad_h * self.scale, 0, pad_w * self.scale, cv2.BORDER_REFLECT_101)
            # Update h and w to the new padded dimensions
            h, w = lr_img.shape[:2]
        # ------------------------------------------------------
        
        # 1. Random Patch Extraction
        lr_x = random.randint(0, w - self.lr_patch_size)
        lr_y = random.randint(0, h - self.lr_patch_size)
        
        hr_x = lr_x * self.scale
        hr_y = lr_y * self.scale
        
        lr_patch = lr_img[lr_y : lr_y + self.lr_patch_size, lr_x : lr_x + self.lr_patch_size]
        hr_patch = hr_img[hr_y : hr_y + self.hr_patch_size, hr_x : hr_x + self.hr_patch_size]
        
        # 2. Convert to PyTorch Tensors
        lr_tensor = torch.from_numpy(lr_patch.transpose(2, 0, 1)).float() / 255.0
        hr_tensor = torch.from_numpy(hr_patch.transpose(2, 0, 1)).float() / 255.0
        
        return lr_tensor, hr_tensor