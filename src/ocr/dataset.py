import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class OCRDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str):
        # The crucial fix: drop any empty/phantom rows immediately
        self.df = pd.read_csv(csv_path).dropna(subset=['filename', 'label'])
        self.img_dir = Path(img_dir)
        
        # Standard CRNN input dimensions
        self.target_h = 32
        self.target_w = 128

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row['filename']
        label = str(row['label']).strip().upper()
        
        # 1. Load Grayscale
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Missing image: {img_path}")

        h, w = img.shape
        
        # 2. Aspect-Ratio Stitching for Stacked Plates
        if w / h < 2.5:
            mid = h // 2
            top_half = img[0:mid, :]
            bottom_half = img[mid:h, :]
            img = np.concatenate((top_half, bottom_half), axis=1)
            h, w = img.shape

        # 3. Resize keeping aspect ratio
        aspect = w / h
        new_w = min(int(self.target_h * aspect), self.target_w)
        img_resized = cv2.resize(img, (new_w, self.target_h))
        
        # 4. Pad to fixed width (128) with white pixels (255)
        img_padded = np.ones((self.target_h, self.target_w), dtype=np.uint8) * 255
        img_padded[:, :new_w] = img_resized
        
        # 5. Convert to PyTorch Tensor (1, H, W)
        tensor = torch.from_numpy(img_padded).float() / 255.0
        tensor = tensor.unsqueeze(0) 
        
        return tensor, label

if __name__ == "__main__":
    dataset = OCRDataset(
        csv_path="data/processed/ocr_dataset/synthetic_labels.csv", 
        img_dir="data/processed/ocr_dataset/synthetic_images"
    )
    img_tensor, txt_label = dataset[0]
    print(f"✅ Dataloader initialized. Tensor Shape: {img_tensor.shape}")