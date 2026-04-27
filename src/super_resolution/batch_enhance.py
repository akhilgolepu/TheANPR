import os
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Import your existing EDSR model class
from model_edsr import EDSR

def batch_enhance(input_dir: str, output_dir: str, model_path: str, csv_path: str):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Batch Enhancement on {device}...")

    # Load the fine-tuned ESRGAN Generator
    model = EDSR(scale_factor=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    valid_exts = ('.png', '.jpg', '.jpeg')
    images = [p for p in in_path.glob("*") if p.suffix.lower() in valid_exts]
    
    if not images:
        print(f"❌ No images found in {in_path}")
        return

    csv_data = []

    print(f"🔍 Processing {len(images)} plates for OCR dataset...")
    with torch.no_grad():
        for img_file in tqdm(images):
            img = cv2.imread(str(img_file))
            if img is None:
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device)
            
            output = model(tensor)
            
            out_img = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            out_img = (out_img * 255.0).clip(0, 255).astype(np.uint8)
            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            
            # Save the enhanced image to the OCR dataset folder
            save_dest = out_path / img_file.name
            cv2.imwrite(str(save_dest), out_img)
            
            # Add to CSV tracking
            csv_data.append({"filename": img_file.name, "label": ""})

    # Generate the blank CSV for manual labeling
    df = pd.DataFrame(csv_data)
    # If a CSV already exists, we don't want to overwrite your hard work!
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
        print(f"📝 Created blank labels file at {csv_path}")
    else:
        print(f"⚠️ {csv_path} already exists. Skipping CSV creation to protect data.")

    print(f"✅ Batch enhancement complete! {len(csv_data)} plates ready for labeling.")

if __name__ == "__main__":
    # Point this to wherever your raw YOLO plate crops are stored
    RAW_PLATES_DIR = "data/raw/hr_plates" 
    
    # OCR Dataset destinations
    OCR_IMAGES_DIR = "data/processed/ocr_dataset/images"
    OCR_CSV_PATH = "data/processed/ocr_dataset/labels.csv"
    
    # The GAN we just trained
    GAN_WEIGHTS = "models/checkpoints/finetuned_esrgan/esrgan_epoch_10.pt"
    
    batch_enhance(
        input_dir=RAW_PLATES_DIR,
        output_dir=OCR_IMAGES_DIR,
        model_path=GAN_WEIGHTS,
        csv_path=OCR_CSV_PATH
    )