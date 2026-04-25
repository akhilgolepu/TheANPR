import cv2
import torch
import numpy as np
import sys
from pathlib import Path
from model_edsr import EDSR

def enhance_plate(image_path: str, model_path: str = "models/checkpoints/sr_models/edsr_epoch_50.pt"):
    img_file = Path(image_path)
    if not img_file.exists():
        print(f"❌ ERROR: Could not find image at '{img_file.resolve()}'")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = EDSR(scale_factor=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load and prep image
    img = cv2.imread(str(img_file))
    if img is None:
        print(f"❌ ERROR: File exists but OpenCV cannot read it (might be corrupted): {image_path}")
        sys.exit(1)
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor
    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)
    
    print(f"🔍 Running EDSR enhancement on {image_path}...")
    with torch.no_grad():
        output = model(tensor)
        
    # Convert back to image
    out_img = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    out_img = (out_img * 255.0).clip(0, 255).astype(np.uint8)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    
    # Save output
    out_path = f"enhanced_{img_file.name}"
    cv2.imwrite(out_path, out_img)
    print(f"✅ Enhanced plate saved to {out_path}")

if __name__ == "__main__":
    target_image = "img_020.png" 
    # Point to the new FINE-TUNED GAN weights
    finetuned_weights = "models/checkpoints/finetuned_esrgan/esrgan_epoch_10.pt"
    
    enhance_plate(target_image, model_path=finetuned_weights)
