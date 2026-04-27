import torch
import cv2
import numpy as np
from pathlib import Path

# Import our custom modules
from model_crnn import CRNN
from vocab import decode_label
from post_process import correct_indian_plate

def predict_plate(img_path: str, model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the Model
    model = CRNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Preprocess the Image (Match the Dataloader exactly)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Could not load {img_path}")
        return
        
    h, w = img.shape
    
    # Aspect-Ratio Stitching for Stacked Plates
    if w / h < 2.5:
        mid = h // 2
        img = np.concatenate((img[0:mid, :], img[mid:h, :]), axis=1)
        h, w = img.shape
        
    target_h, target_w = 32, 128
    aspect = w / h
    new_w = min(int(target_h * aspect), target_w)
    
    img_resized = cv2.resize(img, (new_w, target_h))
    img_padded = np.ones((target_h, target_w), dtype=np.uint8) * 255
    img_padded[:, :new_w] = img_resized
    
    tensor = torch.from_numpy(img_padded).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)
    
    # 3. Inference
    with torch.no_grad():
        preds = model(tensor)
        
        # Greedy Decoding (Take the max probability at each time step)
        _, preds_index = preds.max(2)
        preds_index = preds_index.squeeze(0).cpu().numpy().tolist()
        
        # CTC Decoding (Remove consecutive duplicates and blanks)
        decoded_ints = []
        for i in range(len(preds_index)):
            if preds_index[i] != 0 and (not (i > 0 and preds_index[i - 1] == preds_index[i])):
                decoded_ints.append(preds_index[i])
                
        raw_text = decode_label(decoded_ints)
        
    # 4. Heuristic Post-Processing
    final_text = correct_indian_plate(raw_text)
    
    print(f"File: {Path(img_path).name}")
    print(f"  Raw OCR Output : {raw_text}")
    print(f"  Post-Processed : {final_text}")
    print("-" * 40)

if __name__ == "__main__":
    # Point to the fine-tuned model
    MODEL_WEIGHTS = "models/checkpoints/ocr_models/finetuned/crnn_finetuned_epoch_30.pt"
    
    # Use the same exact real test images from earlier
    TEST_IMAGES = [
        "data/raw/hr_plates/img_005.png",
        "data/raw/hr_plates/img_011.png",
        "data/raw/hr_plates/img_038.png",
        "data/raw/hr_plates/img_085.png",
        "data/raw/hr_plates/img_107.png",
        "data/raw/hr_plates/img_045.png",
    ]
    
    for img_path in TEST_IMAGES:
        if Path(img_path).exists():
            predict_plate(img_path, MODEL_WEIGHTS)
        else:
            print(f"⚠️ Could not find: {img_path}")