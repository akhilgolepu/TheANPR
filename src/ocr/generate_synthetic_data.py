import os
import random
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from tqdm import tqdm

STATES = ['AP', 'AR', 'AS', 'BR', 'CG', 'GA', 'GJ', 'HR', 'HP', 'JH', 'KA', 'KL', 
          'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OD', 'PB', 'RJ', 'SK', 'TN', 'TS', 
          'TR', 'UP', 'UK', 'WB', 'DL']

def generate_plate_string() -> tuple:
    """Returns the text split into top and bottom lines for 2-line plates."""
    state = random.choice(STATES)
    district = f"{random.randint(1, 99):02d}"
    letters = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=random.randint(1, 2)))
    numbers = f"{random.randint(1, 9999):04d}"
    
    # Return as standard string, top line, and bottom line
    full_string = f"{state} {district} {letters} {numbers}"
    top_line = f"{state} {district}"
    bottom_line = f"{letters} {numbers}"
    
    return full_string, top_line, bottom_line

def create_synthetic_plate(font_path: str, is_two_line: bool) -> tuple:
    full_text, top, bottom = generate_plate_string()
    
    # 1. Handle Dimensions & Rendering based on format
    if is_two_line:
        width, height = 128, 128 # Boxy format
        img = Image.new('L', (width, height), color=255)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, size=35)
        
        # Center Top Line
        bbox_t = draw.textbbox((0, 0), top, font=font)
        draw.text(((width - (bbox_t[2] - bbox_t[0])) / 2, 10), top, font=font, fill=0)
        
        # Center Bottom Line
        bbox_b = draw.textbbox((0, 0), bottom, font=font)
        draw.text(((width - (bbox_b[2] - bbox_b[0])) / 2, 60), bottom, font=font, fill=0)
    else:
        width, height = 256, 64 # Wide format
        img = Image.new('L', (width, height), color=255)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, size=40)
        
        bbox = draw.textbbox((0, 0), full_text, font=font)
        draw.text(((width - (bbox[2] - bbox[0])) / 2, 5), full_text, font=font, fill=0)
    
    img_cv = np.array(img)
    
    # 2. GAN Artifacts & Noise
    angle = random.uniform(-2, 2)
    M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
    img_cv = cv2.warpAffine(img_cv, M, (width, height), borderValue=255)
    
    noise = np.random.normal(0, 10, img_cv.shape).astype(np.float32)
    img_cv = np.clip(img_cv.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Draw faint checkerboard grid
    grid_spacing = 4
    for x in range(0, width, grid_spacing):
        cv2.line(img_cv, (x, 0), (x, height), 220, 1)
    for y in range(0, height, grid_spacing):
        cv2.line(img_cv, (0, y), (width, y), 220, 1)
        
    if random.random() > 0.5:
        img_cv = cv2.GaussianBlur(img_cv, (3, 3), 0)

    # For the label, we just return the full text without spaces, 
    # since CTC will predict a single continuous string.
    clean_label = full_text.replace(" ", "")
    return img_cv, clean_label

def generate_dataset(num_samples: int = 10000):
    output_dir = Path("data/processed/ocr_dataset/synthetic_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    font_path = "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf" 
    
    csv_data = []
    
    print(f"🏭 Generating {num_samples} V2 synthetic plates (Mixed formats)...")
    for i in tqdm(range(num_samples)):
        # 30% chance to generate a two-line plate
        is_two_line = random.random() < 0.30 
        
        img, label = create_synthetic_plate(font_path, is_two_line)
        filename = f"synth_v2_{i:05d}.png"
        
        cv2.imwrite(str(output_dir / filename), img)
        csv_data.append({"filename": filename, "label": label})
        
    df = pd.DataFrame(csv_data)
    df.to_csv("data/processed/ocr_dataset/synthetic_labels.csv", index=False)
    print("✅ V2 Dataset generated!")

if __name__ == "__main__":
    generate_dataset()