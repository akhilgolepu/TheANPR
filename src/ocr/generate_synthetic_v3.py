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
    state = random.choice(STATES)
    district = f"{random.randint(1, 99):02d}"
    letters = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=random.randint(1, 2)))
    numbers = f"{random.randint(1, 9999):04d}"
    
    full_string = f"{state} {district} {letters} {numbers}"
    top_line = f"{state} {district}"
    bottom_line = f"{letters} {numbers}"
    return full_string, top_line, bottom_line

def get_random_font(font_dir: Path, size: int):
    # Fallback to system font if no custom fonts are found
    if not font_dir.exists() or not list(font_dir.glob("*.ttf")):
        return ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf", size=size)
    
    fonts = list(font_dir.glob("*.ttf"))
    selected_font = random.choice(fonts)
    try:
        return ImageFont.truetype(str(selected_font), size=size)
    except IOError:
        return ImageFont.load_default()

def create_synthetic_plate(font_dir: Path, is_two_line: bool) -> tuple:
    full_text, top, bottom = generate_plate_string()
    
    if is_two_line:
        width, height = 128, 128
        img = Image.new('L', (width, height), color=255)
        draw = ImageDraw.Draw(img)
        font = get_random_font(font_dir, size=35)
        
        bbox_t = draw.textbbox((0, 0), top, font=font)
        draw.text(((width - (bbox_t[2] - bbox_t[0])) / 2, 10), top, font=font, fill=0)
        
        bbox_b = draw.textbbox((0, 0), bottom, font=font)
        draw.text(((width - (bbox_b[2] - bbox_b[0])) / 2, 60), bottom, font=font, fill=0)
    else:
        width, height = 256, 64
        img = Image.new('L', (width, height), color=255)
        draw = ImageDraw.Draw(img)
        font = get_random_font(font_dir, size=40)
        
        bbox = draw.textbbox((0, 0), full_text, font=font)
        draw.text(((width - (bbox[2] - bbox[0])) / 2, 5), full_text, font=font, fill=0)
    
    img_cv = np.array(img)
    
    # Augmentations: Rotation
    angle = random.uniform(-3, 3)
    M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
    img_cv = cv2.warpAffine(img_cv, M, (width, height), borderValue=255)
    
    # Augmentations: Noise
    noise = np.random.normal(0, 12, img_cv.shape).astype(np.float32)
    img_cv = np.clip(img_cv.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Draw faint checkerboard grid (GAN artifact simulation)
    grid_spacing = random.randint(3, 5)
    for x in range(0, width, grid_spacing):
        cv2.line(img_cv, (x, 0), (x, height), 220, 1)
    for y in range(0, height, grid_spacing):
        cv2.line(img_cv, (0, y), (width, y), 220, 1)
        
    if random.random() > 0.4:
        img_cv = cv2.GaussianBlur(img_cv, (3, 3), 0)

    clean_label = full_text.replace(" ", "")
    return img_cv, clean_label

def generate_dataset(num_samples: int = 20000):
    output_dir = Path("data/processed/ocr_dataset/synthetic_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    font_dir = Path("data/fonts")
    font_dir.mkdir(parents=True, exist_ok=True)
    
    csv_data = []
    
    print(f"🏭 Generating {num_samples} V3 synthetic plates (Multi-Font & Mixed formats)...")
    for i in tqdm(range(num_samples)):
        is_two_line = random.random() < 0.35 
        img, label = create_synthetic_plate(font_dir, is_two_line)
        filename = f"synth_v3_{i:05d}.png"
        
        cv2.imwrite(str(output_dir / filename), img)
        csv_data.append({"filename": filename, "label": label})
        
    df = pd.DataFrame(csv_data)
    df.to_csv("data/processed/ocr_dataset/synthetic_labels.csv", index=False)
    print("✅ V3 Dataset generated!")

if __name__ == "__main__":
    generate_dataset()