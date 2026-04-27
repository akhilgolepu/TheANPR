import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import logging

# Import our custom OCR modules
from dataset import OCRDataset
from model_crnn import CRNN
from train_crnn import collate_fn  # Reuse the exact same collate function

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class CRNNFineTuner:
    def __init__(self, base_model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🚀 Initializing Fine-Tuning on {self.device}...")
        
        # Paths to the real, manually labeled ESRGAN data
        self.csv_path = "data/processed/ocr_dataset/labels.csv" 
        self.img_dir = "data/processed/ocr_dataset/images" 
        
        self.save_dir = Path("models/checkpoints/ocr_models/finetuned")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Fine-Tuning Hyperparameters
        self.epochs = 30  
        self.batch_size = 8  
        self.learning_rate = 5e-5 
        self.seq_length = 31

        # Initialize Dataloader
        self.dataset = OCRDataset(csv_path=self.csv_path, img_dir=self.img_dir)
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=2, pin_memory=True,
            collate_fn=collate_fn
        )

        # Load the Pre-trained Stage 1 Model
        self.model = CRNN().to(self.device)
        logger.info(f"📦 Loading base weights from {base_model_path}")
        self.model.load_state_dict(torch.load(base_model_path, map_location=self.device))
        
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True) 
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            pbar = tqdm(self.dataloader, desc=f"Fine-Tune Epoch {epoch}/{self.epochs}")
            
            for images, targets, target_lengths in pbar:
                images = images.to(self.device)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)
                
                batch_size = images.size(0)
                self.optimizer.zero_grad()
                
                outputs = self.model(images)
                log_probs = nn.functional.log_softmax(outputs, dim=2)
                log_probs = log_probs.permute(1, 0, 2)
                
                input_lengths = torch.full(size=(batch_size,), fill_value=self.seq_length, dtype=torch.long).to(self.device)
                
                loss = self.criterion(log_probs, targets, input_lengths, target_lengths)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'CTC Loss': f"{loss.item():.4f}"})
            
            avg_loss = epoch_loss / len(self.dataloader)
            logger.info(f"📉 Epoch {epoch} | Avg Loss: {avg_loss:.4f}")
            
            # Save every 5 epochs
            if epoch % 5 == 0 or epoch == self.epochs:
                ckpt_path = self.save_dir / f"crnn_finetuned_epoch_{epoch}.pt"
                torch.save(self.model.state_dict(), ckpt_path)
                logger.info(f"💾 Saved Fine-Tuned checkpoint to {ckpt_path}")

if __name__ == "__main__":
    BASE_WEIGHTS = "models/checkpoints/ocr_models/finetuned/crnn_finetuned_epoch_15.pt"
    trainer = CRNNFineTuner(BASE_WEIGHTS)
    trainer.train()