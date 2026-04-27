import os
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
from vocab import encode_label

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def collate_fn(batch):
    """
    Custom collate function for CTC Loss.
    Flattens the text targets into a 1D tensor and tracks their lengths.
    """
    images, labels = zip(*batch)
    
    # Stack images into [Batch, Channels, H, W]
    images = torch.stack(images, 0)
    
    # Track the length of each target string
    target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    
    # Flatten all encoded labels into a single 1D tensor
    encoded_labels = []
    for label in labels:
        encoded_labels.extend(encode_label(label))
    targets = torch.tensor(encoded_labels, dtype=torch.long)
    
    return images, targets, target_lengths

class CRNNTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing OCR Training on {self.device}...")
        
        # Paths (Ensure these match your directory structure)
        self.csv_path = "data/processed/ocr_dataset/synthetic_labels.csv"
        self.img_dir = "data/processed/ocr_dataset/synthetic_images"
        self.save_dir = Path("models/checkpoints/ocr_models")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Hyperparameters
        self.epochs = 50
        self.batch_size = 16
        self.learning_rate = 0.001
        self.seq_length = 31 # The fixed sequence length from our VGG backbone

        # Initialize Dataloader with custom collate_fn
        self.dataset = OCRDataset(csv_path=self.csv_path, img_dir=self.img_dir)
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=4, pin_memory=True,
            collate_fn=collate_fn
        )

        # Initialize Model, Loss, and Optimizer
        self.model = CRNN().to(self.device)
        # zero_infinity=True prevents infinite loss errors early in training
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True) 
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}/{self.epochs}")
            
            for images, targets, target_lengths in pbar:
                images = images.to(self.device)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)
                
                batch_size = images.size(0)
                
                self.optimizer.zero_grad()
                
                # 1. Forward Pass -> [Batch, SeqLen, Classes]
                outputs = self.model(images)
                
                # 2. PyTorch CTCLoss requires log probabilities
                log_probs = nn.functional.log_softmax(outputs, dim=2)
                
                # 3. CTCLoss expects shape [SeqLen, Batch, Classes]
                log_probs = log_probs.permute(1, 0, 2)
                
                # 4. Define input lengths (31 for every image in the batch)
                input_lengths = torch.full(size=(batch_size,), fill_value=self.seq_length, dtype=torch.long).to(self.device)
                
                # 5. Calculate Loss
                loss = self.criterion(log_probs, targets, input_lengths, target_lengths)
                
                # 6. Backpropagation
                loss.backward()
                # Clip gradients to prevent exploding gradients in the LSTM
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'CTC Loss': f"{loss.item():.4f}"})
            
            avg_loss = epoch_loss / len(self.dataloader)
            logger.info(f"Epoch {epoch} | Average CTC Loss: {avg_loss:.4f}")
            
            # Save checkpoints
            if epoch % 1 == 0 or epoch == self.epochs: 
                ckpt_path = self.save_dir / f"crnn_epoch_{epoch}.pt"
                torch.save(self.model.state_dict(), ckpt_path)
                logger.info(f"💾 Saved OCR checkpoint to {ckpt_path}")

if __name__ == "__main__":
    trainer = CRNNTrainer()
    trainer.train()