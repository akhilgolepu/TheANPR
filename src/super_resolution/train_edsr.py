import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import mlflow
import logging

# Import your custom modules
from dataset import SRPatchDataset
from edsr_model import EDSR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class SRTrainer:
    def __init__(self, config_path: str = "configs/sr_config.yaml"):
        # 1. Load Configuration
        project_root = Path(__file__).resolve().parent.parent.parent
        with open(project_root / config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing Training on {self.device}...")

        # 2. Setup DataLoaders
        self.dataset = SRPatchDataset(
            lr_dir=project_root / self.config['paths']['lr_data'],
            hr_dir=project_root / self.config['paths']['hr_data'],
            scale=self.config['params']['scale_factor']
        )
        
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=self.config['training']['batch_size'], 
            shuffle=True, 
            num_workers=self.config['training']['num_workers'],
            pin_memory=True # Speeds up CPU to GPU data transfer
        )

        # 3. Setup Model, Loss, and Optimizer
        self.model = EDSR(scale_factor=self.config['params']['scale_factor']).to(self.device)
        self.criterion = nn.L1Loss() # L1 Loss preserves sharp edges better than MSE
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['training']['learning_rate'])
        
        self.save_dir = project_root / self.config['training']['save_dir']
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        epochs = self.config['training']['epochs']
        
        # Start MLflow Tracking
        mlflow.set_experiment("Super_Resolution_EDSR")
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params(self.config['training'])
            mlflow.log_param("model", "EDSR")
            
            logger.info(f"Starting training for {epochs} epochs...")
            
            for epoch in range(1, epochs + 1):
                self.model.train()
                running_loss = 0.0
                
                # TQDM progress bar for the batches
                pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}/{epochs}")
                
                for lr_imgs, hr_imgs in pbar:
                    # Move data to GPU
                    lr_imgs = lr_imgs.to(self.device)
                    hr_imgs = hr_imgs.to(self.device)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    sr_imgs = self.model(lr_imgs)
                    
                    # Calculate loss and backpropagate
                    loss = self.criterion(sr_imgs, hr_imgs)
                    loss.backward()
                    self.optimizer.step()
                    
                    running_loss += loss.item()
                    pbar.set_postfix({'L1 Loss': f"{loss.item():.4f}"})
                
                # Epoch average metrics
                avg_loss = running_loss / len(self.dataloader)
                logger.info(f"Epoch {epoch} | Average Loss: {avg_loss:.4f}")
                mlflow.log_metric("train_loss", avg_loss, step=epoch)
                
                # Save checkpoint every 10 epochs
                if epoch % 10 == 0 or epoch == epochs:
                    ckpt_path = self.save_dir / f"edsr_epoch_{epoch}.pt"
                    torch.save(self.model.state_dict(), ckpt_path)
                    logger.info(f"💾 Checkpoint saved to {ckpt_path}")

if __name__ == "__main__":
    trainer = SRTrainer()
    trainer.train()