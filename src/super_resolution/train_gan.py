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

# Ensure these match your actual filenames!
from dataset import SRPatchDataset
from model_edsr import EDSR
from discriminator import Discriminator
from loss import PerceptualLoss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class ESRGANTrainer:
    def __init__(self, config_path: str = "configs/sr_config.yaml"):
        project_root = Path(__file__).resolve().parent.parent.parent
        with open(project_root / config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing Adversarial Combat on {self.device}...")

        # 1. Setup DataLoader
        self.dataset = SRPatchDataset(
            lr_dir=project_root / self.config['paths']['lr_data'],
            hr_dir=project_root / self.config['paths']['hr_data'],
            scale=self.config['params']['scale_factor']
        )
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.config['gan_training']['batch_size'], 
            shuffle=True, num_workers=4, pin_memory=True
        )

        # 2. Initialize Networks
        self.generator = EDSR(scale_factor=self.config['params']['scale_factor']).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # WARM START: Load ESRGAN weights
        pretrained_path = project_root / "models/checkpoints/esrgan_models/esrgan_epoch_20.pt"
        self.generator.load_state_dict(torch.load(pretrained_path, map_location=self.device))
        logger.info("✅ Generator warm-started with Epoch 20 ESRGAN weights.")
        
        # 3. Initialize Losses
        self.l1_loss = nn.L1Loss().to(self.device)
        self.perceptual_loss = PerceptualLoss(self.device)
        # BCEWithLogitsLoss because our Discriminator outputs raw logits
        self.bce_loss = nn.BCEWithLogitsLoss().to(self.device) 

        # 4. Initialize Optimizers
        self.opt_g = optim.Adam(self.generator.parameters(), lr=self.config['gan_training']['lr_generator'])
        self.opt_d = optim.Adam(self.discriminator.parameters(), lr=self.config['gan_training']['lr_discriminator'])
        
        self.save_dir = project_root / self.config['gan_training']['save_dir']
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        epochs = self.config['gan_training']['epochs']
        lambda_pixel = self.config['gan_training']['lambda_pixel']
        lambda_perceptual = self.config['gan_training']['lambda_perceptual']
        lambda_adv = self.config['gan_training']['lambda_adv']
        
        mlflow.set_experiment("Super_Resolution_ESRGAN")
        with mlflow.start_run():
            mlflow.log_params(self.config['gan_training'])
            
            for epoch in range(1, epochs + 1):
                self.generator.train()
                self.discriminator.train()
                
                loss_g_tracker, loss_d_tracker = 0.0, 0.0
                pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}/{epochs}")
                
                for lr_imgs, hr_imgs in pbar:
                    lr_imgs, hr_imgs = lr_imgs.to(self.device), hr_imgs.to(self.device)
                    batch_size = lr_imgs.size(0)
                    
                    # Create labels for the Discriminator (1 = Real, 0 = Fake)
                    real_labels = torch.ones((batch_size, 1), device=self.device)
                    fake_labels = torch.zeros((batch_size, 1), device=self.device)
                    
                    # -------------------------
                    #  Train Discriminator
                    # -------------------------
                    self.opt_d.zero_grad()
                    
                    # Generate fake images (detach so we don't calculate gradients for G yet)
                    fake_imgs = self.generator(lr_imgs)
                    
                    # Discriminator predictions
                    pred_real = self.discriminator(hr_imgs)
                    pred_fake = self.discriminator(fake_imgs.detach())
                    
                    # Discriminator Loss
                    loss_d_real = self.bce_loss(pred_real, real_labels)
                    loss_d_fake = self.bce_loss(pred_fake, fake_labels)
                    loss_d = (loss_d_real + loss_d_fake) / 2
                    
                    loss_d.backward()
                    self.opt_d.step()
                    
                    # -------------------------
                    #  Train Generator
                    # -------------------------
                    self.opt_g.zero_grad()
                    
                    # Generate fake images again (this time attached to graph)
                    pred_fake_g = self.discriminator(fake_imgs)
                    
                    # Generator Losses
                    # 1. Adversarial (Try to fool D by comparing output to REAL labels)
                    l_adv = self.bce_loss(pred_fake_g, real_labels)
                    # 2. Perceptual (VGG Features)
                    l_perceptual = self.perceptual_loss(fake_imgs, hr_imgs)
                    # 3. Pixel (L1)
                    l_pixel = self.l1_loss(fake_imgs, hr_imgs)
                    
                    # Total Generator Loss
                    loss_g = (lambda_pixel * l_pixel) + (lambda_perceptual * l_perceptual) + (lambda_adv * l_adv)
                    
                    loss_g.backward()
                    self.opt_g.step()
                    
                    # Update metrics
                    loss_g_tracker += loss_g.item()
                    loss_d_tracker += loss_d.item()
                    
                    pbar.set_postfix({'Loss G': f"{loss_g.item():.3f}", 'Loss D': f"{loss_d.item():.3f}"})
                
                # Epoch Logging
                avg_loss_g = loss_g_tracker / len(self.dataloader)
                avg_loss_d = loss_d_tracker / len(self.dataloader)
                
                logger.info(f"Epoch {epoch} | Generator Loss: {avg_loss_g:.4f} | Discriminator Loss: {avg_loss_d:.4f}")
                mlflow.log_metrics({"generator_loss": avg_loss_g, "discriminator_loss": avg_loss_d}, step=epoch)
                
                # Save Checkpoints
                if epoch % 5 == 0 or epoch == epochs:
                    ckpt_path = self.save_dir / f"esrgan_epoch_{epoch}.pt"
                    torch.save(self.generator.state_dict(), ckpt_path)
                    logger.info(f"💾 Checkpoint saved to {ckpt_path}")

if __name__ == "__main__":
    trainer = ESRGANTrainer()
    trainer.train()