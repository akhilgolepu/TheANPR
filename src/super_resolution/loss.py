import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class PerceptualLoss(nn.Module):
    """
    Calculates the Perceptual (Feature) Loss between two images.
    Uses a pre-trained VGG19 network to extract deep feature maps.
    """
    def __init__(self, device: torch.device):
        super().__init__()
        
        # Load the pre-trained VGG19 network
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        
        # We extract features right before the 5th max-pooling layer (layer 35)
        # This captures deep semantic structures (edges, shapes, text contours)
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:35]).to(device).eval()
        
        # Freeze VGG19 - we use it as a ruler, we don't want to train it!
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # We compare the feature maps using L1 Loss
        self.criterion = nn.L1Loss()
        
        # Normalization constants expected by VGG19
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Normalize images for VGG
        gen_norm = (generated - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        
        # Extract features
        gen_features = self.feature_extractor(gen_norm)
        target_features = self.feature_extractor(target_norm)
        
        # Calculate loss
        return self.criterion(gen_features, target_features)

if __name__ == "__main__":
    # Sanity Check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = PerceptualLoss(device)
    
    dummy_gen = torch.randn(1, 3, 128, 128).to(device)
    dummy_target = torch.randn(1, 3, 128, 128).to(device)
    
    loss_val = loss_fn(dummy_gen, dummy_target)
    print(f"Perceptual Loss Value: {loss_val.item():.4f}")
    print("✅ Perceptual Loss initialized and validated. VGG19 is locked.")