import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    VGG-style Discriminator for ESRGAN.
    Takes an image and predicts if it is Real or Fake (Generated).
    """
    def __init__(self, in_channels: int = 3, features: int = 64):
        super().__init__()
        
        # We use a sequential block to build the VGG structure cleanly
        self.net = nn.Sequential(
            # Input layer (No Batch Norm on the first layer)
            nn.Conv2d(in_channels, features, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Downsample block 1
            self._block(features, features, stride=2),
            self._block(features, features * 2, stride=1),
            
            # Downsample block 2
            self._block(features * 2, features * 2, stride=2),
            self._block(features * 2, features * 4, stride=1),
            
            # Downsample block 3
            self._block(features * 4, features * 4, stride=2),
            self._block(features * 4, features * 8, stride=1),
            
            # Downsample block 4
            self._block(features * 8, features * 8, stride=2)
        )

        # The classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(features * 8, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1) # Outputs a single raw logit (Real vs Fake)
        )

    def _block(self, in_channels: int, out_channels: int, stride: int):
        """Helper to create Conv -> BatchNorm -> LeakyReLU blocks"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.net(x)
        validity = self.classifier(features)
        return validity

if __name__ == "__main__":
    dummy_hr = torch.randn(1, 3, 128, 128) 
    discriminator = Discriminator()
    output = discriminator(dummy_hr)
    print("✅ Discriminator Architecture Validated. Ready for Adversarial Combat.")