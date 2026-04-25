import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Standard EDSR Residual Block
    Note: No Batch normalization layers are used
    """

    def __init__(self, channels: int, res_scale: float = 0.1):
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Scale the residual to stabilize training before adding to input
        res = self.body(x) * self.res_scale
        return x + res

class EDSR(nn.Module):
    """
    Enhanced Deep Residual Network for Single Image Super-Resolution
    """
    def __init__(self, scale_factor: int=4, num_channels: int=3,
                    num_features: int=64, num_blocks: int=16, res_scale: float = 0.1):
        super().__init__()

        # Head: Feature extraction from the LR image
        self.head = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)

        # Body: Deep feature mapping via Residual Blocks
        body_layers = [ResidualBlock(num_features, res_scale) for _ in range(num_blocks)]
        body_layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.body = nn.Sequential(*body_layers)

        # Tail: Upsampling via Sub-Pixel Convolution (PixelShuffle)
        self.tail = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Extract features from the LR image
        x = self.head(x)

        # 2. Deep feature mapping through the body
        res = self.body(x)
        res += x  # Global residual connection

        # 3. Upsample to HR space
        x = self.tail(res)
        return x

if __name__ == "__main__":
    # Sanity check: Test the tensor shapes
    model = EDSR(scale_factor=4)
    dummy_lr = torch.randn(1, 3, 32, 32)  # Batch of 1, 3 channels, 32x32 LR image
    output_hr = model(dummy_lr)

    print(f"LR Input Shape: {dummy_lr.shape}")
    print(f"HR Output Shape: {output_hr.shape}")
    assert output_hr.shape == (1, 3, 128, 128), "Output shape mismatch! Expected (1, 3, 128, 128)"
    print("✅ Model Architechture Validated.")