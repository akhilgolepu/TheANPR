import torch
import torch.nn as nn

try:
    from vocab import NUM_CLASSES
except ImportError:
    # Fallback if running directly: 36 chars + 1 CTC Blank = 37
    NUM_CLASSES = 37 

class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for Optical Character Recognition.
    """
    def __init__(self, img_channels: int = 1, num_classes: int = NUM_CLASSES):
        super(CRNN, self).__init__()
        
        # 1. CNN Feature Extractor (VGG-style)
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32x128 -> 16x64
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16x64 -> 8x32
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Asymmetric pooling to preserve the horizontal sequence length
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)), # 8x32 -> 4x32
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)), # 4x32 -> 2x32
            
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True) # 2x32 -> 1x31
        )
        
        # 2. Sequence Modeling (BiLSTM)
        # 512 input features per time-step
        self.rnn = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, 
                           bidirectional=True, batch_first=True)
                           
        # 3. Classifier Head
        # Bidirectional means hidden_size is doubled (256 * 2)
        self.classifier = nn.Linear(256 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # A. Feature Extraction: [Batch, Channels, H, W] -> [Batch, 512, 1, 31]
        conv_out = self.cnn(x)
        
        # B. Tensor Bridging (The crucial step)
        batch, channels, h, w = conv_out.size()
        assert h == 1, f"CNN output height must be 1, got {h}"
        
        # Remove the height dimension: [Batch, 512, 1, 31] -> [Batch, 512, 31]
        conv_out = conv_out.squeeze(2) 
        
        # Swap Sequence (W) and Features (C): [Batch, 512, 31] -> [Batch, 31, 512]
        # This is the exact format the LSTM expects
        conv_out = conv_out.permute(0, 2, 1) 
        
        # C. Sequence Modeling: [Batch, 31, 512]
        rnn_out, _ = self.rnn(conv_out)
        
        # D. Classification: [Batch, Sequence_Length, NUM_CLASSES]
        output = self.classifier(rnn_out)
        
        return output

if __name__ == "__main__":
    # Sanity Check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN().to(device)
    
    # Simulating a batch of 4 grayscale plates (like the Dataloader output)
    dummy_input = torch.randn(4, 1, 32, 128).to(device)
    
    output = model(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")
    print(f"CRNN Output Shape: {output.shape}")
    print(f"Expected CRNN Output Shape: (4, 31, 37) -> (Batch, Sequence_Length, Num_Classes)")
    print("✅ CRNN Architecture Validated.")