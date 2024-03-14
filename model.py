import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBlock(nn.Module):
    """Double Convolution and Group Normalization block with ReLU activation."""
    def __init__(self, in_channels, out_channels, num_groups=8):
        super(ConvBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Encoder(nn.Module):
    """Downsampling block."""
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.encoder_block = nn.Sequential(
            nn.MaxPool3d(2, 2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder_block(x)

class Decoder(nn.Module):
    """Upsampling block."""
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1_upsampled = self.up(x1)
        # Calculate padding for dimension matching
        padding = [((sz2 - sz1) // 2, (sz2 - sz1) - (sz2 - sz1) // 2) for sz1, sz2 in zip(x1_upsampled.shape[2:], x2.shape[2:])]
        padding = [val for sublist in padding for val in sublist][::-1]  # Flatten and reverse the padding order
        x1_padded = F.pad(x1_upsampled, padding)
        x = torch.cat([x2, x1_padded], dim=1)
        return self.conv_block(x)

class OutputConv(nn.Module):
    """Final convolution to output the segmentation map."""
    def __init__(self, in_channels, out_channels):
        super(OutputConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet3d(nn.Module):
    """3D U-Net Architecture."""
    def __init__(self, in_channels, n_classes):
        super(UNet3d, self).__init__()
        self.in_conv = ConvBlock(in_channels, 16)
        self.e1 = Encoder(16, 16*2)
        self.e2 = Encoder(16*2, 16*4)
        self.e3 = Encoder(16*4, 16*8)
        self.e4 = Encoder(16*8, 16*8)
        self.d1 = Decoder(16*16, 16*4)
        self.d2 = Decoder(16*8, 16*2)
        self.d3 = Decoder(16*4, 16)
        self.d4 = Decoder(16*2, 16)
        self.out_conv = OutputConv(16, n_classes)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.e1(x1)
        x3 = self.e2(x2)
        x4 = self.e3(x3)
        x5 = self.e4(x4)
        x = self.d1(x5, x4)
        x = self.d2(x, x3)
        x = self.d3(x, x2)
        x = self.d4(x, x1)
        logits = self.out_conv(x)
        return logits
    
    
def get_loss(logits, mask):
    from monai.losses import DiceFocalLoss
    # Convert mask to long and add new axis to match logits dimension
    mask_d = mask.long()
    # Initialize DiceFocalLoss with specific settings
    loss = DiceFocalLoss(to_onehot_y=True, sigmoid=True)(logits, mask_d[None, ...])
    return loss

def get_dice(mask1, mask2):
    # Flatten masks and calculate Dice coefficient
    mask1 = (mask1 == 1).flatten()
    mask2 = (mask2 == 1).flatten()
    dice = ((np.sum(mask1 & mask2) * 2) + 1e-6) / (np.sum(mask1) + np.sum(mask2) + 1e-6)
    return dice

