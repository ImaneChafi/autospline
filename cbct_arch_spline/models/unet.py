"""
U-Net with EfficientNet-B3 encoder for CBCT arch heatmap prediction.

Input:  (B, 1, H, W) normalized CBCT axial slice
Output: (B, 1, H, W) keypoint heatmap in [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Decoder blocks
# ---------------------------------------------------------------------------


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.upsample(x)
        if skip is not None:
            # Handle size mismatch
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# EfficientNet U-Net
# ---------------------------------------------------------------------------


class EfficientUNet(nn.Module):
    """
    U-Net using EfficientNet-B3 as encoder.

    Encoder output channels (EfficientNet-B3 feature map channels):
      stage 0 (1/2):   24
      stage 1 (1/4):   32
      stage 2 (1/8):   48
      stage 3 (1/16):  136
      stage 4 (1/32):  1536 (head conv output)
    """

    ENCODER_CHANNELS = [24, 32, 48, 136, 1536]

    def __init__(self, in_channels: int = 1, num_classes: int = 1):
        super().__init__()
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required. pip install timm")

        # Load pretrained EfficientNet-B3 and adapt first conv for 1-channel input
        self.encoder = timm.create_model(
            "efficientnet_b3",
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )

        # Patch the first conv to accept 1-channel input
        first_conv = self.encoder.conv_stem
        new_conv = nn.Conv2d(
            in_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )
        # Initialize by averaging over RGB channels
        with torch.no_grad():
            new_conv.weight = nn.Parameter(first_conv.weight.mean(dim=1, keepdim=True))
        self.encoder.conv_stem = new_conv

        enc = self.ENCODER_CHANNELS

        self.dec4 = DecoderBlock(enc[4], enc[3], 256)
        self.dec3 = DecoderBlock(256, enc[2], 128)
        self.dec2 = DecoderBlock(128, enc[1], 64)
        self.dec1 = DecoderBlock(64, enc[0], 32)
        self.dec0 = DecoderBlock(32, 0, 16)  # no skip at original resolution

        self.head = nn.Sequential(
            ConvBnRelu(16, 16),
            nn.Conv2d(16, num_classes, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0, s1, s2, s3, s4 = self.encoder(x)

        x = self.dec4(s4, s3)
        x = self.dec3(x, s2)
        x = self.dec2(x, s1)
        x = self.dec1(x, s0)
        x = self.dec0(x, None)
        return self.head(x)


# ---------------------------------------------------------------------------
# Fallback: simple U-Net from scratch (no timm dependency)
# ---------------------------------------------------------------------------


class _DoubleConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            ConvBnRelu(in_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )


class SimpleUNet(nn.Module):
    """Lightweight U-Net without pretrained backbone."""

    def __init__(self, in_channels: int = 1, num_classes: int = 1, base_ch: int = 32):
        super().__init__()
        b = base_ch

        self.enc1 = _DoubleConv(in_channels, b)
        self.enc2 = _DoubleConv(b, b * 2)
        self.enc3 = _DoubleConv(b * 2, b * 4)
        self.enc4 = _DoubleConv(b * 4, b * 8)
        self.bottleneck = _DoubleConv(b * 8, b * 16)

        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, kernel_size=2, stride=2)
        self.dec4 = _DoubleConv(b * 16, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.dec3 = _DoubleConv(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(b * 2, b)

        self.head = nn.Sequential(
            nn.Conv2d(b, num_classes, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def build_model(
    use_pretrained: bool = True,
    in_channels: int = 1,
    num_classes: int = 1,
) -> nn.Module:
    """Factory: returns EfficientUNet if timm available, else SimpleUNet."""
    if use_pretrained and TIMM_AVAILABLE:
        return EfficientUNet(in_channels=in_channels, num_classes=num_classes)
    return SimpleUNet(in_channels=in_channels, num_classes=num_classes)
