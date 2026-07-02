"""
ArchSplineNet: CNN encoder -> jaw-conditioned Bezier control-point head.

Input: (B, 3, CANVAS, CANVAS) stacking [mip, label_mip, geo_channel]
       (3 channels deliberately -- matches torchvision's pretrained
       ResNet input shape, so the same model class works whether or not
       you have a pretrained backbone available)
Plus:  jaw (B,) long tensor, 0=lower 1=upper -> conditioned via embedding
       concatenated to the pooled feature vector before the regression head

Output: (B, K, 2) Bezier control points in canvas pixel coordinates.

Backbone: tries torchvision ResNet18 (ImageNet-pretrained) first, since
that matters a lot at n~163; falls back to a small custom CNN if
torchvision isn't available (e.g. offline sandbox) so this file is
still runnable everywhere for testing.
"""
import torch
import torch.nn as nn

try:
    import torchvision.models as tvm
    _HAS_TORCHVISION = True
except ImportError:
    _HAS_TORCHVISION = False


class SmallCNNBackbone(nn.Module):
    """Fallback backbone -- no pretrained weights needed."""
    def __init__(self, in_ch=3, out_dim=512):
        super().__init__()
        def block(cin, cout, stride=2):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.net = nn.Sequential(
            block(in_ch, 32), block(32, 64), block(64, 128),
            block(128, 256), block(256, out_dim),
            nn.AdaptiveAvgPool2d(1))
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x).flatten(1)


class ArchSplineNet(nn.Module):
    def __init__(self, n_control_points=24, use_pretrained=True,
                 jaw_embed_dim=16, max_correction_px=60.0):
        super().__init__()
        self.n_cp = n_control_points
        self.max_correction_px = max_correction_px

        if _HAS_TORCHVISION and use_pretrained:
            backbone = tvm.resnet18(
                weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None)
            backbone.fc = nn.Identity()
            self.backbone = backbone
            feat_dim = 512
        else:
            self.backbone = SmallCNNBackbone(in_ch=3, out_dim=512)
            feat_dim = 512

        self.jaw_embed = nn.Embedding(2, jaw_embed_dim)

        # Explicit numeric encoding of the geometric arch's control points
        # (NOT just the rendered image channel) -- this is what lets the
        # network actually condition on "where the geometric curve is" as
        # numbers, rather than hoping it infers that from a picture of a
        # line. Combined with the residual output below, this makes the
        # "predict a correction to the geometric baseline" framing an
        # architectural fact instead of an emergent (and apparently
        # unreliable, per the F_018/F_066 failures) hope.
        geo_dim = n_control_points * 2
        self.geo_encoder = nn.Sequential(
            nn.Linear(geo_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(feat_dim + jaw_embed_dim + 32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_control_points * 2),
        )

    def forward(self, x, jaw, geo_cp):
        """geo_cp: (B, n_control_points, 2) -- the geometric arch's own
        control points, in the SAME canvas coordinate space as the
        target. Output is geo_cp + predicted_delta: a residual
        correction, not a free-floating absolute prediction. This
        directly bounds predictions near a plausible curve even when the
        learned delta is imperfect, instead of letting the network
        extrapolate into anatomically nonsensical shapes."""
        feat = self.backbone(x)                       # (B, feat_dim)
        jaw_feat = self.jaw_embed(jaw)                  # (B, jaw_embed_dim)
        geo_flat = geo_cp.reshape(geo_cp.shape[0], -1)  # (B, n_cp*2)
        geo_feat = self.geo_encoder(geo_flat)           # (B, 32)
        combined = torch.cat([feat, jaw_feat, geo_feat], dim=1)
        raw_delta = self.head(combined).view(-1, self.n_cp, 2)
        # bound the correction magnitude -- prevents the network from
        # ever producing an unbounded, disconnected jump (e.g. the
        # floating squiggle seen on case 336's posterior region); it can
        # still apply the FULL bound when a large correction is genuinely
        # needed, just can't exceed it
        delta = torch.tanh(raw_delta) * self.max_correction_px
        return geo_cp + delta


if __name__ == "__main__":
    # smoke test
    model = ArchSplineNet(n_control_points=24, use_pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    jaw = torch.tensor([0, 1])
    geo_cp = torch.randn(2, 24, 2)
    out = model(x, jaw, geo_cp)
    print("output shape:", out.shape)
    assert out.shape == (2, 24, 2)
    print("model forward OK (residual: output = geo_cp + delta)")
