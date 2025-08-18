# core/auxiliary_network.py
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))

class AuxiliaryNetwork(nn.Module):
    """
    Predicts a delta in latent space with same shape as z (B, 4, H, W).
    """
    def __init__(self, in_ch=4, base_ch=64, n_blocks=4, out_scale=2.0):  # Changed to 2.0
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResBlock(base_ch) for _ in range(n_blocks)])
        self.head = nn.Conv2d(base_ch, in_ch, 3, padding=1)
        self.out_scale = out_scale

    def forward(self, z):
        h = self.stem(z)
        h = self.body(h)
        dz = self.head(h)
        dz_debug = torch.tanh(dz) * self.out_scale  # Debug output
        print(f"dz shape: {dz.shape}, dz mean: {dz_debug.mean().item():.4f}, dz max: {dz_debug.max().item():.4f}")
        return dz_debug