import torch
import torch.nn as nn

class AuxiliaryNetwork(nn.Module):
    """
    Predicts a delta in latent space with same shape as z (B, 256).
    """
    def __init__(self, latent_dim=256, out_scale=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.Tanh()
        )
        self.out_scale = out_scale

    def forward(self, z):
        return self.net(z) * self.out_scale