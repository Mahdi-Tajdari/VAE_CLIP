import torch
import torch.nn as nn

class AuxiliaryNetwork(nn.Module):
    """
    A more complex neural network to find the disentangled latent direction.
    """
    def __init__(self, latent_dim=4 * 64 * 64, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, latent_dim),
            nn.Tanh()
        )
        
    def forward(self, latent_vector):
        """
        Input: latent_vector (a 4D tensor)
        Output: a small delta_vector (a 4D tensor)
        """
        batch_size = latent_vector.shape[0]
        flattened_vector = latent_vector.view(batch_size, -1)
        
        delta_flattened = self.net(flattened_vector)
        
        delta_vector = delta_flattened.view(latent_vector.shape)
        
        return delta_vector