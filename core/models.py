
import torch
import torch.nn as nn
import torch.nn.functional as F

class DisentanglingVAE(nn.Module):
    def __init__(self, latent_dim=256, image_channels=3, hidden_dims=[32, 64, 128, 256]):
        super(DisentanglingVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        modules = []
        in_channels = image_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 8 * 8, latent_dim)  # فرض: تصویر 128x128
        self.fc_logvar = nn.Linear(hidden_dims[-1] * 8 * 8, latent_dim)

        # Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 8 * 8)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        self.decoder = nn.Sequential(
            *modules,
            nn.ConvTranspose2d(hidden_dims[-1], out_channels=image_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  # خروجی [-1, 1]
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(h.size(0), -1, 8, 8)  # Unflatten to match encoder output
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, beta=4.0):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl_div, recon_loss, kl_div
