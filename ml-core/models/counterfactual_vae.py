"""
Counterfactual VAE (Domain Transfer)

This module provides a variational autoencoder that conditions on a domain
embedding to produce counterfactual images. It is intended for use as a frozen
generator during hash training (CF-SimCLR + DHD).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class CounterfactualVAE(nn.Module):
    def __init__(
        self,
        num_domains: int,
        latent_dim: int = 512,
        domain_dim: int = 64,
        base_channels: int = 256,
    ):
        super().__init__()
        if num_domains <= 0:
            raise ValueError("num_domains must be > 0")
        self.num_domains = num_domains
        self.latent_dim = latent_dim

        resnet = models.resnet18(weights=None)
        encoder_out = resnet.fc.in_features
        self.content_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.content_proj = nn.Sequential(
            nn.Linear(encoder_out, latent_dim),
            nn.SiLU(),
        )
        self.post_norm = nn.GroupNorm(_group_count(latent_dim), latent_dim)
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.logvar = nn.Linear(latent_dim, latent_dim)

        self.domain_embed = nn.Embedding(num_domains, domain_dim)
        self.fc = nn.Linear(latent_dim + domain_dim, base_channels * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels, 128, kernel_size=4, stride=2, padding=1),  # 7 -> 14
            nn.GroupNorm(_group_count(128), 128),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 14 -> 28
            nn.GroupNorm(_group_count(64), 64),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 28 -> 56
            nn.GroupNorm(_group_count(32), 32),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 56 -> 112
            nn.GroupNorm(_group_count(16), 16),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),   # 112 -> 224
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        content = self.content_encoder(x)
        content = torch.flatten(content, 1)
        hidden = self.content_proj(content)
        hidden = self.post_norm(hidden.view(hidden.size(0), self.latent_dim, 1, 1)).view(hidden.size(0), self.latent_dim)
        return self.mu(hidden), self.logvar(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        domain = self.domain_embed(domain_id)
        z = torch.cat([z, domain], dim=1)
        h = self.fc(z).view(z.size(0), -1, 7, 7)
        return self.decoder(h)

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, domain_id)
        return recon, mu, logvar

    def generate_counterfactual(
        self,
        x: torch.Tensor,
        source_domain: torch.Tensor,
        target_domain: torch.Tensor,
    ) -> torch.Tensor:
        del source_domain
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, target_domain)
