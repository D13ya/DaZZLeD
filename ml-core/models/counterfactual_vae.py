"""
Counterfactual VAE (Domain Transfer)

This module provides a lightweight VAE-style generator that conditions on a
domain embedding to produce counterfactual images. It is intended for use as a
frozen generator during hash training (CF-SimCLR + DHD).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class CounterfactualVAE(nn.Module):
    def __init__(
        self,
        num_domains: int,
        content_dim: int = 512,
        domain_dim: int = 64,
        base_channels: int = 256,
    ):
        super().__init__()
        if num_domains <= 0:
            raise ValueError("num_domains must be > 0")
        self.num_domains = num_domains

        resnet = models.resnet18(weights=None)
        encoder_out = resnet.fc.in_features
        self.content_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.content_proj = nn.Linear(encoder_out, content_dim)

        self.domain_embed = nn.Embedding(num_domains, domain_dim)
        self.fc = nn.Linear(content_dim + domain_dim, base_channels * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels, 128, kernel_size=4, stride=2, padding=1),  # 7 -> 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 14 -> 28
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 28 -> 56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 56 -> 112
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),   # 112 -> 224
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        content = self.content_encoder(x)
        content = torch.flatten(content, 1)
        return self.content_proj(content)

    def decode(self, content: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        domain = self.domain_embed(domain_id)
        z = torch.cat([content, domain], dim=1)
        h = self.fc(z).view(z.size(0), -1, 7, 7)
        return self.decoder(h)

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        content = self.encode(x)
        return self.decode(content, domain_id)

    def generate_counterfactual(
        self,
        x: torch.Tensor,
        source_domain: torch.Tensor,
        target_domain: torch.Tensor,
    ) -> torch.Tensor:
        del source_domain
        return self.forward(x, target_domain)
