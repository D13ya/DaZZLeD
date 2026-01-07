import torch
import torch.nn as nn
import torch.nn.functional as F


class RecursiveHasher(nn.Module):
    def __init__(self, state_dim=128, hash_dim=96):
        super().__init__()

        # Tiny encoder to compress image features into the state space.
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, state_dim),
        )

        # Recursive core holding the thought vector.
        self.gru = nn.GRUCell(input_size=state_dim, hidden_size=state_dim)

        # Hash head projects the thought vector to the final fingerprint.
        self.head = nn.Linear(state_dim, hash_dim)

    def forward(self, img, prev_state):
        features = self.encoder(img)
        next_state = self.gru(features, prev_state)
        raw_hash = self.head(next_state)
        final_hash = F.normalize(raw_hash, p=2, dim=1)
        return next_state, final_hash
