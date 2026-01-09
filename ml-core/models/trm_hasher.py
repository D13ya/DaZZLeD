"""
Tiny Recursion Model (TRM) for Perceptual Hashing

Implementation based on: "Less is More: Recursive Reasoning with Tiny Networks"
https://arxiv.org/abs/2510.04871

Designed to address NeuralHash weaknesses identified in:
"Black-box Collision Attacks on Apple NeuralHash and Microsoft PhotoDNA"

Key design choices (paper-compliant):
1. Single tiny 2-layer network (shared for z and y updates)
2. Addition-based feature combination: z <- f(x + y + z), y <- f(y + z)
3. Deep supervision with N_sup steps
4. Larger hash_dim (128-192) for better collision resistance
5. y update excludes x (key property for task differentiation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Building Blocks
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class SwiGLU(nn.Module):
    """SwiGLU activation function."""
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, in_dim, bias=False)
        self.w3 = nn.Linear(in_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TRMBlock(nn.Module):
    """Single TRM block: RMSNorm -> SwiGLU MLP with residual."""
    def __init__(self, dim: int, hidden_mult: int = 4):
        super().__init__()
        hidden_dim = dim * hidden_mult
        self.norm = RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden_dim)

    def forward(self, x):
        return x + self.mlp(self.norm(x))


class TinyRecursiveNetwork(nn.Module):
    """
    The single tiny network used in TRM.

    Uses addition-based input composition (x + y + z) or (y + z).
    The network learns to distinguish the two tasks based on input composition.

    Architecture: 2 layers as per TRM paper (smaller = less overfitting).
    """
    def __init__(self, dim: int, n_layers: int = 2, hidden_mult: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            TRMBlock(dim, hidden_mult) for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class ImageEncoder(nn.Module):
    """
    Lightweight CNN encoder: image -> embedding.

    Uses GroupNorm (not BatchNorm) for stable inference with small batches.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        assert embed_dim % 8 == 0, f"embed_dim ({embed_dim}) must be divisible by 8"

        self.encoder = nn.Sequential(
            # Stage 1: 224 -> 112
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            # Stage 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            # Stage 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            # Stage 4: 28 -> 7
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=4, padding=1, bias=False),
            nn.GroupNorm(8, embed_dim),
            nn.GELU(),
            # Global pool -> flatten
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, img):
        return self.encoder(img)


# ============================================================================
# Main TRM Hasher
# ============================================================================

class TRMHasher(nn.Module):
    """
    TRM-based Perceptual Hasher.

    Paper-compliant implementation:
    - Uses ADDITION (x + y + z) not concatenation
    - Single shared network for both z and y updates
    - z update includes x: z <- f(x + y + z)
    - y update excludes x: y <- f(y + z)

    Args:
        embed_dim: Dimension for x, y, z embeddings (default: 256)
        hash_dim: Final hash output dimension (default: 128)
        n_layers: Number of layers in tiny network (default: 2)
        n_latent: Number of z updates per cycle (default: 6)
        t: Number of cycles per supervision step (default: 3)
    """
    def __init__(
        self,
        embed_dim: int = 256,
        hash_dim: int = 128,
        n_layers: int = 2,
        n_latent: int = 6,
        t: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hash_dim = hash_dim
        self.n_latent = n_latent
        self.t = t

        self.image_encoder = ImageEncoder(embed_dim)
        self.net = TinyRecursiveNetwork(embed_dim, n_layers=n_layers)
        self.output_head = nn.Linear(embed_dim, hash_dim, bias=False)

        self.y_init = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
        self.z_init = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

    def latent_recursion(self, x, y, z):
        """
        One full latent recursion cycle (paper-exact).

        z <- f(x + y + z)  [n times, includes x]
        y <- f(y + z)      [once, no x]
        """
        for _ in range(self.n_latent):
            z = self.net(x + y + z)
        y = self.net(y + z)
        return y, z

    def deep_recursion(self, x, y, z):
        """
        Deep recursion with t cycles:
        - t-1 cycles without gradients
        - 1 cycle with gradients
        """
        with torch.no_grad():
            for _ in range(self.t - 1):
                y, z = self.latent_recursion(x, y, z)

        y, z = self.latent_recursion(x, y, z)
        return y, z

    def forward(self, img, y=None, z=None):
        batch_size = img.size(0)
        device = img.device

        x = self.image_encoder(img)

        if y is None:
            y = self.y_init.expand(batch_size, -1).to(device)
        if z is None:
            z = self.z_init.expand(batch_size, -1).to(device)

        y, z = self.deep_recursion(x, y, z)

        hash_out = torch.sigmoid(self.output_head(y))

        return y.detach(), z.detach(), hash_out

    def inference(self, img, n_sup: int = 16):
        """Full inference with N_sup supervision steps."""
        batch_size = img.size(0)
        device = img.device

        with torch.no_grad():
            x = self.image_encoder(img)
            y = self.y_init.expand(batch_size, -1).to(device)
            z = self.z_init.expand(batch_size, -1).to(device)

            for _ in range(n_sup):
                for _ in range(self.t):
                    y, z = self.latent_recursion(x, y, z)

            hash_out = torch.sigmoid(self.output_head(y))

        return hash_out

    def get_binary_hash(self, img, n_sup: int = 16):
        """Get binary hash for storage/comparison."""
        continuous_hash = self.inference(img, n_sup)
        return (continuous_hash > 0.5).float()

    def hamming_distance(self, hash1, hash2):
        """Compute Hamming distance between binary hashes."""
        return (hash1 != hash2).sum(dim=-1)

    def cosine_similarity(self, img1, img2, n_sup: int = 16):
        """Compute cosine similarity between two images."""
        hash1 = self.inference(img1, n_sup)
        hash2 = self.inference(img2, n_sup)
        hash1 = F.normalize(hash1 - 0.5, p=2, dim=-1)
        hash2 = F.normalize(hash2 - 0.5, p=2, dim=-1)
        return F.cosine_similarity(hash1, hash2, dim=-1)


# ============================================================================
# ONNX Export Wrapper
# ============================================================================

class TRMHasherONNX(nn.Module):
    """
    ONNX-exportable wrapper for TRMHasher.

    Exports a single-step model. The runtime calls this N_sup times.
    Each call runs t cycles internally.
    """
    def __init__(self, trm_hasher: TRMHasher):
        super().__init__()
        self.image_encoder = trm_hasher.image_encoder
        self.net = trm_hasher.net
        self.output_head = trm_hasher.output_head
        self.n_latent = trm_hasher.n_latent
        self.embed_dim = trm_hasher.embed_dim
        self.hash_dim = trm_hasher.hash_dim
        self.t = trm_hasher.t

        self.register_buffer("y_init", trm_hasher.y_init.data.clone())
        self.register_buffer("z_init", trm_hasher.z_init.data.clone())

    def get_initial_states(self, batch_size: int):
        y = self.y_init.expand(batch_size, -1)
        z = self.z_init.expand(batch_size, -1)
        return y, z

    def encode_image(self, img):
        return self.image_encoder(img)

    def latent_recursion_step(self, x, y, z):
        for _ in range(self.n_latent):
            z = self.net(x + y + z)
        y = self.net(y + z)
        return y, z

    def forward(self, img, y, z, x_cached):
        x = self.image_encoder(img)
        mask = (x_cached.abs().sum(dim=-1, keepdim=True) > 0).float()
        x = mask * x_cached + (1 - mask) * x

        for _ in range(self.t):
            y, z = self.latent_recursion_step(x, y, z)

        hash_out = torch.sigmoid(self.output_head(y))

        return x, y, z, hash_out


# ============================================================================
# Legacy Wrapper
# ============================================================================

class RecursiveHasher(nn.Module):
    """Legacy interface for backward compatibility."""
    def __init__(self, state_dim: int = 256, hash_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.hash_dim = hash_dim
        self._trm = TRMHasher(embed_dim=state_dim, hash_dim=hash_dim)

    def forward(self, img, state=None):
        if state is None:
            y, z = None, None
        else:
            y = state[:, :self.state_dim]
            z = state[:, self.state_dim:]

        y_new, z_new, hash_out = self._trm(img, y, z)
        new_state = torch.cat([y_new, z_new], dim=-1)
        return new_state, hash_out

    def inference(self, img, n_sup: int = 16):
        return self._trm.inference(img, n_sup)
