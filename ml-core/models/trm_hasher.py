"""
Tiny Recursion Model (TRM) for Perceptual Hashing

Implementation based on: "Less is More: Recursive Reasoning with Tiny Networks"
https://arxiv.org/abs/2510.04871

Key differences from standard distillation:
1. Single tiny 2-layer network (not separate encoder/decoder)
2. Two features: y (hash/answer) and z (latent reasoning)
3. Deep supervision: train over N_sup steps, carrying (y,z) forward
4. Network distinguishes tasks by input composition:
   - z = net(x, y, z)  # Update latent (includes input x)
   - y = net(y, z)     # Refine answer (no input x)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (as used in TRM paper)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class SwiGLU(nn.Module):
    """SwiGLU activation (as used in TRM paper)."""
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, in_dim, bias=False)
        self.w3 = nn.Linear(in_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TRMBlock(nn.Module):
    """Single TRM block: RMSNorm -> SwiGLU MLP (no attention for vision)."""
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
    
    Takes concatenated (x, y, z) or (y, z) and outputs updated feature.
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
    Lightweight CNN to convert image to embedding space.
    This replaces the input_embedding() function from TRM paper.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            # Stage 1: 224 -> 112
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            # Stage 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # Stage 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # Stage 4: 28 -> 7
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            # Global pool
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, img):
        return self.encoder(img)


class TRMHasher(nn.Module):
    """
    TRM-based Perceptual Hasher.
    
    Adapts TRM for image hashing with DINOv3 distillation:
    - x: Image embedding (from lightweight CNN encoder)
    - y: Current hash estimate (the "answer")
    - z: Latent reasoning state
    
    Training uses deep supervision: at each supervision step, we compare
    the current hash y against the teacher's embedding.
    
    Args:
        embed_dim: Dimension for x, y, z embeddings (default: 256)
        hash_dim: Final hash output dimension (default: 96)
        n_layers: Number of layers in tiny network (default: 2)
        n_latent: Number of latent recursion steps per cycle (default: 6)
        T: Number of cycles (T-1 without grad, 1 with grad) (default: 3)
    """
    def __init__(
        self,
        embed_dim: int = 256,
        hash_dim: int = 96,
        n_layers: int = 2,
        n_latent: int = 6,
        T: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hash_dim = hash_dim
        self.n_latent = n_latent
        self.T = T

        # Image encoder: image -> x embedding
        self.image_encoder = ImageEncoder(embed_dim)

        # Single tiny network (shared for both z and y updates)
        # Input: concatenated features, output: same dim as single feature
        self.net = TinyRecursiveNetwork(embed_dim, n_layers=n_layers)

        # Projection layers for concatenation
        # For z update: concat(x, y, z) -> project to embed_dim
        self.proj_xyz = nn.Linear(embed_dim * 3, embed_dim, bias=False)
        # For y update: concat(y, z) -> project to embed_dim  
        self.proj_yz = nn.Linear(embed_dim * 2, embed_dim, bias=False)

        # Output head: y -> final hash
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, hash_dim, bias=False),
        )

        # Learnable initial embeddings for y and z
        self.y_init = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
        self.z_init = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

    def latent_recursion(self, x, y, z):
        """
        One full latent recursion cycle:
        1. Update z n_latent times using (x, y, z)
        2. Update y once using (y, z) - no x!
        """
        # Latent reasoning: update z n times
        for _ in range(self.n_latent):
            # z = net(x, y, z)
            xyz = torch.cat([x, y, z], dim=-1)
            z = self.net(self.proj_xyz(xyz))

        # Refine answer: update y once (no x - this is key!)
        # y = net(y, z)
        yz = torch.cat([y, z], dim=-1)
        y = self.net(self.proj_yz(yz))

        return y, z

    def deep_recursion(self, x, y, z):
        """
        Deep recursion with T cycles:
        - T-1 cycles without gradients (to improve y, z)
        - 1 cycle with gradients (for backprop)
        """
        # T-1 cycles without gradients
        with torch.no_grad():
            for _ in range(self.T - 1):
                y, z = self.latent_recursion(x, y, z)

        # Final cycle with gradients
        y, z = self.latent_recursion(x, y, z)

        # Compute hash from y
        hash_out = self.output_head(y)
        hash_out = F.normalize(hash_out, p=2, dim=-1)

        return y.detach(), z.detach(), hash_out

    def forward(self, img, y=None, z=None):
        """
        Forward pass for one deep recursion.
        
        Args:
            img: Input image [B, 3, 224, 224]
            y: Previous y state (or None for initial)
            z: Previous z state (or None for initial)
            
        Returns:
            y: Updated y state (detached)
            z: Updated z state (detached)
            hash_out: Current hash prediction [B, hash_dim]
        """
        batch_size = img.size(0)

        # Encode image
        x = self.image_encoder(img)

        # Initialize y and z if not provided
        if y is None:
            y = self.y_init.expand(batch_size, -1)
        if z is None:
            z = self.z_init.expand(batch_size, -1)

        # Run deep recursion
        y, z, hash_out = self.deep_recursion(x, y, z)

        return y, z, hash_out

    def inference(self, img, n_sup: int = 16):
        """
        Full inference with N_sup supervision steps (as done at test time).
        
        Returns the final hash after all supervision steps.
        """
        batch_size = img.size(0)
        device = img.device

        # Encode image once (reused across all supervision steps)
        x = self.image_encoder(img)

        # Initialize
        y = self.y_init.expand(batch_size, -1).to(device)
        z = self.z_init.expand(batch_size, -1).to(device)

        # Run N_sup supervision steps
        with torch.no_grad():
            for _ in range(n_sup):
                # Deep recursion (all T cycles without grad at inference)
                for _ in range(self.T):
                    y, z = self.latent_recursion(x, y, z)

        # Final hash
        hash_out = self.output_head(y)
        hash_out = F.normalize(hash_out, p=2, dim=-1)

        return hash_out


class TRMHasherONNX(nn.Module):
    """
    ONNX-exportable wrapper for TRMHasher.
    
    Since ONNX doesn't support dynamic control flow well, we export a
    single-step model that performs one deep recursion cycle.
    The Go runtime will call this repeatedly for N_sup steps.
    
    Inputs:
        - image: [B, 3, 224, 224] (only used on first call for encoding)
        - y: [B, embed_dim] current answer state
        - z: [B, embed_dim] current latent state
        - x: [B, embed_dim] cached image encoding (from first call)
        
    Outputs:
        - x: [B, embed_dim] image encoding (pass through for caching)
        - y_new: [B, embed_dim] updated answer state
        - z_new: [B, embed_dim] updated latent state
        - hash: [B, hash_dim] current hash prediction
    """
    def __init__(self, trm_hasher: TRMHasher):
        super().__init__()
        self.image_encoder = trm_hasher.image_encoder
        self.net = trm_hasher.net
        self.proj_xyz = trm_hasher.proj_xyz
        self.proj_yz = trm_hasher.proj_yz
        self.output_head = trm_hasher.output_head
        self.n_latent = trm_hasher.n_latent
        self.T = trm_hasher.T
        
        # Register y_init and z_init as buffers (not parameters) for ONNX
        self.register_buffer('y_init', trm_hasher.y_init.data.clone())
        self.register_buffer('z_init', trm_hasher.z_init.data.clone())
    
    def encode_image(self, img):
        """Encode image to x embedding (call once per image)."""
        return self.image_encoder(img)
    
    def latent_recursion_step(self, x, y, z):
        """One full latent recursion cycle (unrolled for ONNX)."""
        # Latent reasoning: update z n_latent times
        for _ in range(self.n_latent):
            xyz = torch.cat([x, y, z], dim=-1)
            z = self.net(self.proj_xyz(xyz))
        
        # Refine answer: update y once
        yz = torch.cat([y, z], dim=-1)
        y = self.net(self.proj_yz(yz))
        
        return y, z
    
    def forward(self, img, y, z, x_cached):
        """
        Single supervision step for ONNX export.
        
        Args:
            img: Input image [B, 3, 224, 224] - used if x_cached is zeros
            y: Current y state [B, embed_dim]
            z: Current z state [B, embed_dim]
            x_cached: Cached image encoding [B, embed_dim] - zeros on first call
            
        Returns:
            x: Image encoding (for caching)
            y_new: Updated y state
            z_new: Updated z state
            hash: Current hash prediction
        """
        # Check if we need to encode image (x_cached is zeros on first call)
        # For ONNX, we always run encoder but this could be optimized in Go
        x = self.image_encoder(img)
        
        # Use cached x if provided (non-zero)
        # Note: In Go, you'd pass the same x for all steps after first
        mask = (x_cached.abs().sum(dim=-1, keepdim=True) > 0).float()
        x = mask * x_cached + (1 - mask) * x
        
        # Run T cycles of latent recursion
        for _ in range(self.T):
            y, z = self.latent_recursion_step(x, y, z)
        
        # Compute hash
        hash_out = self.output_head(y)
        hash_out = F.normalize(hash_out, p=2, dim=-1)
        
        return x, y, z, hash_out
    
    def get_initial_states(self, batch_size: int, device):
        """Get initial y and z states."""
        y = self.y_init.expand(batch_size, -1).to(device)
        z = self.z_init.expand(batch_size, -1).to(device)
        return y, z


# For backward compatibility with existing code
class RecursiveHasher(nn.Module):
    """
    Legacy RecursiveHasher interface for backward compatibility.
    
    This provides the old (img, state) -> (next_state, hash) interface
    while internally using a simplified recursive architecture.
    
    For proper TRM training, use TRMHasher directly.
    """
    def __init__(self, state_dim=128, hash_dim=96):
        super().__init__()
        self.state_dim = state_dim
        self.hash_dim = hash_dim
        
        # Simple CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        
        # GRU for recursive state update
        self.gru = nn.GRUCell(128, state_dim)
        
        # Hash output head
        self.hash_head = nn.Linear(state_dim, hash_dim)

    def forward(self, img, prev_state):
        """
        One recursive step.
        
        Args:
            img: [B, 3, 224, 224]
            prev_state: [B, state_dim]
            
        Returns:
            next_state: [B, state_dim]
            hash: [B, hash_dim]
        """
        features = self.encoder(img)
        next_state = self.gru(features, prev_state)
        hash_out = self.hash_head(next_state)
        hash_out = F.normalize(hash_out, p=2, dim=-1)
        return next_state, hash_out
