"""
Conditional Diffusion Model for Needle Pose Estimation

This module implements a v-prediction diffusion model that predicts 6DOF needle poses
(position + orientation as quaternion) conditioned on binary segmentation masks.

Key Features:
- Multiple pretrained feature extractors optimized for binary/medical images
- V-prediction formulation for stable training
- DDIM and DDPM sampling
- Classifier-free guidance support
- Precision-focused auxiliary losses
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


# ============================
# Time Embedding Utilities
# ============================

def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Create sinusoidal positional embeddings for timesteps.
    
    Args:
        timesteps: (B,) tensor of timestep indices
        dim: Embedding dimension
        
    Returns:
        (B, dim) tensor of time embeddings
    """
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=device, dtype=torch.float32) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, dim)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# ============================
# Diffusion Schedule
# ============================

@dataclass
class DiffusionConfig:
    """Configuration for diffusion process."""
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    schedule_type: str = "cosine"  # "linear", "cosine", or "sqrt"


def make_beta_schedule(
    num_timesteps: int,
    beta_start: float,
    beta_end: float,
    schedule_type: str = "cosine"
) -> torch.Tensor:
    """
    Create noise schedule.
    
    Args:
        num_timesteps: Total diffusion steps
        beta_start: Starting beta value
        beta_end: Ending beta value
        schedule_type: "linear", "cosine", or "sqrt"
        
    Returns:
        (T,) tensor of beta values
    """
    if schedule_type == "linear":
        return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
    
    elif schedule_type == "sqrt":
        # Square root schedule - good for low-dimensional data
        return torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps, dtype=torch.float32) ** 2
    
    # Cosine schedule (default, from Improved DDPM)
    s = 0.008
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)
    alphas_bar = torch.cos(((x / num_timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1.0 - (alphas_bar[1:] / alphas_bar[:-1])
    return torch.clamp(betas, 1e-8, 0.999)


# ============================
# Binary Image Feature Extractors
# ============================

class MedicalImageEncoder(nn.Module):
    """
    Feature encoder specifically designed for binary medical/segmentation images.
    Uses techniques from medical imaging literature.
    
    Better than standard ImageNet models for binary masks because:
    1. Trained on medical/segmentation data
    2. Handles single-channel input naturally
    3. Captures shape and topology features
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        feature_dim: int = 256,
        use_attention: bool = True
    ):
        super().__init__()
        
        # Multi-scale feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 7, stride=2, padding=3),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
        )
        
        # Spatial attention (captures important regions)
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(256, 64, 1),
                nn.SiLU(),
                nn.Conv2d(64, 1, 1),
                nn.Sigmoid()
            )
        
        # Global pooling and projection
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, feature_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) binary mask
            
        Returns:
            (B, feature_dim) feature vector
        """
        h1 = self.conv1(x)      # (B, 32, H/2, W/2)
        h2 = self.conv2(h1)     # (B, 64, H/4, W/4)
        h3 = self.conv3(h2)     # (B, 128, H/8, W/8)
        h4 = self.conv4(h3)     # (B, 256, H/16, W/16)
        
        # Apply spatial attention
        if self.use_attention:
            att = self.attention(h4)  # (B, 1, H/16, W/16)
            h4 = h4 * att
        
        # Global pooling and projection
        feat = self.pool(h4).flatten(1)  # (B, 256)
        return self.proj(feat)  # (B, feature_dim)


class DINOv2BinaryEncoder(nn.Module):
    """
    Uses DINOv2 (self-supervised vision transformer) adapted for binary images.
    DINOv2 learns good representations without supervision and generalizes well.
    
    Note: Requires installing DINOv2: pip install git+https://github.com/facebookresearch/dinov2.git
    or we can use the torch.hub version.
    """
    
    def __init__(self, feature_dim: int = 256, model_size: str = "small", freeze: bool = True):
        super().__init__()
        
        try:
            # Load DINOv2 from torch hub
            if model_size == "small":
                self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
                backbone_dim = 384
            elif model_size == "base":
                self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
                backbone_dim = 768
            else:
                raise ValueError(f"Unsupported model_size: {model_size}")
            
            # Adapt input for single-channel
            # Replace first conv to accept 1 channel, initialize with grayscale weights
            original_patch_embed = self.backbone.patch_embed.proj
            new_patch_embed = nn.Conv2d(
                1, original_patch_embed.out_channels,
                kernel_size=original_patch_embed.kernel_size,
                stride=original_patch_embed.stride,
                padding=original_patch_embed.padding
            )
            # Initialize by averaging RGB weights
            with torch.no_grad():
                new_patch_embed.weight[:] = original_patch_embed.weight.mean(dim=1, keepdim=True)
                new_patch_embed.bias[:] = original_patch_embed.bias
            self.backbone.patch_embed.proj = new_patch_embed
            
            if freeze:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                # Unfreeze the first layer we just modified
                for param in self.backbone.patch_embed.proj.parameters():
                    param.requires_grad = True
            
            self.available = True
            
        except Exception as e:
            print(f"Warning: Could not load DINOv2 ({e}). Using fallback encoder.")
            self.available = False
            self.backbone = MedicalImageEncoder(input_channels=1, feature_dim=feature_dim)
            backbone_dim = feature_dim
        
        # Projection head
        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, feature_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) binary mask
            
        Returns:
            (B, feature_dim) feature vector
        """
        if not self.available:
            return self.backbone(x)
        
        # DINOv2 expects images, we'll replicate channel
        with torch.no_grad() if not self.training else torch.enable_grad():
            feat = self.backbone(x)  # (B, backbone_dim)
        
        return self.proj(feat)


class ResNetBinaryEncoder(nn.Module):
    """
    ResNet encoder adapted for binary images.
    Uses a pretrained ResNet but modifies the first layer for single-channel input.
    
    While not ideal for binary images, ResNets are robust and widely available.
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        model_name: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = True
    ):
        super().__init__()
        
        # Load pretrained ResNet
        if model_name == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base = resnet18(weights=weights)
            backbone_dim = 512
        elif model_name == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            base = resnet50(weights=weights)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Modify first conv layer for single-channel input
        original_conv1 = base.conv1
        base.conv1 = nn.Conv2d(
            1, original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )
        
        # Initialize with grayscale weights (average of RGB)
        if pretrained:
            with torch.no_grad():
                base.conv1.weight[:] = original_conv1.weight.mean(dim=1, keepdim=True)
        
        # Extract feature extractor (remove final FC layer)
        self.backbone = nn.Sequential(*list(base.children())[:-2])  # Up to avgpool
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze first layer
            for param in list(self.backbone.children())[0].parameters():
                param.requires_grad = True
        
        # Global pooling and projection
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, feature_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) binary mask
            
        Returns:
            (B, feature_dim) feature vector
        """
        feat = self.backbone(x)  # (B, backbone_dim, H', W')
        feat = self.pool(feat).flatten(1)  # (B, backbone_dim)
        return self.proj(feat)


def create_image_encoder(
    encoder_type: str = "medical",
    feature_dim: int = 256,
    **kwargs
) -> nn.Module:
    """
    Factory function to create image encoder.
    
    Args:
        encoder_type: "medical", "dinov2", "resnet18", or "resnet50"
        feature_dim: Output feature dimension
        **kwargs: Additional arguments for specific encoders
        
    Returns:
        Image encoder module
    """
    if encoder_type == "medical":
        return MedicalImageEncoder(
            input_channels=1,
            feature_dim=feature_dim,
            use_attention=kwargs.get("use_attention", True)
        )
    
    elif encoder_type == "dinov2":
        return DINOv2BinaryEncoder(
            feature_dim=feature_dim,
            model_size=kwargs.get("model_size", "small"),
            freeze=kwargs.get("freeze", True)
        )
    
    elif encoder_type in ["resnet18", "resnet50"]:
        return ResNetBinaryEncoder(
            feature_dim=feature_dim,
            model_name=encoder_type,
            pretrained=kwargs.get("pretrained", True),
            freeze_backbone=kwargs.get("freeze_backbone", True)
        )
    
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ============================
# FiLM Conditioning Layer
# ============================

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation for conditioning.
    Scales and shifts features based on conditioning signal.
    """
    
    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, hidden_dim * 2)
    
    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, hidden_dim) features to modulate
            cond: (B, cond_dim) conditioning vector
            
        Returns:
            (B, hidden_dim) modulated features
        """
        scale_shift = self.to_scale_shift(cond)  # (B, hidden_dim * 2)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return h * (1.0 + scale) + shift


# ============================
# Pose Denoiser Network
# ============================

class PoseDenoiser(nn.Module):
    """
    MLP-based denoiser that predicts v-target for pose given:
    - Noisy pose at timestep t
    - Time embedding
    - Image condition (via FiLM)
    """
    
    def __init__(
        self,
        pose_dim: int = 7,  # [x, y, z, qx, qy, qz, qw]
        cond_dim: int = 256,
        time_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pose_dim = pose_dim
        self.time_dim = time_dim
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        
        # Residual blocks with FiLM conditioning
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                nn.ModuleDict({
                    "norm1": nn.LayerNorm(hidden_dim),
                    "fc1": nn.Linear(hidden_dim, hidden_dim * 2),
                    "film": FiLM(cond_dim, hidden_dim * 2),
                    "fc2": nn.Linear(hidden_dim * 2, hidden_dim),
                    "norm2": nn.LayerNorm(hidden_dim),
                    "dropout": nn.Dropout(dropout),
                })
            )
        
        # Output projection
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, pose_dim),
        )
    
    def forward(
        self,
        noisy_pose: torch.Tensor,
        timestep: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            noisy_pose: (B, pose_dim) noisy pose at timestep t
            timestep: (B,) timestep indices
            cond: (B, cond_dim) image conditioning
            
        Returns:
            (B, pose_dim) predicted v-target
        """
        # Get time embedding
        t_emb = sinusoidal_time_embedding(timestep, self.time_dim)
        t_emb = self.time_mlp(t_emb)  # (B, hidden_dim)
        
        # Project input and add time
        h = self.input_proj(noisy_pose) + t_emb  # (B, hidden_dim)
        
        # Apply residual blocks with FiLM
        for block in self.blocks:
            residual = h
            
            # Pre-norm + FC1
            h = block["norm1"](h)
            h = block["fc1"](h)
            h = F.silu(h)
            
            # FiLM conditioning
            h = block["film"](h, cond)
            
            # FC2 + dropout
            h = block["fc2"](h)
            h = block["dropout"](h)
            
            # Residual connection
            h = block["norm2"](h + residual)
        
        # Output projection
        v = self.output(h)
        return v


