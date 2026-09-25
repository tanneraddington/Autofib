# ============================
# File: retract_diffusion_vpred.py
# ============================
import math
import copy
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# -------------------------
# Utilities: time embedding
# -------------------------
def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    timesteps: (B,) int64/float tensor
    returns: (B, dim)
    """
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=device, dtype=torch.float32) / (half - 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, 2*half)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# -------------------------
# Diffusion schedule
# -------------------------
@dataclass
class DiffusionSchedule:
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    schedule: str = "cosine"  # "linear" or "cosine"


def make_beta_schedule(T: int, beta_start: float, beta_end: float, schedule: str = "cosine") -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)

    # Cosine schedule (Nichol & Dhariwal style)
    s = 0.008
    steps = T + 1
    x = torch.linspace(0, T, steps, dtype=torch.float32)
    alphas_bar = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return torch.clamp(betas, 1e-8, 0.999)


# -------------------------
# Frozen backbone feature encoder
# -------------------------
class FrozenImageEncoder(nn.Module):
    """
    Frozen pretrained CNN -> pooled feature vector -> MLP -> cond embedding
    """

    def __init__(self, backbone: str = "resnet50", cond_dim: int = 256):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(base.children())[:-2])  # (B,2048,H/32,W/32)
            feat_channels = 2048

        elif backbone == "efficientnet_b0":
            base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.backbone = base.features  # (B,1280,?,?)
            feat_channels = 1280

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(feat_channels, 512),
            nn.SiLU(),
            nn.Linear(512, cond_dim),
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        feats = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
        return self.proj(feats)  # (B, cond_dim)


# -------------------------
# FiLM conditioning block
# -------------------------
class FiLM(nn.Module):
    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, hidden_dim * 2)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        ss = self.to_scale_shift(cond)
        scale, shift = ss.chunk(2, dim=-1)
        return h * (1 + scale) + shift


# -------------------------
# Denoiser MLP (action space)
# -------------------------
class ActionDenoiser(nn.Module):
    """
    Predicts v for a 6D action vector given noisy action x_t,
    time t, and image condition embedding (FiLM).
    """

    def __init__(
        self,
        action_dim: int = 6,
        cond_dim: int = 256,
        time_dim: int = 128,
        hidden_dim: int = 512,
        depth: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.action_dim = action_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.in_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                nn.ModuleDict(
                    dict(
                        fc1=nn.Linear(hidden_dim, hidden_dim),
                        fc2=nn.Linear(hidden_dim, hidden_dim),
                        film=FiLM(cond_dim, hidden_dim),
                        drop=nn.Dropout(dropout),
                    )
                )
            )

        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_h = self.time_mlp(t_emb)

        h = self.in_proj(x_t) + t_h

        for blk in self.blocks:
            y = F.silu(blk["fc1"](h))
            y = blk["film"](y, cond)
            y = blk["drop"](y)
            y = blk["fc2"](y)
            h = F.silu(h + y)

        v = self.out(h)
        return v


# -------------------------
# Conditional Diffusion Model (v-pred)
# -------------------------
class RetractDiffV(nn.Module):
    """
    Conditional diffusion model over 6D actions in *normalized* space.
    Conditioning: frozen image encoder -> cond embedding.

    Training:
      - v-prediction loss (MSE)
      - + x0 reconstruction loss (SmoothL1) to encourage mm-scale accuracy
      - CFG drop during training

    Sampling:
      - DDPM ancestral (default)
      - DDIM deterministic (recommended)
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        action_dim: int = 6,
        cond_dim: int = 256,
        schedule: DiffusionSchedule = DiffusionSchedule(),
        denoiser_hidden: int = 512,
        denoiser_depth: int = 4,
        denoiser_dropout: float = 0.0,
        cfg_drop_prob: float = 0.1,
        lambda_x0: float = 0.5,     # weight for x0 loss
        start_weight: float = 2.0,  # weight start coords in x0 loss
        end_weight: float = 1.0,    # weight end coords in x0 loss
        use_snr_weight: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.cond_dim = cond_dim
        self.cfg_drop_prob = cfg_drop_prob

        self.lambda_x0 = float(lambda_x0)
        self.start_weight = float(start_weight)
        self.end_weight = float(end_weight)
        self.use_snr_weight = bool(use_snr_weight)

        self.image_encoder = FrozenImageEncoder(backbone=backbone, cond_dim=cond_dim)
        self.denoiser = ActionDenoiser(
            action_dim=action_dim,
            cond_dim=cond_dim,
            time_dim=128,
            hidden_dim=denoiser_hidden,
            depth=denoiser_depth,
            dropout=denoiser_dropout,
        )

        # schedule buffers
        self.T = int(schedule.T)
        betas = make_beta_schedule(self.T, schedule.beta_start, schedule.beta_end, schedule.schedule)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)                   # (T,)
        self.register_buffer("alphas", alphas)                 # (T,)
        self.register_buffer("alphas_bar", alphas_bar)         # (T,)
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar))

        # posterior variance for DDPM sampling
        alphas_bar_prev = torch.cat(
            [torch.tensor([1.0], dtype=torch.float32, device=betas.device), alphas_bar[:-1]]
        )
        posterior_var = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
        self.register_buffer("posterior_var", posterior_var.clamp(min=1e-20))

        # null cond for CFG
        self.register_buffer("null_cond", torch.zeros(cond_dim))

    # ---- helpers ----
    def _extract(self, buf: torch.Tensor, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return buf[t].unsqueeze(1).to(dtype=like.dtype)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._extract(self.sqrt_alphas_bar, t, x0)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_bar, t, x0)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def compute_v_target(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        # v = sqrt(ab)*eps - sqrt(1-ab)*x0
        sqrt_ab = self._extract(self.sqrt_alphas_bar, t, x0)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_bar, t, x0)
        return sqrt_ab * eps - sqrt_1mab * x0

    def v_to_x0(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # x0 = sqrt(ab)*x_t - sqrt(1-ab)*v
        sqrt_ab = self._extract(self.sqrt_alphas_bar, t, x_t)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_bar, t, x_t)
        return sqrt_ab * x_t - sqrt_1mab * v

    def v_to_eps(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # eps = sqrt(ab)*v + sqrt(1-ab)*x_t
        sqrt_ab = self._extract(self.sqrt_alphas_bar, t, x_t)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_bar, t, x_t)
        return sqrt_ab * v + sqrt_1mab * x_t

    def predict_v(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.denoiser(x_t, t, cond)

    def forward(self, images: torch.Tensor, x0_actions: torch.Tensor):
        """
        x0_actions: (B,6) normalized
        Returns:
          loss (scalar),
          v_pred (B,6),
          v_target (B,6),
          extra dict (for logging)
        """
        B = x0_actions.size(0)
        device = x0_actions.device
        t = torch.randint(0, self.T, (B,), device=device, dtype=torch.long)

        eps = torch.randn_like(x0_actions)
        x_t = self.q_sample(x0_actions, t, eps)

        cond = self.image_encoder(images)

        # CFG drop during training
        if self.training and self.cfg_drop_prob > 0:
            drop = (torch.rand(B, device=device) < self.cfg_drop_prob).float().unsqueeze(1)
            null = self.null_cond.to(device).unsqueeze(0).expand(B, -1)
            cond = cond * (1.0 - drop) + null * drop

        v_target = self.compute_v_target(x0_actions, t, eps)
        v_pred = self.predict_v(x_t, t, cond)

        loss_v = F.mse_loss(v_pred, v_target, reduction="mean")

        # x0 reconstruction loss (precision)
        x0_pred = self.v_to_x0(x_t, t, v_pred)

        start_l = F.smooth_l1_loss(x0_pred[:, :3], x0_actions[:, :3], beta=0.5)
        end_l   = F.smooth_l1_loss(x0_pred[:, 3:], x0_actions[:, 3:], beta=0.5)
        loss_x0 = self.start_weight * start_l + self.end_weight * end_l

        # optional SNR weighting (stabilizes x0 loss across t)
        if self.use_snr_weight:
            a_bar = self.alphas_bar[t].to(x0_actions.dtype)
            snr = a_bar / (1.0 - a_bar + 1e-8)
            w = (snr / (snr + 1.0)).mean().clamp(0.0, 1.0)
        else:
            w = torch.tensor(1.0, device=device, dtype=x0_actions.dtype)

        loss = loss_v + (self.lambda_x0 * w) * loss_x0

        extra = {
            "loss_v": float(loss_v.detach().cpu()),
            "loss_x0": float(loss_x0.detach().cpu()),
            "w_snr": float(w.detach().cpu()),
        }
        return loss, v_pred, v_target, extra

    # -------------------------
    # Reverse process (DDPM)
    # -------------------------
    @torch.no_grad()
    def p_sample_ddpm(self, x_t: torch.Tensor, t: int, cond: torch.Tensor, guidance_scale: float = 0.0) -> torch.Tensor:
        B = x_t.size(0)
        device = x_t.device
        tt = torch.full((B,), t, device=device, dtype=torch.long)

        if guidance_scale > 0.0:
            v_c = self.predict_v(x_t, tt, cond)
            null = self.null_cond.to(device).unsqueeze(0).expand(B, -1)
            v_u = self.predict_v(x_t, tt, null)
            v = v_u + guidance_scale * (v_c - v_u)
        else:
            v = self.predict_v(x_t, tt, cond)

        eps = self.v_to_eps(x_t, tt, v)

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_bar[t]

        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps)

        if t == 0:
            return x0_pred

        var = self.posterior_var[t]
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var) * noise

    # -------------------------
    # Reverse process (DDIM)
    # -------------------------
    @torch.no_grad()
    def p_sample_ddim(
        self,
        x_t: torch.Tensor,
        t: int,
        t_prev: int,
        cond: torch.Tensor,
        guidance_scale: float = 0.0,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Deterministic (eta=0) / stochastic (eta>0) DDIM step.
        """
        B = x_t.size(0)
        device = x_t.device
        tt = torch.full((B,), t, device=device, dtype=torch.long)
        tp = torch.full((B,), t_prev, device=device, dtype=torch.long)

        if guidance_scale > 0.0:
            v_c = self.predict_v(x_t, tt, cond)
            null = self.null_cond.to(device).unsqueeze(0).expand(B, -1)
            v_u = self.predict_v(x_t, tt, null)
            v = v_u + guidance_scale * (v_c - v_u)
        else:
            v = self.predict_v(x_t, tt, cond)

        x0 = self.v_to_x0(x_t, tt, v)
        eps = self.v_to_eps(x_t, tt, v)

        a_t = self.alphas_bar[t]
        a_prev = self.alphas_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device, dtype=a_t.dtype)

        # sigma for ddim
        if eta > 0.0:
            sigma = (
                eta
                * torch.sqrt((1 - a_prev) / (1 - a_t))
                * torch.sqrt(1 - a_t / a_prev)
            )
        else:
            sigma = 0.0

        # direction pointing to x_t
        dir_xt = torch.sqrt(1 - a_prev - sigma**2) * eps
        x_prev = torch.sqrt(a_prev) * x0 + dir_xt

        if eta > 0.0 and t_prev >= 0:
            x_prev = x_prev + sigma * torch.randn_like(x_t)

        return x_prev

    @torch.no_grad()
    def sample_actions(
        self,
        images: torch.Tensor,
        n_steps: Optional[int] = 250,
        guidance_scale: float = 1.5,
        sampler: str = "ddim",   # "ddim" or "ddpm"
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Generate actions for a batch of images.
        Returns x0 in *normalized* action space (unnormalize outside).
        """
        self.eval()
        device = images.device
        B = images.size(0)

        # timesteps (descending)
        steps = self.T if (n_steps is None) else int(n_steps)
        timesteps = torch.linspace(self.T - 1, 0, steps, device=device).long().tolist()

        cond = self.image_encoder(images)

        x = torch.randn(B, self.action_dim, device=device)

        if sampler.lower() == "ddpm":
            for t in timesteps:
                x = self.p_sample_ddpm(x, t, cond, guidance_scale=guidance_scale)
            return x

        # DDIM
        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if (i + 1) < len(timesteps) else -1
            if t_prev < 0:
                # final x0 using a_prev=1.0 in p_sample_ddim
                x = self.p_sample_ddim(x, t, t_prev, cond, guidance_scale=guidance_scale, eta=eta)
                break
            x = self.p_sample_ddim(x, t, t_prev, cond, guidance_scale=guidance_scale, eta=eta)

        return x


# -------------------------
# EMA helper
# -------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            if k in msd and v.shape == msd[k].shape:
                v.copy_(v * self.decay + msd[k] * (1.0 - self.decay))

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd, strict=True)
