"""
SDE Diffusion Model for 10x3 Displacement Prediction (VP-SDE / DDPM-style).

Conditioning: ONLY
- Frozen EfficientNet-B0 image features
- Encoded start point (x,y,z)

Predicts:
- action/displacement vector of dim 30 (10 waypoints x 3)

Supports:
- v-pred (recommended) or eps-pred
- CFG training via null conditioning dropout
- Reverse-time sampling:
  - Euler–Maruyama (stochastic SDE)
  - Reverse ODE (deterministic)

Utilities expected from diffusion_utils.py:
- sinusoidal_time_embedding(t, dim)
- FiLM(cond_dim, hidden_dim)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from .diffusion_utils import (
    sinusoidal_time_embedding,
    FiLM,
)


# ============================================================
# Noise schedule (continuous-time VP schedule)
# ============================================================
class VPScheduler:
    """
    Continuous-time VP schedule on t in [0,1].

    log_alpha(t) = -0.25 t^2 (beta_max - beta_min) - 0.5 t beta_min
    alpha(t) = exp(log_alpha(t))
    sigma(t) = sqrt(1 - alpha(t)^2)

    beta(t) = beta_min + t (beta_max - beta_min)
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        log_alpha = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        return torch.exp(log_alpha)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        a = self.alpha(t)
        return torch.sqrt(torch.clamp(1.0 - a * a, min=1e-12))

    def alpha_sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.alpha(t)
        s = torch.sqrt(torch.clamp(1.0 - a * a, min=1e-12))
        return a, s


# ============================================================
# Conditioning encoder: image + start point ONLY
# ============================================================
class StartPointEncoder(nn.Module):
    def __init__(self, start_dim: int = 3, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(start_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(self, start_points: torch.Tensor) -> torch.Tensor:
        return self.net(start_points)


class ConditionEncoder(nn.Module):
    """
    image_feat + start_feat -> fused cond
    with explicit null cond for CFG.
    """

    def __init__(self, image_feat_dim: int, start_feat_dim: int, fused_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.fused_dim = fused_dim

        self.fuse = nn.Sequential(
            nn.Linear(image_feat_dim + start_feat_dim, fused_dim * 2),
            nn.LayerNorm(fused_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim * 2, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.SiLU(),
            nn.Linear(fused_dim, fused_dim),
        )

        self.register_buffer("null_cond", torch.zeros(fused_dim))

    def forward(self, image_feats: torch.Tensor, start_feats: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image_feats, start_feats], dim=1)
        return self.fuse(x)

    def get_null(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.null_cond.unsqueeze(0).expand(batch_size, -1).to(device)


# ============================================================
# Denoiser: predicts v or eps in action space (dim=30)
# ============================================================
class DisplacementDenoiserFiLM(nn.Module):
    def __init__(
        self,
        action_dim: int = 30,
        cond_dim: int = 512,
        time_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.in_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                nn.ModuleDict({
                    "norm1": nn.LayerNorm(hidden_dim),
                    "fc1": nn.Linear(hidden_dim, hidden_dim * 2),
                    "film": FiLM(cond_dim, hidden_dim * 2),
                    "fc2": nn.Linear(hidden_dim * 2, hidden_dim),
                    "drop": nn.Dropout(dropout),
                    "norm2": nn.LayerNorm(hidden_dim),
                })
            )

        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        h = self.in_proj(x_t) + t_emb

        for b in self.blocks:
            res = h
            h = b["norm1"](h)
            h = F.silu(b["fc1"](h))
            h = b["film"](h, cond)
            h = b["fc2"](h)
            h = b["drop"](h)
            h = b["norm2"](h + res)

        return self.out(h)


# ============================================================
# Main model
# ============================================================
@dataclass
class DiffusionSDEConfig:
    action_dim: int = 30          # 10 waypoints x 3
    start_dim: int = 3

    # EfficientNet config
    freeze_image_encoder: bool = True
    imagenet_weights: bool = True
    image_feat_dim: int = 256     # projected dim after EfficientNet

    # start-point encoder
    start_feat_dim: int = 256

    # fused conditioning
    fused_cond_dim: int = 512

    # denoiser
    denoiser_hidden: int = 512
    denoiser_layers: int = 8
    denoiser_dropout: float = 0.1

    # VP schedule
    beta_min: float = 0.1
    beta_max: float = 20.0

    # CFG
    cfg_drop_prob: float = 0.1

    # prediction type: "v" or "eps"
    pred_type: str = "v"

    # ---------- NEW: action normalization ----------
    use_action_norm: bool = True

    # ---------- NEW: x0 recon loss ----------
    lambda_x0: float = 0.5          # 0 disables x0 loss
    x0_beta: float = 0.5            # SmoothL1 beta in normalized space
    use_snr_weight: bool = True     # stabilize x0 term across t


class DisplacementDiffusionSDE(nn.Module):
    """
    VP-SDE diffusion predicting displacement vectors.

    Conditioning: EfficientNet-B0 (frozen) + start point encoder.

    Important: Training is done in (optionally) normalized action space for stability.
               predict() returns METERS (unnormalized), so downstream code stays clean.
    """

    def __init__(self, cfg: DiffusionSDEConfig):
        super().__init__()
        self.cfg = cfg
        self.scheduler = VPScheduler(beta_min=cfg.beta_min, beta_max=cfg.beta_max)

        # -------- NEW: action norm buffers (stored in model for inference) --------
        self.register_buffer("action_mean", torch.zeros(cfg.action_dim))
        self.register_buffer("action_std", torch.ones(cfg.action_dim))
        self.register_buffer("has_action_norm", torch.tensor(0, dtype=torch.uint8))

        # -------- Image encoder: EfficientNet-B0 --------
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if cfg.imagenet_weights else None
        self.image_encoder = efficientnet_b0(weights=weights)
        self.image_encoder.classifier = nn.Identity()  # output 1280

        if cfg.freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            self.image_encoder.eval()

        self.image_raw_dim = 1280
        self.image_proj = nn.Sequential(
            nn.Linear(self.image_raw_dim, cfg.image_feat_dim),
            nn.LayerNorm(cfg.image_feat_dim),
            nn.SiLU(),
        )

        # -------- Start point encoder --------
        self.start_encoder = StartPointEncoder(
            start_dim=cfg.start_dim, out_dim=cfg.start_feat_dim, dropout=cfg.denoiser_dropout
        )

        # -------- Condition fusion (+ null) --------
        self.cond_encoder = ConditionEncoder(
            image_feat_dim=cfg.image_feat_dim,
            start_feat_dim=cfg.start_feat_dim,
            fused_dim=cfg.fused_cond_dim,
            dropout=cfg.denoiser_dropout,
        )

        # -------- Denoiser --------
        self.denoiser = DisplacementDenoiserFiLM(
            action_dim=cfg.action_dim,
            cond_dim=cfg.fused_cond_dim,
            time_dim=128,
            hidden_dim=cfg.denoiser_hidden,
            num_layers=cfg.denoiser_layers,
            dropout=cfg.denoiser_dropout,
        )

        if cfg.pred_type not in ("v", "eps"):
            raise ValueError("pred_type must be 'v' or 'eps'")

    # ---------- NEW: set action norm (called by trainer after fitting) ----------
    @torch.no_grad()
    def set_action_norm(self, mean: torch.Tensor, std: torch.Tensor):
        assert mean.shape[-1] == self.cfg.action_dim
        assert std.shape[-1] == self.cfg.action_dim
        self.action_mean.copy_(mean.to(self.action_mean.device))
        self.action_std.copy_(std.to(self.action_std.device).clamp_min(1e-8))
        self.has_action_norm.fill_(1)

    def _maybe_norm(self, x_meters: torch.Tensor) -> torch.Tensor:
        if (not self.cfg.use_action_norm) or (self.has_action_norm.item() == 0):
            return x_meters
        return (x_meters - self.action_mean.to(x_meters.device)) / self.action_std.to(x_meters.device)

    def _maybe_unnorm(self, x_norm: torch.Tensor) -> torch.Tensor:
        if (not self.cfg.use_action_norm) or (self.has_action_norm.item() == 0):
            return x_norm
        return x_norm * self.action_std.to(x_norm.device) + self.action_mean.to(x_norm.device)

    # ---------- conditioning ----------
    def encode_condition(self, images: torch.Tensor, start_points: torch.Tensor, use_null: bool = False) -> torch.Tensor:
        if use_null:
            return self.cond_encoder.get_null(images.size(0), images.device)

        # EfficientNet features
        if self.cfg.freeze_image_encoder:
            with torch.no_grad():
                img_raw = self.image_encoder(images)  # [B,1280]
        else:
            img_raw = self.image_encoder(images)

        img_feat = self.image_proj(img_raw)               # [B,image_feat_dim]
        sp_feat = self.start_encoder(start_points)        # [B,start_feat_dim]
        return self.cond_encoder(img_feat, sp_feat)       # [B,fused_cond_dim]

    # ---------- forward diffusion ----------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if eps is None:
            eps = torch.randn_like(x0)
        a, s = self.scheduler.alpha_sigma(t)
        a = a.view(-1, 1)
        s = s.view(-1, 1)
        x_t = a * x0 + s * eps
        return x_t, eps

    # ---------- v/eps targets ----------
    def target_v(self, x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # v = alpha*eps - sigma*x0
        a, s = self.scheduler.alpha_sigma(t)
        a = a.view(-1, 1)
        s = s.view(-1, 1)
        return a * eps - s * x0

    def v_to_eps(self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # eps = sigma*x_t + alpha*v
        a, s = self.scheduler.alpha_sigma(t)
        a = a.view(-1, 1)
        s = s.view(-1, 1)
        return s * x_t + a * v

    def v_to_x0(self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # from:
        #   x_t = a x0 + s eps
        #   v   = a eps - s x0
        # => x0 = a x_t - s v
        a, s = self.scheduler.alpha_sigma(t)
        a = a.view(-1, 1)
        s = s.view(-1, 1)
        return a * x_t - s * v

    def eps_to_score(self, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # score = -(1/sigma) * eps
        _, s = self.scheduler.alpha_sigma(t)
        s = s.view(-1, 1)
        return -eps / (s + 1e-12)

    def snr_weight(self, t: torch.Tensor) -> torch.Tensor:
        # SNR = (a/s)^2 ; we use snr/(snr+1) in [0,1]
        a, s = self.scheduler.alpha_sigma(t)
        snr = (a / (s + 1e-12)) ** 2
        w = snr / (snr + 1.0)
        return w  # [B]

    # ---------- training ----------
    def forward(
        self,
        images: torch.Tensor,
        start_points: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Training forward pass with CFG dropout.

        actions are in METERS.
        Internally we diffuse in normalized space (if action_norm is set).
        """
        B = actions.size(0)
        device = actions.device

        x0 = self._maybe_norm(actions)               # normalized if enabled
        t = torch.rand(B, device=device)            # continuous time in [0,1]
        x_t, eps = self.q_sample(x0, t)

        # CFG dropout
        if self.training and self.cfg.cfg_drop_prob > 0:
            drop = (torch.rand(B, device=device) < self.cfg.cfg_drop_prob)
            cond_y = self.encode_condition(images, start_points, use_null=False)
            cond_0 = self.encode_condition(images, start_points, use_null=True)
            dropf = drop.float().unsqueeze(1)
            cond = cond_y * (1.0 - dropf) + cond_0 * dropf
        else:
            cond = self.encode_condition(images, start_points, use_null=False)

        pred = self.denoiser(x_t, t, cond)

        # primary diffusion loss (v or eps)
        target = eps if self.cfg.pred_type == "eps" else self.target_v(x0, eps, t)
        loss_main = F.mse_loss(pred, target)

        # x0 recon loss (like your old model vibe)
        loss_x0 = torch.tensor(0.0, device=device)
        w_snr = torch.tensor(0.0, device=device)

        if self.cfg.lambda_x0 > 0.0:
            if self.cfg.pred_type == "eps":
                eps_hat = pred
                a, s = self.scheduler.alpha_sigma(t)
                a = a.view(-1, 1)
                s = s.view(-1, 1)
                x0_pred = (x_t - s * eps_hat) / (a + 1e-12)
            else:
                x0_pred = self.v_to_x0(x_t, pred, t)

            loss_x0 = F.smooth_l1_loss(x0_pred, x0, beta=self.cfg.x0_beta)

            if self.cfg.use_snr_weight:
                w_snr = self.snr_weight(t).mean().clamp(0.0, 1.0)
            else:
                w_snr = torch.tensor(1.0, device=device)

            loss = loss_main + (self.cfg.lambda_x0 * w_snr) * loss_x0
        else:
            loss = loss_main

        return loss, {
            "loss": float(loss.detach().cpu()),
            "loss_main": float(loss_main.detach().cpu()),
            "loss_x0": float(loss_x0.detach().cpu()),
            "w_snr": float(w_snr.detach().cpu()),
        }

    # ---------- sampling ----------
    @torch.no_grad()
    def sample_sde_euler_maruyama(
        self,
        images: torch.Tensor,
        start_points: torch.Tensor,
        num_steps: int = 250,
        guidance_scale: float = 0.0,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Reverse-time VP-SDE sampling (stochastic) with Euler–Maruyama.

        dx = [ -0.5*beta(t) x - beta(t) * score(x,t|y) ] dt + sqrt(beta(t)) dW
        integrate from t=1 -> 0 with dt=-1/N.

        NOTE:
          - guidance_scale=0.0 preserves multimodality best.
          - if you use CFG, prefer modest guidance (0..1) for multi-mode actions.
        """
        self.eval()
        device = images.device
        B = images.size(0)

        # CFG conditioning
        if guidance_scale > 0.0:
            cond_y = self.encode_condition(images, start_points, use_null=False)
            cond_0 = self.encode_condition(images, start_points, use_null=True)
        else:
            cond_y = self.encode_condition(images, start_points, use_null=False)
            cond_0 = None

        # sample in (normalized) diffusion space
        x = torch.randn(B, self.cfg.action_dim, device=device)

        dt = -1.0 / float(num_steps)
        traj = [x.clone()] if return_trajectory else None

        for i in range(num_steps):
            t = torch.full((B,), 1.0 + (i * dt), device=device).clamp(0.0, 1.0)
            beta_t = self.scheduler.beta(t).view(-1, 1)

            if guidance_scale > 0.0:
                pred_c = self.denoiser(x, t, cond_y)
                pred_u = self.denoiser(x, t, cond_0)
                if self.cfg.pred_type == "eps":
                    eps_c, eps_u = pred_c, pred_u
                else:
                    eps_c = self.v_to_eps(x, pred_c, t)
                    eps_u = self.v_to_eps(x, pred_u, t)
                eps_hat = eps_u + guidance_scale * (eps_c - eps_u)
            else:
                pred = self.denoiser(x, t, cond_y)
                eps_hat = pred if self.cfg.pred_type == "eps" else self.v_to_eps(x, pred, t)

            score = self.eps_to_score(eps_hat, t)
            drift = (-0.5 * beta_t * x) - (beta_t * score)
            g = torch.sqrt(torch.clamp(beta_t, min=1e-12))

            z = torch.randn_like(x)
            x = x + drift * dt + g * math.sqrt(-dt) * z

            if return_trajectory:
                traj.append(x.clone())

        return torch.stack(traj, dim=1) if return_trajectory else x

    @torch.no_grad()
    def sample_ode(
        self,
        images: torch.Tensor,
        start_points: torch.Tensor,
        num_steps: int = 250,
        guidance_scale: float = 0.0,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Deterministic reverse ODE (no noise term). Useful for debugging.
        """
        self.eval()
        device = images.device
        B = images.size(0)

        if guidance_scale > 0.0:
            cond_y = self.encode_condition(images, start_points, use_null=False)
            cond_0 = self.encode_condition(images, start_points, use_null=True)
        else:
            cond_y = self.encode_condition(images, start_points, use_null=False)
            cond_0 = None

        x = torch.randn(B, self.cfg.action_dim, device=device)

        dt = -1.0 / float(num_steps)
        traj = [x.clone()] if return_trajectory else None

        for i in range(num_steps):
            t = torch.full((B,), 1.0 + (i * dt), device=device).clamp(0.0, 1.0)
            beta_t = self.scheduler.beta(t).view(-1, 1)

            if guidance_scale > 0.0:
                pred_c = self.denoiser(x, t, cond_y)
                pred_u = self.denoiser(x, t, cond_0)
                if self.cfg.pred_type == "eps":
                    eps_c, eps_u = pred_c, pred_u
                else:
                    eps_c = self.v_to_eps(x, pred_c, t)
                    eps_u = self.v_to_eps(x, pred_u, t)
                eps_hat = eps_u + guidance_scale * (eps_c - eps_u)
            else:
                pred = self.denoiser(x, t, cond_y)
                eps_hat = pred if self.cfg.pred_type == "eps" else self.v_to_eps(x, pred, t)

            score = self.eps_to_score(eps_hat, t)
            drift = (-0.5 * beta_t * x) - (beta_t * score)
            x = x + drift * dt

            if return_trajectory:
                traj.append(x.clone())

        return torch.stack(traj, dim=1) if return_trajectory else x

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        start_points: torch.Tensor,
        num_steps: int = 250,
        guidance_scale: float = 0.0,
        sampler: str = "sde",  # "sde" or "ode"
    ) -> torch.Tensor:
        """
        Returns predicted actions in METERS (unnormalized).
        """
        if sampler == "ode":
            x_n = self.sample_ode(images, start_points, num_steps, guidance_scale, return_trajectory=False)
        else:
            x_n = self.sample_sde_euler_maruyama(images, start_points, num_steps, guidance_scale, return_trajectory=False)

        return self._maybe_unnorm(x_n)


if __name__ == "__main__":
    torch.manual_seed(0)

    cfg = DiffusionSDEConfig(action_dim=30, pred_type="v", cfg_drop_prob=0.1, lambda_x0=0.5, use_action_norm=True)
    model = DisplacementDiffusionSDE(cfg)

    B = 4
    images = torch.randn(B, 3, 224, 224)
    start = torch.randn(B, 3)
    actions = torch.randn(B, 30)

    model.train()
    loss, metrics = model(images, start, actions)
    print("loss:", loss.item(), metrics)

    model.eval()
    samp = model.predict(images, start, num_steps=25, guidance_scale=0.0, sampler="sde")
    print("sample:", samp.shape)
