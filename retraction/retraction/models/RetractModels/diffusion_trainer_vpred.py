# ============================
# File: diffusion_trainer_vpred.py
# ============================
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm

from push_dataset_raw import create_push_data_loaders
from retract_diffusion_vpred import DiffusionSchedule, RetractDiffV, EMA


# -------------------------
# Action normalization
# -------------------------
class RunningNorm:
    """
    Compute mean/std over REAL action vectors (meters).
    Used to normalize training targets; sampling outputs must be unnormalized.
    """
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.mean = None
        self.std = None

    def fit_from_loader(self, loader, max_batches=None):
        xs = []
        with torch.no_grad():
            for i, (_, start, disp) in enumerate(loader):
                # end = start + disp
                x = torch.cat([start, disp], dim=1).float()
                xs.append(x)
                if max_batches is not None and (i + 1) >= max_batches:
                    break
        X = torch.cat(xs, dim=0)
        self.mean = X.mean(dim=0)
        self.std = X.std(dim=0).clamp_min(self.eps)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def save(self, path: str):
        d = {"mean": self.mean.cpu().numpy().tolist(), "std": self.std.cpu().numpy().tolist()}
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @staticmethod
    def load(path: str):
        rn = RunningNorm()
        with open(path, "r") as f:
            d = json.load(f)
        rn.mean = torch.tensor(d["mean"], dtype=torch.float32)
        rn.std = torch.tensor(d["std"], dtype=torch.float32)
        return rn


# -------------------------
# Sampling eval (best-of-k)
# -------------------------
@torch.no_grad()
def best_of_k_sampling_eval(
    model: RetractDiffV,
    norm: RunningNorm,
    images: torch.Tensor,
    gt_actions_real: torch.Tensor,
    k: int = 8,
    n_steps: int = 250,
    guidance_scale: float = 1.5,
    sampler: str = "ddim",
    eta: float = 0.0,
):
    model.eval()
    device = images.device
    B = images.size(0)

    samples = []
    for _ in range(k):
        x_n = model.sample_actions(
            images,
            n_steps=n_steps,
            guidance_scale=guidance_scale,
            sampler=sampler,
            eta=eta,
        )  # (B,6) normalized
        x = norm.unnormalize(x_n)  # real (meters)
        samples.append(x)

    samples = torch.stack(samples, dim=0)  # (K,B,6)

    diff = samples - gt_actions_real.unsqueeze(0)
    mse_per = (diff ** 2).mean(dim=2)         # (K,B)
    mae_per = diff.abs().mean(dim=2)          # (K,B)

    best_idx = torch.argmin(mse_per, dim=0)   # (B,)
    best = samples[best_idx, torch.arange(B, device=device)]

    best_mse = F.mse_loss(best, gt_actions_real).item()
    best_mae = F.l1_loss(best, gt_actions_real).item()

    pred_start = best[:, :3]
    pred_disp  = best[:, 3:]
    gt_start   = gt_actions_real[:, :3]
    gt_disp    = gt_actions_real[:, 3:]

    start_l2 = torch.norm(pred_start - gt_start, dim=1).mean().item()
    disp_l2  = torch.norm(pred_disp  - gt_disp,  dim=1).mean().item()

    pred_end = pred_start + pred_disp
    gt_end   = gt_start + gt_disp
    end_l2   = torch.norm(pred_end - gt_end, dim=1).mean().item()

    return {
        "best_mse": best_mse,
        "best_mae": best_mae,
        "best_start_l2": start_l2,
        "best_disp_l2": disp_l2,
        "best_end_l2": end_l2,
        "best_start_mm": start_l2 * 1000.0,
        "best_disp_mm": disp_l2 * 1000.0,
        "best_end_mm": end_l2 * 1000.0,
        "best_samples": best.detach().cpu(),
        "best_idx": best_idx.detach().cpu(),
        "all_samples": samples.detach().cpu(),
        "gt": gt_actions_real.detach().cpu(),
        "per_sample_mse": mse_per.detach().cpu(),
        "per_sample_mae": mae_per.detach().cpu(),
    }


# -------------------------
# Teacher-forced x0 recon (fast, interpretable)
# -------------------------
@torch.no_grad()
def teacher_forced_x0_mm(
    model: RetractDiffV,
    norm: RunningNorm,
    images: torch.Tensor,
    gt_actions_real: torch.Tensor,
):
    """
    One cheap eval: choose random t, diffuse gt->x_t, predict v, reconstruct x0, compute mm errors.
    """
    device = images.device
    B = images.size(0)

    gt_n = norm.normalize(gt_actions_real)  # normalized
    t = torch.randint(0, model.T, (B,), device=device, dtype=torch.long)
    eps = torch.randn_like(gt_n)
    x_t = model.q_sample(gt_n, t, eps)

    cond = model.image_encoder(images)
    v = model.predict_v(x_t, t, cond)
    x0_pred_n = model.v_to_x0(x_t, t, v)
    x0_pred = norm.unnormalize(x0_pred_n)

    pred_start = x0_pred[:, :3]
    pred_disp  = x0_pred[:, 3:]
    gt_start   = gt_actions_real[:, :3]
    gt_disp    = gt_actions_real[:, 3:]

    start_l2 = torch.norm(pred_start - gt_start, dim=1).mean().item()
    disp_l2  = torch.norm(pred_disp  - gt_disp,  dim=1).mean().item()

    pred_end = pred_start + pred_disp
    gt_end   = gt_start + gt_disp
    end_l2   = torch.norm(pred_end - gt_end, dim=1).mean().item()

    return {
        "tf_start_l2": start_l2,
        "tf_disp_l2": disp_l2,
        "tf_end_l2": end_l2,
        "tf_start_mm": start_l2 * 1000.0,
        "tf_disp_mm": disp_l2 * 1000.0,
        "tf_end_mm": end_l2 * 1000.0,
    }

# -------------------------
# Train loop
# -------------------------
def train(
    model: RetractDiffV,
    train_loader,
    val_loader,
    out_dir="retract_diffusion_vpred_weights",
    n_epochs=50,
    lr=2e-4,
    device="cuda",
    ema_decay=0.999,
    grad_clip=1.0,
    amp=True,
    guidance_drop_prob=0.1,
    best8_every=5,
    best8_k=8,
    best8_steps=250,
    best8_sampler="ddim",
    best8_eta=0.0,
    best8_guidance=1.5,
):
    os.makedirs(out_dir, exist_ok=True)

    model.cfg_drop_prob = guidance_drop_prob

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    ema = EMA(model, decay=ema_decay)

    # normalization stats
    norm = RunningNorm()
    norm.fit_from_loader(train_loader, max_batches=None)
    norm_path = os.path.join(out_dir, "action_norm.json")
    norm.save(norm_path)
    print(f"[norm] saved {norm_path}")
    print(f"[norm] mean={norm.mean.numpy()}, std={norm.std.numpy()}")

    best_val = float("inf")
    train_losses, val_losses = [], []
    tf_start_mm_hist, tf_end_mm_hist = [], []

    for epoch in range(n_epochs):
        # ---- train ----
        model.train()
        t0 = time.time()
        total = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [train]")
        for images, start, disp in pbar:
            images = images.to(device)
            start = start.to(device).float()
            disp = disp.to(device).float()

            # end = start + disp
            actions_real = torch.cat([start, disp], dim=1)
            actions_n = norm.normalize(actions_real)

            optimizer.zero_grad(set_to_none=True)

            # Use old autocast API if your torch is older; modern torch prefers torch.amp.autocast("cuda")
            with torch.cuda.amp.autocast(enabled=amp):
                loss, _, _, extra = model(images, actions_n)

            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", v=f"{extra['loss_v']:.4f}", x0=f"{extra['loss_x0']:.4f}")

        avg_train = total / len(train_loader)
        train_losses.append(avg_train)

        # ---- validate with EMA weights ----
        model.eval()
        val_total = 0.0

        # swap to EMA for eval
        current_sd = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.state_dict(), strict=True)

        # teacher-forced mm metric (cheap)
        tf_mm = None

        # best-of-k every N epochs
        do_best8 = (best8_every > 0) and ((epoch + 1) % best8_every == 0)
        best8_metrics = None

        # prepare one val batch for tf + best8
        images0, start0, disp0 = next(iter(val_loader))
        images0 = images0.to(device)
        start0 = start0.to(device).float()
        disp0 = disp0.to(device).float()
        # end0 = start0 + disp0
        gt0 = torch.cat([start0, disp0], dim=1)  # real

        # compute tf mm on up to 32 examples
        B_tf = min(32, images0.size(0))
        tf_mm = teacher_forced_x0_mm(model, norm, images0[:B_tf], gt0[:B_tf])
        tf_start_mm_hist.append(tf_mm["tf_start_mm"])
        tf_end_mm_hist.append(tf_mm["tf_end_mm"])

        with torch.no_grad():
            for images, start, disp in tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [val]"):
                images = images.to(device)
                start = start.to(device).float()
                disp = disp.to(device).float()

                # end = start + disp
                actions_real = torch.cat([start, disp], dim=1)
                actions_n = norm.normalize(actions_real)

                loss, _, _, _ = model(images, actions_n)
                val_total += loss.item()

        avg_val = val_total / len(val_loader)
        val_losses.append(avg_val)

        # best-of-k sampling eval
        if do_best8:
            B_eval = min(16, images0.size(0))
            best8_metrics = best_of_k_sampling_eval(
                model=model,
                norm=norm,
                images=images0[:B_eval],
                gt_actions_real=gt0[:B_eval],
                k=best8_k,
                n_steps=best8_steps,
                guidance_scale=best8_guidance,
                sampler=best8_sampler,
                eta=best8_eta,
            )
            print(
                f"   [best-of-{best8_k} {best8_sampler}@{best8_steps}] "
                f"MSE(real): {best8_metrics['best_mse']:.6f} | "
                f"start(mm): {best8_metrics['best_start_mm']:.3f} | "
                f"end(mm): {best8_metrics['best_end_mm']:.3f}"
            )
            snap_path = os.path.join(out_dir, f"best{best8_k}_epoch{epoch+1:03d}.npz")
            np.savez(
                snap_path,
                gt=best8_metrics["gt"].numpy(),
                best_samples=best8_metrics["best_samples"].numpy(),
                best_idx=best8_metrics["best_idx"].numpy(),
                all_samples=best8_metrics["all_samples"].numpy(),
                per_sample_mse=best8_metrics["per_sample_mse"].numpy(),
                per_sample_mae=best8_metrics["per_sample_mae"].numpy(),
            )
            print(f"   [best-of-k] saved {snap_path}")

        # restore train weights
        model.load_state_dict(current_sd, strict=True)

        dt = time.time() - t0
        print(
            f"Epoch {epoch+1}/{n_epochs} | train={avg_train:.4f} val={avg_val:.4f} | "
            f"TF start={tf_mm['tf_start_mm']:.3f}mm end={tf_mm['tf_end_mm']:.3f}mm | {dt:.1f}s"
        )

        # save best (by avg_val)
        if avg_val < best_val:
            best_val = avg_val
            best_path = os.path.join(out_dir, "retract_diffuser_vpred_best.pth")
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_val": best_val,
                },
                best_path,
            )
            print(f"[save] best -> {best_path} (val={best_val:.4f})")

        # save last
        last_path = os.path.join(out_dir, "retract_diffuser_vpred_last.pth")
        torch.save(
            {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_val": best_val,
            },
            last_path,
        )

        # plots
        plt.figure()
        plt.plot(train_losses, label="train")
        plt.plot(val_losses, label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss (v + x0)")
        plt.title("RetractDiffV Training vs Validation")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "loss_plot.png"))
        plt.close()

        plt.figure()
        plt.plot(tf_start_mm_hist, label="TF start mm")
        plt.plot(tf_end_mm_hist, label="TF end mm")
        plt.xlabel("epoch")
        plt.ylabel("mm")
        plt.title("Teacher-forced x0 recon (mm)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "tf_mm_plot.png"))
        plt.close()

        np.save(os.path.join(out_dir, "train_losses.npy"), np.array(train_losses))
        np.save(os.path.join(out_dir, "val_losses.npy"), np.array(val_losses))
        np.save(os.path.join(out_dir, "tf_start_mm.npy"), np.array(tf_start_mm_hist))
        np.save(os.path.join(out_dir, "tf_end_mm.npy"), np.array(tf_end_mm_hist))

    print(f"Done. Best val: {best_val:.4f}")
    return norm


@torch.no_grad()
def quick_sample_demo(model: RetractDiffV, norm: RunningNorm, images: torch.Tensor, out_dir: str, device="cuda"):
    model.eval()
    x0_n = model.sample_actions(
        images.to(device),
        n_steps=250,
        guidance_scale=1.5,
        sampler="ddim",
        eta=0.0,
    )
    x0 = norm.unnormalize(x0_n).cpu().numpy()

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "sampled_actions.npy"), x0)
    print(f"[sample] saved {os.path.join(out_dir, 'sampled_actions.npy')}")
    print("[sample] first:", x0[0])


def main():
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_path = "train_data/fibroid_augmented_combined"
    val_path   = "val_data/saved_demos_fib"

    train_loader, _ = create_push_data_loaders(
        data_path=train_path,
        batch_size=32,
        transform=transform,
        val_split=0.0,
        num_workers=4,
    )

    val_loader, _ = create_push_data_loaders(
        data_path=val_path,
        batch_size=32,
        transform=transform,
        val_split=0.0,
        num_workers=4,
    )

    print(f"Train size: {len(train_loader.dataset)} | Val size: {len(val_loader.dataset)}")

    schedule = DiffusionSchedule(T=1000, schedule="cosine")
    model = RetractDiffV(
        backbone="efficientnet_b0",
        action_dim=6,
        cond_dim=256,
        schedule=schedule,
        denoiser_hidden=512,
        denoiser_depth=4,
        denoiser_dropout=0.05,
        cfg_drop_prob=0.1,
        lambda_x0=0.5,       # try 0.5 then 1.0
        start_weight=2.0,    # emphasize grasp/start point
        end_weight=1.0,
        use_snr_weight=True,
    ).to(device)

    out_dir = "retract_diffusion_vpred_weights"
    norm = train(
        model,
        train_loader,
        val_loader,
        out_dir=out_dir,
        n_epochs=50,
        lr=2e-4,
        device=device,
        ema_decay=0.999,
        grad_clip=1.0,
        amp=True,
        guidance_drop_prob=0.1,
        best8_every=5,
        best8_k=8,
        best8_steps=250,
        best8_sampler="ddim",
        best8_eta=0.0,
        best8_guidance=1.5,
    )

    # load EMA for sampling demo
    ckpt = torch.load(os.path.join(out_dir, "retract_diffuser_vpred_best.pth"), map_location=device)
    model.load_state_dict(ckpt["ema"], strict=True)

    images, _, _ = next(iter(val_loader))
    quick_sample_demo(model, norm, images[:8], out_dir=out_dir, device=device)


if __name__ == "__main__":
    main()
