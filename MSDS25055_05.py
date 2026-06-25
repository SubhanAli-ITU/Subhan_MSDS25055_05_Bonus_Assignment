
import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.utils as vutils

# ─────────────────────────────────────────────────────────────
# 1. DATA LOADER
# ─────────────────────────────────────────────────────────────

class AnimalDataset(Dataset):
    def __init__(self, dataset_path: str, image_size: int = 64,
                 num_classes: int = 5, images_per_class: int = 20):
        super().__init__()
        self.image_size = image_size
        self.paths = []

        root = Path(dataset_path)
        class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])[:num_classes]

        for cls_dir in class_dirs:
            img_files = sorted(
                [f for f in cls_dir.iterdir()
                 if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')]
            )[:images_per_class]
            self.paths.extend(img_files)

        if len(self.paths) == 0:
            raise RuntimeError(
                f"No images found under '{dataset_path}'. "
                "Check that the path contains class sub-folders with images."
            )

        # Standard augmentation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),                        # [0, 1]
            transforms.Normalize([0.5, 0.5, 0.5],        # → [-1, 1]
                                  [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


# ─────────────────────────────────────────────────────────────
# 2. NOISE SCHEDULE & FORWARD PROCESS
# ─────────────────────────────────────────────────────────────

class NoiseScheduler:
    def __init__(self, T: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
                 device: str = "cpu"):
        self.T = T
        self.device = device

        # β_t  – linearly spaced
        betas = torch.linspace(beta_start, beta_end, T, device=device)   # (T,)

        # Pre-compute cumulative products
        alphas        = 1.0 - betas                                        # α_t
        alphas_cumprod = torch.cumprod(alphas, dim=0)                     # ᾱ_t

        # Store as buffers used in forward / reverse steps
        self.betas            = betas
        self.alphas           = alphas
        self.alphas_cumprod   = alphas_cumprod
        self.sqrt_alphas_cumprod        = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()

    # ── Forward process q(x_t | x_0) ──────────────────────────
    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> tuple:
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        epsilon = torch.randn_like(x0)                      # ε ~ N(0, I)
        xt = sqrt_alpha_bar * x0 + sqrt_one_minus * epsilon  # reparameterisation trick
        return xt, epsilon


# ─────────────────────────────────────────────────────────────
# 3. U-NET DENOISING MODEL
# ─────────────────────────────────────────────────────────────

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]          # (B, half)
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),   # GN instead of BN → works with small batch sizes
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        # Projection for skip connection when channels differ
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)  # add time info
        h = self.block2(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, num_heads=4, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, -1).transpose(1, 2)   # (B, HW, C)
        h, _ = self.attn(h, h, h)
        h = h.transpose(1, 2).view(B, C, H, W)
        return x + h


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, base_ch: int = 64,
                 ch_mults: tuple = (1, 2, 4), time_emb_dim: int = 256):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbedding(base_ch),
            nn.Linear(base_ch, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        channels = [base_ch * m for m in ch_mults]   # e.g. [64, 128, 256]

        # ── Input projection ──────────────────────────────────
        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # ── Encoder ───────────────────────────────────────────
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        prev_ch = base_ch
        for ch in channels:
            self.down_blocks.append(ResBlock(prev_ch, ch, time_emb_dim))
            self.down_samples.append(
                nn.Conv2d(ch, ch, 4, stride=2, padding=1)  # stride-2 halves spatial dims
            )
            prev_ch = ch

        # ── Bottleneck ────────────────────────────────────────
        self.mid_block1 = ResBlock(prev_ch, prev_ch, time_emb_dim)
        self.mid_attn   = AttentionBlock(prev_ch)
        self.mid_block2 = ResBlock(prev_ch, prev_ch, time_emb_dim)

        # ── Decoder ───────────────────────────────────────────
        self.up_blocks  = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for ch in reversed(channels):
            # After upsampling + skip concat, input channels = prev_ch + ch
            self.up_samples.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(prev_ch, ch, 3, padding=1)
                )
            )
            self.up_blocks.append(ResBlock(ch * 2, ch, time_emb_dim))
            prev_ch = ch

        # ── Output ────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(8, prev_ch)
        self.out_act  = nn.SiLU()
        self.out_conv = nn.Conv2d(prev_ch, in_ch, 1)   # 1×1 conv → noise prediction

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)          # (B, time_emb_dim)

        h = self.init_conv(x)
        skips = []

        # Encode
        for res, down in zip(self.down_blocks, self.down_samples):
            h = res(h, t_emb)
            skips.append(h)
            h = down(h)

        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Decode
        for up_sample, res in zip(self.up_samples, self.up_blocks):
            h = up_sample(h)
            skip = skips.pop()
            # Handle spatial size mismatch (can occur at borders)
            if h.shape != skip.shape:
                h = F.interpolate(h, size=skip.shape[-2:],
                                  mode='bilinear', align_corners=False)
            h = torch.cat([h, skip], dim=1)   # skip connection
            h = res(h, t_emb)

        return self.out_conv(self.out_act(self.out_norm(h)))


# ─────────────────────────────────────────────────────────────
# 4. CUSTOM LOSS FUNCTION
# ─────────────────────────────────────────────────────────────

class HybridDiffusionLoss(nn.Module):
    def __init__(self, lambda_l2: float = 0.8):
        super().__init__()
        self.lambda_l2 = lambda_l2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l2 = F.mse_loss(pred, target)
        l1 = F.l1_loss(pred, target)
        return self.lambda_l2 * l2 + (1.0 - self.lambda_l2) * l1


# ─────────────────────────────────────────────────────────────
# 5. REVERSE (SAMPLING) PROCESS
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def sample(model: nn.Module, scheduler: NoiseScheduler,
           n_samples: int = 4, image_size: int = 64,
           device: str = "cpu") -> torch.Tensor:
    model.eval()
    x = torch.randn(n_samples, 3, image_size, image_size, device=device)

    for t_idx in reversed(range(scheduler.T)):
        t_tensor = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)

        eps_pred = model(x, t_tensor)

        beta_t        = scheduler.betas[t_idx]
        alpha_t       = scheduler.alphas[t_idx]
        alpha_bar_t   = scheduler.alphas_cumprod[t_idx]
        sqrt_one_minus = scheduler.sqrt_one_minus_alphas_cumprod[t_idx]

        # Posterior mean
        coef = beta_t / sqrt_one_minus
        mean = (1.0 / alpha_t.sqrt()) * (x - coef * eps_pred)

        if t_idx > 0:
            # Posterior variance  σ_t^2 = β_t  (simplified as in original DDPM)
            noise = torch.randn_like(x)
            x = mean + beta_t.sqrt() * noise
        else:
            x = mean   # no noise at final step

    return x.clamp(-1.0, 1.0)


# ─────────────────────────────────────────────────────────────
# 6. VISUALISATION HELPERS
# ─────────────────────────────────────────────────────────────

def denorm(t: torch.Tensor) -> torch.Tensor:
    return (t * 0.5 + 0.5).clamp(0, 1)


def save_noise_progression(x0: torch.Tensor, scheduler: NoiseScheduler,
                            save_path: str, n_steps: int = 10):
    steps = torch.linspace(0, scheduler.T - 1, n_steps, dtype=torch.long)
    imgs  = [denorm(x0[:1])]   # original

    for t in steps:
        xt, _ = scheduler.add_noise(x0[:1].to(scheduler.device),
                                    torch.tensor([t], device=scheduler.device))
        imgs.append(denorm(xt.cpu()))

    grid = vutils.make_grid(torch.cat(imgs, dim=0), nrow=len(imgs), padding=2)
    vutils.save_image(grid, save_path)
    print(f"  Noise progression saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# 7. TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f" DDPM Training  |  device={device}")
    print(f"{'='*60}\n")

    # ── Dataset ───────────────────────────────────────────────
    dataset = AnimalDataset(
        dataset_path=args.dataset_path,
        image_size=args.image_size,
        num_classes=args.num_classes,
        images_per_class=args.images_per_class,
    )
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=2, pin_memory=True)
    print(f"Dataset: {len(dataset)} images  |  Batches/epoch: {len(loader)}")

    # ── Noise scheduler ───────────────────────────────────────
    scheduler = NoiseScheduler(T=args.T, device=device)

    # ── Model ─────────────────────────────────────────────────
    model = UNet(in_ch=3, base_ch=args.base_ch,
                 ch_mults=(1, 2, 4), time_emb_dim=256).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"U-Net parameters: {n_params:.2f}M\n")

    # ── Optimiser & loss ──────────────────────────────────────
    optimiser = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=1e-4)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=1e-6
    )
    criterion = HybridDiffusionLoss(lambda_l2=0.8)

    # Save one noise-progression visualisation before training
    os.makedirs(args.save_dir, exist_ok=True)
    sample_batch = next(iter(loader)).to(device)
    save_noise_progression(
        sample_batch, scheduler,
        os.path.join(args.save_dir, "noise_progression.png")
    )

    loss_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            x0 = batch.to(device)                              # (B, C, H, W)
            B  = x0.size(0)

            # Sample random time-steps
            t = torch.randint(0, args.T, (B,), device=device)

            # Forward diffusion (closed-form)
            xt, eps_true = scheduler.add_noise(x0, t)

            # Predict noise
            eps_pred = model(xt, t)

            # Custom hybrid loss
            loss = criterion(eps_pred, eps_true)

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
            optimiser.step()

            epoch_loss += loss.item()

        scheduler_lr.step()
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)

        print(f"Epoch [{epoch:3d}/{args.epochs}]  loss={avg_loss:.5f}  "
              f"lr={optimiser.param_groups[0]['lr']:.2e}")

        # ── Periodic saves ────────────────────────────────────
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(
                args.save_dir, f"ddpm_epoch_{epoch:04d}.pth"
            )
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "opt_state":   optimiser.state_dict(),
                "loss":        avg_loss,
                "args":        vars(args),
            }, ckpt_path)
            print(f"  Checkpoint saved → {ckpt_path}")

            # Sample and save generated images
            gen = sample(model, scheduler, n_samples=8,
                         image_size=args.image_size, device=device)
            gen_path = os.path.join(args.save_dir,
                                    f"samples_epoch_{epoch:04d}.png")
            vutils.save_image(denorm(gen), gen_path, nrow=4)
            print(f"  Samples saved     → {gen_path}")

    # ── Final model save ──────────────────────────────────────
    final_path = os.path.join(args.save_dir, "ddpm_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved → {final_path}")

    # ── Loss curve ────────────────────────────────────────────
    plt.figure(figsize=(9, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history, linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Hybrid Loss (L2+L1)")
    plt.title("DDPM Training Loss – MSDS25055")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    curve_path = os.path.join(args.save_dir, "loss_curve.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"Loss curve saved  → {curve_path}")


# ─────────────────────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DDPM – Assignment 5 | MSDS25055")
    p.add_argument("--dataset_path",    type=str, required=True,
                   help="Path to dataset root (contains class sub-folders)")
    p.add_argument("--save_dir",        type=str, default="saved_models",
                   help="Where to store checkpoints and sample images")
    p.add_argument("--epochs",          type=int, default=50)
    p.add_argument("--batch_size",      type=int, default=16)
    p.add_argument("--lr",              type=float, default=2e-4)
    p.add_argument("--T",               type=int, default=1000,
                   help="Number of diffusion time steps")
    p.add_argument("--image_size",      type=int, default=64)
    p.add_argument("--base_ch",         type=int, default=64,
                   help="Base channel width of U-Net")
    p.add_argument("--num_classes",     type=int, default=5,
                   help="Number of animal classes to use")
    p.add_argument("--images_per_class",type=int, default=20,
                   help="Max images per class")
    p.add_argument("--save_every",      type=int, default=10,
                   help="Save checkpoint + samples every N epochs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
