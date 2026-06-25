
import os
import argparse
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Import all components from training script
from MSDS25055_05 import UNet, NoiseScheduler, sample, denorm


# ─────────────────────────────────────────────────────────────
# Helper: show denoising trajectory for ONE sample
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_with_trajectory(model, scheduler, image_size=64,
                            device="cpu", n_snapshots=10):
    model.eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)

    snapshot_steps = set(
        np.linspace(scheduler.T - 1, 0, n_snapshots, dtype=int).tolist()
    )
    trajectory = []

    for t_idx in reversed(range(scheduler.T)):
        t_tensor = torch.full((1,), t_idx, device=device, dtype=torch.long)
        eps_pred  = model(x, t_tensor)

        beta_t       = scheduler.betas[t_idx]
        alpha_t      = scheduler.alphas[t_idx]
        sqrt_om      = scheduler.sqrt_one_minus_alphas_cumprod[t_idx]

        coef = beta_t / sqrt_om
        mean = (1.0 / alpha_t.sqrt()) * (x - coef * eps_pred)

        if t_idx > 0:
            x = mean + beta_t.sqrt() * torch.randn_like(x)
        else:
            x = mean

        if t_idx in snapshot_steps:
            trajectory.append(x.clamp(-1, 1).cpu())

    return x.clamp(-1, 1).cpu(), trajectory


# ─────────────────────────────────────────────────────────────
# Main inference function
# ─────────────────────────────────────────────────────────────

def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading model from: {args.model_path}")
    print(f"Device: {device}\n")

    # ── Load checkpoint ───────────────────────────────────────
    ckpt = torch.load(args.model_path, map_location=device)

    # Support both full checkpoint dict and plain state_dict
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        saved_args = ckpt.get("args", {})
        image_size = saved_args.get("image_size", args.image_size)
        base_ch    = saved_args.get("base_ch",    args.base_ch)
        T          = saved_args.get("T",          args.T)
        print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}, "
              f"loss={ckpt.get('loss', float('nan')):.5f})")
    else:
        state_dict = ckpt
        image_size, base_ch, T = args.image_size, args.base_ch, args.T

    # ── Build model & scheduler ───────────────────────────────
    model = UNet(in_ch=3, base_ch=base_ch, ch_mults=(1, 2, 4),
                 time_emb_dim=256).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    scheduler = NoiseScheduler(T=T, device=device)
    print(f"Model loaded.  T={T}, image_size={image_size}, base_ch={base_ch}")

    # ── Generate a grid of samples ────────────────────────────
    print(f"\nGenerating {args.n_samples} sample images …")
    gen = sample(model, scheduler, n_samples=args.n_samples,
                 image_size=image_size, device=device)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    vutils.save_image(denorm(gen), args.output_path, nrow=4, padding=2)
    print(f"Grid saved → {args.output_path}")

    # ── Generate one sample with trajectory ───────────────────
    print("\nGenerating denoising trajectory (single sample) …")
    final, traj = sample_with_trajectory(
        model, scheduler, image_size=image_size,
        device=device, n_snapshots=10
    )

    n_cols = len(traj) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(2.2 * n_cols, 2.5))
    fig.suptitle("Denoising Trajectory  –  MSDS25055", fontsize=12, y=1.02)

    for i, img_t in enumerate(traj):
        axes[i].imshow(denorm(img_t[0]).permute(1, 2, 0).numpy())
        step = scheduler.T - 1 - int(i * (scheduler.T - 1) / max(len(traj) - 1, 1))
        axes[i].set_title(f"t={step}", fontsize=7)
        axes[i].axis("off")

    axes[-1].imshow(denorm(final[0]).permute(1, 2, 0).numpy())
    axes[-1].set_title("Final (x₀)", fontsize=7)
    axes[-1].axis("off")

    traj_path = args.output_path.replace(".png", "_trajectory.png")
    plt.tight_layout()
    plt.savefig(traj_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Trajectory saved → {traj_path}")

    print("\nDone.")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DDPM Inference – MSDS25055")
    p.add_argument("--model_path",  type=str, required=True,
                   help="Path to saved checkpoint (.pth)")
    p.add_argument("--output_path", type=str, default="generated.png",
                   help="Output path for generated image grid")
    p.add_argument("--n_samples",   type=int, default=8)
    p.add_argument("--image_size",  type=int, default=64,
                   help="Used only if checkpoint has no saved args")
    p.add_argument("--base_ch",     type=int, default=64)
    p.add_argument("--T",           type=int, default=1000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
