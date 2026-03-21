"""
train.py — Train cloud (full) and edge (compressed) Rectified Flow models on CIFAR-10.

Usage:
    python train.py --mode cloud --epochs 100 --save_dir checkpoints/
    python train.py --mode edge  --epochs 100 --save_dir checkpoints/
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models import build_cloud_model, build_edge_model, count_params
from rectified_flow import RectifiedFlowTrainer, RectifiedFlowSampler


def get_cifar10_loader(batch_size: int = 128, train: bool = True) -> DataLoader:
    """CIFAR-10 dataloader with standard normalization to [-1, 1]."""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # → [-1, 1]
    ])
    dataset = datasets.CIFAR10(
        root='./data', train=train, download=True, transform=transform
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train,
        num_workers=4, pin_memory=True, drop_last=train,
    )


def train_model(
    model: nn.Module,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 2e-4,
    save_dir: str = 'checkpoints/',
    model_name: str = 'model',
    device: str = 'cuda',
    log_every: int = 100,
    save_every: int = 10,
):
    """Train a Rectified Flow model."""
    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    trainer = RectifiedFlowTrainer(model)
    loader = get_cifar10_loader(batch_size=batch_size, train=True)

    print(f"Training {model_name} ({count_params(model)/1e6:.1f}M params) "
          f"for {epochs} epochs on {device}")
    print(f"Dataset: CIFAR-10, Batch size: {batch_size}, LR: {lr}")

    best_loss = float('inf')
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for batch_idx, (x, _) in enumerate(pbar):
            x = x.to(device)
            loss = trainer.compute_loss(x)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % log_every == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / num_batches
        print(f"  Epoch {epoch}: avg_loss = {avg_loss:.4f}, lr = {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        if epoch % save_every == 0 or avg_loss < best_loss:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(ckpt, os.path.join(save_dir, f'{model_name}_best.pt'))
                print(f"  → Saved best model (loss={avg_loss:.4f})")
            if epoch % save_every == 0:
                torch.save(ckpt, os.path.join(save_dir, f'{model_name}_epoch{epoch}.pt'))

    print(f"Training complete. Best loss: {best_loss:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train RF models for edge-cloud experiment")
    parser.add_argument('--mode', choices=['cloud', 'edge', 'both'], default='both',
                        help='Which model to train')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--save_dir', type=str, default='checkpoints/')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if args.mode in ('cloud', 'both'):
        print("\n" + "=" * 60)
        print("Training CLOUD model (full-size UNet)")
        print("=" * 60)
        cloud_model = build_cloud_model()
        train_model(
            cloud_model, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, save_dir=args.save_dir, model_name='cloud',
            device=device,
        )

    if args.mode in ('edge', 'both'):
        print("\n" + "=" * 60)
        print("Training EDGE model (pruned UNet)")
        print("=" * 60)
        edge_model = build_edge_model()
        train_model(
            edge_model, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, save_dir=args.save_dir, model_name='edge',
            device=device,
        )


if __name__ == "__main__":
    main()
