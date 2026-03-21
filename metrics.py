"""
metrics.py — FID computation and other evaluation metrics.

Uses pytorch-fid under the hood for reliable FID computation.
Also includes lightweight metrics for quick iteration.
"""

import os
import torch
import numpy as np
from pathlib import Path
from torchvision.utils import save_image


def save_samples_to_dir(
    samples: torch.Tensor,
    output_dir: str,
    prefix: str = "sample",
) -> str:
    """
    Save generated samples as PNG images for FID computation.

    Args:
        samples:    tensor [N, 3, 32, 32] in [-1, 1]
        output_dir: directory to save images
        prefix:     filename prefix
    Returns:
        output_dir path
    """
    os.makedirs(output_dir, exist_ok=True)
    # Rescale from [-1,1] to [0,1]
    samples = (samples.clamp(-1, 1) + 1) / 2

    for i in range(samples.shape[0]):
        save_image(samples[i], os.path.join(output_dir, f"{prefix}_{i:05d}.png"))

    return output_dir


def compute_fid(
    generated_dir: str,
    reference_dir: str = None,
    device: str = "cuda",
    batch_size: int = 64,
) -> float:
    """
    Compute FID between generated samples and reference dataset.

    If reference_dir is None, uses CIFAR-10 train statistics
    (requires precomputed stats or falls back to pytorch-fid).

    Args:
        generated_dir: path to folder with generated .png images
        reference_dir: path to folder with reference .png images
        device:        computation device
        batch_size:    batch size for feature extraction
    Returns:
        FID score (float)
    """
    try:
        from pytorch_fid import fid_score
        fid = fid_score.calculate_fid_given_paths(
            [generated_dir, reference_dir],
            batch_size=batch_size,
            device=device,
            dims=2048,
        )
        return fid
    except ImportError:
        print("Warning: pytorch-fid not installed. Using simplified FID.")
        return _simplified_fid(generated_dir, reference_dir, device)


def _simplified_fid(gen_dir, ref_dir, device):
    """
    Simplified FID using InceptionV3 features directly.
    Fallback when pytorch-fid is not available.
    """
    from torchvision import models, transforms
    from scipy import linalg

    # Load InceptionV3
    inception = models.inception_v3(pretrained=True, transform_input=False)
    inception.fc = torch.nn.Identity()  # Remove classifier
    inception = inception.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
    ])

    def get_features(img_dir):
        from PIL import Image
        features = []
        for fname in sorted(os.listdir(img_dir)):
            if not fname.endswith('.png'):
                continue
            img = Image.open(os.path.join(img_dir, fname)).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = inception(img)
            features.append(feat.cpu().numpy())
        return np.concatenate(features, axis=0)

    feat_gen = get_features(gen_dir)
    feat_ref = get_features(ref_dir)

    mu_gen, sigma_gen = feat_gen.mean(0), np.cov(feat_gen, rowvar=False)
    mu_ref, sigma_ref = feat_ref.mean(0), np.cov(feat_ref, rowvar=False)

    diff = mu_gen - mu_ref
    covmean, _ = linalg.sqrtm(sigma_gen @ sigma_ref, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_gen + sigma_ref - 2 * covmean)
    return float(fid)


def prepare_cifar10_reference(output_dir: str = "data/cifar10_ref/", num_images: int = 10000):
    """
    Save CIFAR-10 test images as PNGs for FID reference.
    Only needs to run once.
    """
    from torchvision import datasets, transforms

    os.makedirs(output_dir, exist_ok=True)
    dataset = datasets.CIFAR10(
        root='./data', train=False, download=True,
        transform=transforms.ToTensor(),
    )
    for i in range(min(num_images, len(dataset))):
        img, _ = dataset[i]
        save_image(img, os.path.join(output_dir, f"ref_{i:05d}.png"))
    print(f"Saved {min(num_images, len(dataset))} reference images to {output_dir}")
    return output_dir


# ============================================================
# Lightweight proxy metrics (for fast iteration, not for paper)
# ============================================================

def pixel_mse(generated: torch.Tensor, reference: torch.Tensor) -> float:
    """Simple MSE between generated and reference batches."""
    return ((generated - reference) ** 2).mean().item()


def sample_diversity(samples: torch.Tensor) -> float:
    """
    Measure diversity as average pairwise L2 distance.
    Higher = more diverse.
    """
    B = samples.shape[0]
    flat = samples.reshape(B, -1)
    # Random subset for efficiency
    n = min(B, 100)
    idx = torch.randperm(B)[:n]
    sub = flat[idx]
    dists = torch.cdist(sub, sub)
    # Average off-diagonal
    mask = ~torch.eye(n, dtype=bool)
    return dists[mask].mean().item()
