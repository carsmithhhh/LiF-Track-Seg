import torch
from torchvision import transforms
import random
from torch.utils.data import Dataset
import numpy as np

# cubes is a list of our 3d images
class CubeDataset(torch.utils.data.Dataset): # returns dataset on cpu
  def __init__(self, cubes):
    self.cubes = cubes
    np.random.seed(42)

  def __len__(self):
    return len(self.cubes)

  def __getitem__(self, idx):
    return self.cubes[idx]


def random_masking(x, mask_ratio=0.6):
    """
    x: (B, N, D) patch embeddings
    Returns:
        visible_x: (B, N_vis, D)
        mask_indices: (B, N_masked) indices of masked patches
        unmask_indices: (B, N_vis) indices of visible patches
    """
    B, N, _ = x.shape
    N_vis = int(N * (1 - mask_ratio))

    noise = torch.rand(B, N, device=x.device)  # (B, N)
    ids_sorted = torch.argsort(noise, dim=1)   # ascending order
    ids_keep = ids_sorted[:, :N_vis]
    ids_mask = ids_sorted[:, N_vis:]

    # Gather visible tokens
    batch_idx = torch.arange(B).unsqueeze(-1).to(x.device)  # (B, 1)
    x_visible = x[batch_idx, ids_keep]

    return x_visible, ids_keep, ids_mask

def get_patch_centers(volume: torch.Tensor, patch_size: tuple) -> torch.Tensor:
    """
    Extract normalized patch center positions from a 3D volume.

    Args:
        volume (Tensor): Input tensor of shape (B, 1, Z, Y, X)
        patch_size (tuple): Tuple of (pz, py, px) indicating patch size

    Returns:
        Tensor: (B, N, 3) tensor of normalized patch centers in [-1, 1]^3
    """
    B, C, Z, Y, X = volume.shape
    pz, py, px = patch_size
    assert Z % pz == 0 and Y % py == 0 and X % px == 0

    # Number of patches in each dim
    nz = Z // pz
    ny = Y // py
    nx = X // px
    N = nz * ny * nx  # total number of patches per sample

    # Compute the *absolute* (unnormalized) voxel center positions
    z_centers = torch.arange(pz//2, Z, step=pz, dtype=torch.float32)
    y_centers = torch.arange(py//2, Y, step=py, dtype=torch.float32)
    x_centers = torch.arange(px//2, X, step=px, dtype=torch.float32)

    zz, yy, xx = torch.meshgrid(z_centers, y_centers, x_centers, indexing="ij")
    coords = torch.stack([zz, yy, xx], dim=-1)  # (nz, ny, nx, 3)
    coords = coords.view(-1, 3)  # (N, 3)

    # Normalize to [-1, 1] using shape
    norm_coords = coords.clone()
    norm_coords[:, 0] = 2 * (coords[:, 0] / (Z - 1)) - 1
    norm_coords[:, 1] = 2 * (coords[:, 1] / (Y - 1)) - 1
    norm_coords[:, 2] = 2 * (coords[:, 2] / (X - 1)) - 1

    # Repeat for all batch elements
    norm_coords = norm_coords.unsqueeze(0).repeat(B, 1, 1)  # (B, N_patches, 3)
    return norm_coords

# will be used to access ground truth data for evaluating loss!!
def extract_voxel_patches(volume, patch_size):
    """
    Extract 3D non-overlapping patches from volume.
    
    Args:
        volume: (B, 1, Z, Y, X) torch.Tensor
        patch_size: (pz, py, px)
        
    Returns:
        patches: (B, N, P) where P = pz * py * px
    """
    B, C, Z, Y, X = volume.shape
    pz, py, px = patch_size
    assert C == 1, "Expected single-channel input"
    
    # unfold splits along sliding windows
    patches = volume.unfold(2, pz, pz).unfold(3, py, py).unfold(4, px, px)
    # (B, 1, Z//pz, Y//py, X//px, pz, py, px)

    patches = patches.contiguous().view(B, -1, pz * py * px)  # (B, N, P)
    return patches