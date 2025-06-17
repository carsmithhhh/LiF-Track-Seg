import torch
import torch.nn as nn
from torch import nn

class PatchEmbed3D(torch.nn.Module):
    '''
    Input: a voxel cube of shape (B, C=1, Z, Y, X)
    Returns: a list of patches, each of shape (5, 16, 16) (z, y, x)
    '''
    def __init__(self, in_channels=1, patch_size=(5, 16, 16), embed_dim=128):
        super().__init__()
        self.proj = torch.nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C=1, Z, H, W)
        x = self.proj(x)  # → (B, embed_dim, Z', H', W')
        x = x.flatten(2).transpose(1, 2)  # → (B, N_patches, embed_dim)
        return x
