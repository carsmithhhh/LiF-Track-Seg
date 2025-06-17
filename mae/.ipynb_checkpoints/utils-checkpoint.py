import torch
from torchvision import transforms
import random
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
from data_utils import *
from tqdm import tqdm
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Model(nn.Module):
    def __init__(self, patch_model, pos_model, encoder, decoder, patch_size, embed_dim, output_dim, device, masking_ratio=0.6, token_size=(5, 10, 10)):
        super().__init__()
        self.patch_model = patch_model
        self.pos_model = pos_model
        self.encoder = encoder
        self.decoder = decoder

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.masking_ratio = masking_ratio

        self.device = device

        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.05)
        self.token_size = token_size

    def forward(self, x):
        # patch embeddings
        patch_embeddings = self.patch_model(x) 

        # positional encodings
        patch_centers = get_patch_centers(x, self.patch_size)  # (B, N_patches, 3)
        patch_centers = patch_centers.to(self.device)
        pos_embeddings = self.pos_model(patch_centers)

        # Masking, getting visible tokens, storing indicies of masked tokens
        B, N, _ = patch_embeddings.shape
        # patch_embed_vis, ids_keep, ids_mask = random_masking(patch_embeddings, mask_ratio=self.masking_ratio) # original mask ratio 0.6
        Z, Y, X = x.shape[2:]
        Pz, Py, Px = self.token_size
        grid_shape = (Z // Pz, Y // Py, X // Px) 
        patch_embed_vis, ids_keep, ids_mask = random_masking(patch_embeddings, grid_shape=grid_shape) # does stratified masking
        pos_embed_vis = pos_embeddings[torch.arange(B).unsqueeze(1), ids_keep]

        # element-wise add visible tokens & positional encodings
        x = patch_embed_vis + pos_embed_vis

        # encode with transformer
        if self.masking_ratio == 0.0:
            latents = self.encoder(patch_embeddings + pos_embeddings)
            recon = self.decoder(patch_embeddings + pos_embeddings)
            return recon, None, None

        else:
            latents = self.encoder(x)
    
            x_full = torch.zeros(B, N, self.embed_dim, device=self.device)
            x_full.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim), patch_embed_vis)
            x_full.scatter_(1, ids_mask.unsqueeze(-1).expand(-1, -1, self.embed_dim), self.mask_token.expand(B, ids_mask.size(1), -1))
    
            decoder_input = x_full + pos_embeddings
            recon = self.decoder(decoder_input)
            return recon, ids_keep, ids_mask
        
def pretrain(model, train_loader, optimizer, epoch, epochs, device='cuda', patch_size=(5, 25, 25), batch_size=128, masking_ratio=0.6):
    model.train()
    log_file_path = os.path.join('./', f'per-it-train-loss.txt')
    total_loss = 0.0
    it_num = 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False, position=0)
    
    # with open(log_file_path, 'a') as log_file:
    for cubes in progress:
        # cubes = cubes.unsqueeze(dim=2) # shape (B, C=1, Z, Y, X)
        cubes = cubes.to(device)

        reconstructed_patches, ids_keep, ids_mask = model(cubes)

        all_patches = extract_voxel_patches(cubes, patch_size)  # (B, N, P)
        target_masked = all_patches[torch.arange(batch_size).unsqueeze(1), ids_mask]  # (B, N_mask, P)
        recon_masked = reconstructed_patches[torch.arange(batch_size).unsqueeze(1), ids_mask]  # (B, N_mask, P)

        if masking_ratio == 0.0:
            loss = F.mse_loss(reconstructed_patches, all_patches)
        else:
            loss = F.mse_loss(recon_masked, target_masked)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # per epoch stats
        total_loss += loss.item()
        it_num += 1
        avg_loss = total_loss / it_num
        progress.set_postfix(loss=f"{avg_loss:.4f}")
            # can decrease number of writes
            # if it_num % 16 == 0:
            #     log_file.write(f"{epoch},{it_num},{avg_loss:.6f}\n")
            #     log_file.flush()
            #     progress.set_postfix(loss=f"{avg_loss:.4f}")

    return avg_loss

@torch.no_grad()
def validate(model, val_loader, epoch, device='cuda', patch_size=(5, 10, 10), batch_size=64, save_dir='val_images4'):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    total_loss = 0.0
    iters = 0
    Pz, Py, Px = patch_size

    progress = tqdm(val_loader, desc=f"Epoch {epoch}", leave=False, position=0)
    # START HERE - DEBUG WHY THIS FAILS NOW SAD
    for batch_idx, cubes in enumerate(progress):
    # for batch_idx, (cubes, _, _) in enumerate(progress):
        # print(type(cubes))
        cubes = cubes.to(device) # (B, 1, Z, Y, X)

        reconstructed_patches, ids_keep, ids_mask = model(cubes)

        all_patches = extract_voxel_patches(cubes, patch_size)  # (B, N, P)
        target_masked = all_patches[torch.arange(cubes.shape[0]).unsqueeze(1), ids_mask]  # (B, N_mask, P)
        recon_masked = reconstructed_patches[torch.arange(cubes.shape[0]).unsqueeze(1), ids_mask]  # (B, N_mask, P)

        loss = F.mse_loss(recon_masked, target_masked)
        total_loss += loss.item()
        iters += 1

        for i in range(min(cubes.shape[0], 1)):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            # 1. Ground Truth full projection
            full_gt = cubes[i, 0].cpu().numpy()  # shape (Z, Y, X)
            full_proj = full_gt.sum(axis=0)
            im0 = axes[0].imshow(full_proj, cmap='viridis')
            axes[0].set_title(f'Full GT Patch, Epoch {epoch}')
            axes[0].axis('off')
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            
            # Draw red boxes for masked patch locations
            n_patches_y = full_gt.shape[1] // Py
            n_patches_x = full_gt.shape[2] // Px
            
            for j, patch_idx in enumerate(ids_mask[i].cpu().tolist()):
                if j == 0:
                    specific_patch_idx = patch_idx  # Save for later
                    continue  # Skip specific token for now
            
                row = patch_idx // n_patches_x
                col = patch_idx % n_patches_x
                y0 = row * Py
                x0 = col * Px
                rect = Rectangle((x0, y0), Px, Py, edgecolor='gray', facecolor='none', linewidth=1.5)
                axes[0].add_patch(rect)
            
            # Now draw the specific token in red (on top)
            row = specific_patch_idx // n_patches_x
            col = specific_patch_idx % n_patches_x
            y0 = row * Py
            x0 = col * Px
            rect = Rectangle((x0, y0), Px, Py, edgecolor='red', facecolor='none', linewidth=2.0)
            axes[0].add_patch(rect)
            
            # 2. Ground truth masked patch
            gt_patch = target_masked[i, 0].reshape(patch_size).cpu().numpy()
            gt_proj = gt_patch.sum(axis=0)
            im1 = axes[1].imshow(gt_proj, cmap='viridis')
            axes[1].set_title('Masked GT Patch')
            axes[1].axis('off')
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            # 3. Predicted patch
            pred_patch = recon_masked[i, 0].reshape(patch_size).cpu().numpy()
            pred_proj = pred_patch.sum(axis=0)
            im2 = axes[2].imshow(pred_proj, cmap='viridis')
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            
            fig.tight_layout()
            filename = os.path.join(save_dir, f'epoch{epoch}_batch{batch_idx}_sample{i}.png')
            plt.savefig(filename, dpi=150)
            plt.close(fig)

    avg_loss = total_loss / iters if iters > 0 else float('inf')
    return avg_loss

@torch.no_grad()
def overfit_test(model, overfit_loader, epoch, device='cuda', patch_size=(5, 25, 25), batch_size=256, save_dir='overfit_train_full', plot_num = 1):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    total_loss = 0.0
    iters = 0
    Pz, Py, Px = patch_size

    for batch_idx, cubes in enumerate(overfit_loader):
        # cubes = cubes.unsqueeze(1).to(device)  # (B, 1, Z, Y, X)
        cubes = cubes.to(device)
        Z, Y, X = cubes.shape[2:]
        grid_shape = (Z // Pz, Y // Py, X // Px)

        reconstructed_patches, ids_keep, ids_mask = model(cubes)

        all_patches = extract_voxel_patches(cubes, patch_size)  # (B, N, P)

        loss = F.mse_loss(reconstructed_patches, all_patches)
        total_loss += loss.item()
        iters += 1

        # Visualization loop
        for i in range(min(cubes.shape[0], plot_num)):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
            # Ground Truth
            full_gt = cubes[i, 0].cpu().numpy()  # shape (Z, Y, X)
            full_proj = full_gt.sum(axis=0)
            im0 = axes[0].imshow(full_proj, cmap='viridis')
            axes[0].set_title('Full GT Projection')
            axes[0].axis('off')
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
            # Reconstructed Volume
            recon_volume = reconstruct_from_patches(
                reconstructed_patches[i], patch_size, grid_shape
            ).cpu().numpy()
            recon_proj = recon_volume.sum(axis=0)
            im1 = axes[1].imshow(recon_proj, cmap='viridis')
            axes[1].set_title('Full Reconstructed Projection')
            axes[1].axis('off')
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
            # Save or show
            fig.tight_layout()
            filename = os.path.join(save_dir, f'epoch{epoch}_batch{batch_idx}_sample{i}.png')
            plt.savefig(filename, dpi=150)
            plt.close(fig)

    avg_loss = total_loss / iters if iters > 0 else float('inf')
    return avg_loss

def reconstruct_from_patches(patches, patch_size, grid_shape):
    pz, py, px = patch_size
    gz, gy, gx = grid_shape
    patches = patches.view(gz, gy, gx, pz, py, px)
    volume = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    return volume.view(gz * pz, gy * py, gx * px)

@torch.no_grad()
def final_test(model, test_loader, device='cuda', patch_size=(5, 10, 10), batch_size=64, save_dir='test_imgs'):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    total_loss = 0.0
    iters = 0
    Pz, Py, Px = patch_size
    
    for batch_idx, cubes in enumerate(test_loader):
        cubes = cubes.to(device)
        Z, Y, X = cubes.shape[2:]
        grid_shape = (Z // Pz, Y // Py, X // Px)

        reconstructed_patches, ids_keep, ids_mask = model(cubes)

        all_patches = extract_voxel_patches(cubes, patch_size)  # (B, N, P)
        target_masked = all_patches[torch.arange(cubes.shape[0]).unsqueeze(1), ids_mask]  # (B, N_mask, P)
        recon_masked = reconstructed_patches[torch.arange(cubes.shape[0]).unsqueeze(1), ids_mask]  # (B, N_mask, P)

        loss = F.mse_loss(recon_masked, target_masked)
        total_loss += loss.item()
        iters += 1

        # Visualizing Full Reconstructed Projections
        for i in range(cubes.shape[0]):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
            # Ground Truth
            full_gt = cubes[i, 0].cpu().numpy()  # shape (Z, Y, X)
            full_proj = full_gt.sum(axis=0)
            im0 = axes[0].imshow(full_proj, cmap='viridis')
            axes[0].set_title('Full TEST Ground Truth Projection')
            axes[0].axis('off')
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
            # Reconstructed Volume
            recon_volume = reconstruct_from_patches(
                reconstructed_patches[i], patch_size, grid_shape
            ).cpu().numpy()
            recon_proj = recon_volume.sum(axis=0)
            im1 = axes[1].imshow(recon_proj, cmap='viridis')
            axes[1].set_title('Full TEST Reconstructed Projection')
            axes[1].axis('off')
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
            # Grid dimensions
            Pz, Py, Px = patch_size
            n_patches_y = full_gt.shape[1] // Py
            n_patches_x = full_gt.shape[2] // Px
            N = n_patches_y * n_patches_x
        
            # Draw grid (gray) on both projections
            for patch_idx in range(N):
                row = patch_idx // n_patches_x
                col = patch_idx % n_patches_x
                y0 = row * Py
                x0 = col * Px
                rect_gt = Rectangle((x0, y0), Px, Py, edgecolor='gray', facecolor='none', linewidth=1.0)
                rect_recon = Rectangle((x0, y0), Px, Py, edgecolor='gray', facecolor='none', linewidth=1.0)
                axes[0].add_patch(rect_gt)
                axes[1].add_patch(rect_recon)
        
            # Overlay red boxes for masked patches
            for patch_idx in ids_mask[i].cpu().tolist():
                row = patch_idx // n_patches_x
                col = patch_idx % n_patches_x
                y0 = row * Py
                x0 = col * Px
                rect_gt = Rectangle((x0, y0), Px, Py, edgecolor='red', facecolor='none', linewidth=2.0)
                rect_recon = Rectangle((x0, y0), Px, Py, edgecolor='red', facecolor='none', linewidth=2.0)
                axes[0].add_patch(rect_gt)
                axes[1].add_patch(rect_recon)
        
            # Save
            fig.tight_layout()
            filename = os.path.join(save_dir, f'test_batch{batch_idx}_sample{i}.png')
            plt.savefig(filename, dpi=150)
            plt.close(fig)
        # for i in range(cubes.shape[0]):
        #     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        #     # Ground Truth
        #     full_gt = cubes[i, 0].cpu().numpy()  # shape (Z, Y, X)
        #     full_proj = full_gt.sum(axis=0)
        #     im0 = axes[0].imshow(full_proj, cmap='viridis')
        #     axes[0].set_title('Full TEST Ground Truth Projection')
        #     axes[0].axis('off')
        #     fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        #     # Reconstructed Volume
        #     recon_volume = reconstruct_from_patches(
        #         reconstructed_patches[i], patch_size, grid_shape
        #     ).cpu().numpy()
        #     recon_proj = recon_volume.sum(axis=0)
        #     im1 = axes[1].imshow(recon_proj, cmap='viridis')
        #     axes[1].set_title('Full TEST Reconstructed Projection')
        #     axes[1].axis('off')
        #     fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        #     # Save or show
        #     fig.tight_layout()
        #     filename = os.path.join(save_dir, f'test_batch{batch_idx}_sample{i}.png')
        #     plt.savefig(filename, dpi=150)
        #     plt.close(fig)

        # Visualizing Reconstructed Patches
        for i in range(min(cubes.shape[0], 30)):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            # 1. Ground Truth full projection
            full_gt = cubes[i, 0].cpu().numpy()  # shape (Z, Y, X)
            full_proj = full_gt.sum(axis=0)
            im0 = axes[0].imshow(full_proj, cmap='viridis')
            axes[0].set_title(f'Full Ground Truth Projection, Test Epoch')
            axes[0].axis('off')
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            
            # Draw red boxes for masked patch locations
            n_patches_y = full_gt.shape[1] // Py
            n_patches_x = full_gt.shape[2] // Px
            
            for j, patch_idx in enumerate(ids_mask[i].cpu().tolist()):
                if j == 0:
                    specific_patch_idx = patch_idx  # Save for later
                    continue  # Skip specific token for now
            
                row = patch_idx // n_patches_x
                col = patch_idx % n_patches_x
                y0 = row * Py
                x0 = col * Px
                rect = Rectangle((x0, y0), Px, Py, edgecolor='gray', facecolor='none', linewidth=1.5)
                axes[0].add_patch(rect)
            
            # Now draw the specific token in red (on top)
            row = specific_patch_idx // n_patches_x
            col = specific_patch_idx % n_patches_x
            y0 = row * Py
            x0 = col * Px
            rect = Rectangle((x0, y0), Px, Py, edgecolor='red', facecolor='none', linewidth=2.0)
            axes[0].add_patch(rect)
            
            # 2. Ground truth masked patch
            gt_patch = target_masked[i, 0].reshape(patch_size).cpu().numpy()
            gt_proj = gt_patch.sum(axis=0)
            im1 = axes[1].imshow(gt_proj, cmap='viridis')
            axes[1].set_title('Masked GT Patch')
            axes[1].axis('off')
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            # 3. Predicted patch
            pred_patch = recon_masked[i, 0].reshape(patch_size).cpu().numpy()
            pred_proj = pred_patch.sum(axis=0)
            im2 = axes[2].imshow(pred_proj, cmap='viridis')
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            
            fig.tight_layout()
            filename = os.path.join(save_dir, f'testepoch_batch{batch_idx}_sample{i}.png')
            plt.savefig(filename, dpi=150)
            plt.close(fig)

    avg_loss = total_loss / iters if iters > 0 else float('inf')
    return avg_loss
