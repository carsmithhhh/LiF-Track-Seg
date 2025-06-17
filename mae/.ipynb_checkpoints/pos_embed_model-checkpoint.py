import torch
import torch.nn as nn

# TODO: get_pos_embed
#@torch.no_grad()

class LearnedPositionalEncoder(nn.Module):
    """
    Applies a learned positional encoding to 3D coordinates (with optional intensity).
    Transforms (x, y, z) or (x, y, z, intensity) into a vector of dimension `embed_dim`.
    """
    def __init__(self, in_dim: int = 3, embed_dim: int = 128, use_layernorm: bool = False):
        """
        Args:
            in_dim (int): Input feature size (3 for x,y,z or 4 if including intensity).
            embed_dim (int): Output embedding dimension.
            use_layernorm (bool): Whether to apply layer normalization at the end.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.use_layernorm = use_layernorm

        self.pos_enc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )

        if self.use_layernorm:
            self.norm = nn.LayerNorm(embed_dim)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos (Tensor): (B, N, 3) or (B, N, 4) tensor of positions (x, y, z [, intensity]).
        Returns:
            Tensor: Positional embeddings of shape (B, N, embed_dim).
        """
        assert pos.shape[-1] in (3, 4)
        pos_embed = self.pos_enc(pos)
        if self.use_layernorm:
            pos_embed = self.norm(pos_embed)
        return pos_embed


# class LearnedPositionalEncoder(nn.Module):
# # potentailly option to use intensity values or not (use_relative_features in polarmae)

#   def __init__(self, num_channels: int, embed_dim: int):
#     super().__init__()
#     self.embed_dim = embed_dim
    
#     self.pos_enc = nn.Sequential(
#       nn.Linear(3, 128), #xyz --> big
#       nn.GELU(),
#       nn.Linear(128, embed_dim) #big --> 
#     )

#   def reset_parameters(self):
#     for p in self.parameters():
#       if isinstance(p, nn.Linear):
#         p.reset_parameters()

#   def forward(self, pos: torch.Tensor) -> torch.Tensor:
#     pos_embed = self.pos_enc(pos)
#     return pos_embed



