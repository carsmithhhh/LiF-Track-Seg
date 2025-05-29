import torch
import torch.nn as nn

# TODO: get_pos_embed
#@torch.no_grad()

class LearnedPositionalEncoder(nn.Module):
# potentailly option to use intensity values or not (see sam relative features)

  def __init__(self, num_channels: int, embed_dim: int):
    super().__init__()
    self.embed_dim = embed_dim
    
    self.pos_enc = nn.Sequential(
      nn.Linear(3, 128), #xyz --> big
      nn.GELU(),
      nn.Linear(128, embed_dim) #big --> 
    )

  def reset_parameters(self):
    for p in self.parameters():
      if isinstance(p, nn.Linear):
        p.reset_parameters()

  def forward(self, pos: torch.Tensor) -> torch.Tensor:
    #sam line: pos = self.pos_enc(pos[...,:3])
    pos_embed = self.pos_enc(pos)
    return pos_embed



