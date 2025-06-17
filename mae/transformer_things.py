import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    attn = MultiHeadAttention(embed_dim, num_heads=2)

    # self-attention
    data = torch.randn(batch_size, sequence_length, embed_dim)
    self_attn_output = attn(query=data, key=data, value=data)

    # attention using two inputs
    other_data = torch.randn(batch_size, sequence_length, embed_dim)
    attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, query, key, value, attn_mask=None):
        """
        N is the batch size, S is the source sequence length, T 
        is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
          i in the source should not influence token j in the target.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        N, S, E = query.shape
        N, T, E = value.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, E))
        # splitting shape
        H = self.n_head
        head_dim = self.head_dim
        query = self.query(query).reshape(N, S, H, head_dim).transpose(1, 2)  # (N, H, S, head_dim)
        key = self.key(key).reshape(N, T, H, head_dim).transpose(1, 2) # (N, H, T, head_dim)
        value = self.value(value).reshape(N, T, H, head_dim).transpose(1, 2)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)  # (N, H, S, T)

        # apply attn-mask on arg to set values above diagonal ot -inf
        if attn_mask is not None:
          attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)  # (N, H, S, T)
        attn_weights = self.attn_drop(attn_weights)
        
        attn = torch.matmul(attn_weights, value)  # (N, H, S, head_dim)
        # merging the heads
        attn = attn.transpose(1, 2).reshape(N, S, E)  # (N, S, E)
        output = self.proj(attn)
        
        return output

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        """
        Simple two-layer feed-forward network with dropout and ReLU activation.

        Inputs:
        - embed_dim: Dimension of input and output embeddings
        - ffn_dim: Hidden dimension in the feedforward network
        - dropout: Dropout probability
        """
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass for the feedforward network.

        Inputs:
        - x: Input tensor of shape (N, T, D)

        Returns:
        - out: Output tensor of the same shape as input
        """
        out = torch.empty_like(x)

        out = self.fc1(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

class TransformerEncoderLayer(nn.Module):
    """
    A single layer of a Transformer encoder, to be used with TransformerEncoder.
    """
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        """
        Construct a TransformerEncoderLayer instance.

        Inputs:
         - input_dim: Number of expected features in the input.
         - num_heads: Number of attention heads.
         - dim_feedforward: Dimension of the feedforward network model.
         - dropout: The dropout value.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(input_dim, dim_feedforward, dropout)

        self.norm_self = nn.LayerNorm(input_dim)
        self.norm_ffn = nn.LayerNorm(input_dim)

        self.dropout_self = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        """
        Pass the inputs (and mask) through the encoder layer.

        Inputs:
        - src: the sequence to the encoder layer, of shape (N, S, D)
        - src_mask: the parts of the source sequence to mask, of shape (S, S)

        Returns:
        - out: the Transformer features, of shape (N, S, D)
        """
        # self-attn layer
        shortcut = src
        self_attn = self.self_attn(query=src, key=src, value=src, attn_mask=src_mask)
        self_attn = self.dropout_self(self_attn)
        src = self.norm_self(self_attn + shortcut)

        # feedforward layer
        shortcut = src
        ffn = self.ffn(src)
        ffn = self.dropout_ffn(ffn)
        src = self.norm_ffn(ffn + shortcut)
        return src

class VisionTransformer3D(nn.Module):
    def __init__(self, input_dim=384, num_layers=6, num_heads=4,
                 dim_feedforward=1024, dropout=0.1):
        """
        Transformer encoder for 3D patch sequences (no classification head).
        """
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            input_dim=input_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.encoder = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = nn.LayerNorm(input_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, embeddings):
        """
        Args:
            concatenation of: 
            patch_embed: (B, N, D) patch embeddings
            pos_embed:   (B, N, D) positional embeddings

        Returns:
            out: (B, N, D) context-enriched features
        """
        # x = patch_embed + pos_embed  # (B, N, D)
        x = embeddings
        
        for layer in self.encoder:
            x = layer(x)

        x = self.norm(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class Decoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim=256, num_layers=2, num_heads=4, dropout=0.1, output_dim=3125):
        """
        Args:
            embed_dim: Input dim from encoder
            decoder_dim: Decoder hidden dim
            output_dim: Output dim per patch (e.g., 3 for coords, 4 for (x,y,z,intensity))
        """
        super().__init__()

        self.proj = nn.Linear(embed_dim, hidden_dim, bias=True)
        self.blocks = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        # Final projection to predict patch output
        self.head = nn.Linear(hidden_dim, output_dim) 

    def forward(self, x):
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.head(x)  # (B, N, output_dim)
        return x



  