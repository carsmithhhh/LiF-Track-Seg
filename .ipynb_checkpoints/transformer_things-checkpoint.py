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

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) implementation.
    """
    def __init__(self, img_size=32, patch_size=8, in_channels=3,
                 embed_dim=128, num_layers=6, num_heads=4,
                 dim_feedforward=256, num_classes=10, dropout=0.1):
        """
        Inputs:
         - img_size: Size of input image (assumed square).
         - patch_size: Size of each patch (assumed square).
         - in_channels: Number of image channels.
         - embed_dim: Embedding dimension for each patch.
         - num_layers: Number of Transformer encoder layers.
         - num_heads: Number of attention heads.
         - dim_feedforward: Hidden size of feedforward network.
         - num_classes: Number of classification labels.
         - dropout: Dropout probability.
        """
        super().__init__()
        self.num_classes = num_classes
        # self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        # self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout)

        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classification layer to predict class scores from pooled token.
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        """
        Initialize the weights of the network.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Forward pass of Vision Transformer.

        Inputs:
         - x: Input image tensor of shape (N, C, H, W)

        Returns:
         - logits: Output classification logits of shape (N, num_classes)
        """
        N = x.size(0)
        logits = torch.zeros(N, self.num_classes, device=x.device)
        
        patches = self.patch_embed(x)
        patches = self.positional_encoding(patches)

        embeddings = self.transformer(patches) # (3, 16, 128) --> want to pool over class dim, 16?
        avg_embeddings = torch.mean(embeddings, dim=1) # (3, 128)
        
        logits = self.head(avg_embeddings)

        return logits


  