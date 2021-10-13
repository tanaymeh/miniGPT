import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

def casualAttentionMask(batch_size, dest_size, src_size, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = torch.range(dest_size)[:, None]
    j = torch.range(src_size)
    m = i >= j - src_size + dest_size
    
    mask = torch.tensor(m, dtype=dtype)
    mask = torch.reshape(mask, [1, dest_size, src_size])
    mult = torch.cat(
        [torch.unsqueeze(batch_size, -1), torch.tensor([1, 1], dtype=torch.int32)], 0
    )

    return torch.tile(mask, mult)

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(0.35),
        )

        self.n1 = nn.LayerNorm(embedding_dim)
        self.n2 = nn.LayerNorm(embedding_dim)
    
    def forward(self, inputs):
        shape = inputs.shape
        batch_size = shape[0]
        seq_len = shape[1]

        casual_mask = casualAttentionMask(batch_size, seq_len, seq_len, torch.bool)        
        # attn_out = self.attn(inputs, inputs, )