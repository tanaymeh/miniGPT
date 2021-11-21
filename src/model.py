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

        self.attn = nn.MultiheadAttention(num_heads=num_heads, embed_dim=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim)
        )

        self.drop1 = nn.Dropout(rate)
        self.drop2 = nn.Dropout(rate)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x_shape = x.shape()
        batch_size = x_shape[0]
        seq_len = x_shape[1]

        casual_attn_mask = casualAttentionMask(
            batch_size=batch_size,
            dest_size=seq_len, 
            src_size=seq_len, 
            dtype=torch.bool
        )

        attn_out = self.attn(x, x, attn_mask=casual_attn_mask)
        attn_out = self.drop1(attn_out)
        out1 = self.ln1(x + attn_out)
        mlp_out = self.mlp(out1)
        mlp_out = self.drop2(mlp_out)
        a = self.ln2(out1) + mlp_out

        print(a.shape)
        return self.ln2(out1 + mlp_out)

class TokenPositionEmbedding(nn.Module):
    """
    2 Seperate embedding layers: 1 for tokens and 1 for positional embeddings
    """
    def __init__(self, vocab_size, embed_dim, maxlen):
        super(TokenPositionEmbedding, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(maxlen, embed_dim)
    
    def forward(self, x):
        maxlen = x.size(-1)
        positions = torch.range(start=0, end=maxlen, step=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)

        return x + positions

class GPT(nn.Module):
    """
    Binding all the functions in one single module to make the mini GPT model
    """
    def __init__(self, config):
        self.emb = TokenPositionEmbedding(
            maxlen=config.maxlen, 
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim
        )

        self.transformer_block = TransformerBlock(
            embedding_dim=config.embed_dim,
            num_heads=config.num_heads,
            ff_dim=config.feed_forward_dim
        )
        
        self.d1 = nn.Linear(config.embed_size, config.vocab_size)
    
    def forward(self, x):
        x = self.emb(x)
        x = self.transformer_block(x)
        return x, self.d1(x)