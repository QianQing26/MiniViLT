# models/vilt.py
import torch
import torch.nn as nn
from models.embedding import MultiModalEmbedder, SimpleTokenizer
from models.transformer import TransformerEncoder


class MiniViLT(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, depth=4, n_heads=4, max_text_len=32):
        super().__init__()
        self.embedder = MultiModalEmbedder(
            vocab_size=vocab_size, embed_dim=embed_dim, max_text_len=max_text_len
        )
        self.encoder = TransformerEncoder(dim=embed_dim, depth=depth, n_heads=n_heads)

        self.itm_head = nn.Linear(embed_dim, 2)

    def forward(self, img_pil, token_ids):
        x = self.embedder(img_pil, token_ids)
        x = self.encoder(x)
        cls_token = x[:, 0]
        logits = self.itm_head(cls_token)
        return logits
