# models/embedding.py

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


# simple tokenizer
class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.unk_id = vocab.get("[UNK]", 1)
        self.pad_id = vocab.get("[PAD]", 0)

    def tokenize(self, text, max_len=32):
        tokens = text.lower().split()
        ids = [self.vocab.get(w, self.unk_id) for w in tokens]
        ids = ids[:max_len]
        ids += [self.pad_id] * (max_len - len(ids))
        return torch.tensor(ids)


# text embedding
class TextEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len=32):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(max_len, embed_dim))
        self.type_embed = nn.Parameter(torch.randn(1, embed_dim))  # text

    def forward(self, token_ids):
        x = self.token_embed(token_ids)  # [B, L, D]
        pos = self.pos_embed[: x.size(1)]  # [L, D]
        type_embed = self.type_embed.expand_as(pos)
        return x + pos + type_embed


# patch embedding
class PatchEmbeddings(nn.Module):
    def __init__(self, img_size=224, patch_size=32, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(self.n_patches, embed_dim))
        self.type_embed = nn.Parameter(torch.randn(1, embed_dim))

        self.transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])

    def forward(self, img_pil):
        img = self.transform(img_pil).unsqueeze(0)  # [1, 3, H, W]
        x = self.proj(img)  # [1, D, H', W']
        x = x.flatten(2).transpose(1, 2)  # [1, N, D]
        pos = self.pos_embed[: x.size(1)]
        type_embed = self.type_embed.expand_as(pos)
        return x + pos + type_embed


# multi-model embedding
class MultiModalEmbedder(nn.Module):
    def __init__(
        self, vocab_size, embed_dim=256, img_size=224, patch_size=32, max_text_len=32
    ):
        super().__init__()
        self.text_encoder = TextEmbeddings(vocab_size, embed_dim, max_len=max_text_len)
        self.vision_encoder = PatchEmbeddings(img_size, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, img_pil, token_ids):
        image_embed = self.vision_encoder(img_pil)  # [1, N_img, D]
        text_embed = self.text_encoder(token_ids.unsqueeze(0))  # [1, N_txt, D]
        cls = self.cls_token.expand(1, 1, image_embed.size(-1))  # [1, 1, D]
        output = torch.cat([cls, image_embed, text_embed], dim=1)  # [1+N+M, D]
        return output  # [1, seq_len, dim]
