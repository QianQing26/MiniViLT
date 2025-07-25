from models.transformer import TransformerEncoder
import torch

dummy_input = torch.randn(1, 64, 256)  # [B, seq_len, dim]
model = TransformerEncoder(dim=256, depth=6, n_heads=4)

out = model(dummy_input)
print("✅ 输出 shape:", out.shape)  # [1, 64, 256]
