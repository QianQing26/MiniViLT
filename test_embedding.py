from models.embedding import *
import json
from PIL import Image

# === 模拟小 vocab ===
vocab = {
    "[PAD]": 0,
    "[UNK]": 1,
    "a": 2,
    "child": 3,
    "in": 4,
    "pink": 5,
    "dress": 6,
    "is": 7,
    "climbing": 8,
    "stairs": 9,
    "entry": 10,
    "way": 11,
}
tokenizer = SimpleTokenizer(vocab)

# === 加载样本数据 ===
with open("data/flickr8k/flickr8k_data.json") as f:
    sample = json.load(f)[0]

img_path = sample["image_path"]
text = sample["captions"][0]

img = Image.open(img_path).convert("RGB")
token_ids = tokenizer.tokenize(text)

model = MultiModalEmbedder(vocab_size=len(vocab), embed_dim=256)
output = model(img, token_ids)

print("输出 shape:", output.shape)  # [1, seq_len, 256]
