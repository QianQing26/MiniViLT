from models.vilt import MiniViLT
from models.embedding import SimpleTokenizer
from PIL import Image
import json
import torch

# === dummy vocab ===
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

# === 加载示例数据 ===
with open("data/flickr8k/flickr8k_data.json") as f:
    sample = json.load(f)[0]

img = Image.open(sample["image_path"]).convert("RGB")
text = sample["captions"][0]
token_ids = tokenizer.tokenize(text)

# === 构建模型并前向 ===
model = MiniViLT(vocab_size=len(vocab))
logits = model(img, token_ids)

print("图文匹配 logits:", logits)
