# train.py

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os, csv

from models.vilt import MiniViLT
from models.embedding import SimpleTokenizer
from data.itm_dataset import ITMDataset

# config
DATA_JSON = "data/flickr8k/flickr8k_data.json"
CHECKPOINT_DIR = "checkpoints"
LOG_FILE = "logs/training_log.csv"
EPOCHS = 10
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# vocab
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
    "up": 9,
    "stairs": 10,
    "entry": 11,
    "way": 12,
    "girl": 13,
    "wooden": 14,
    "playhouse": 15,
    "going": 16,
    "into": 17,
    "dog": 18,
    "and": 19,
    "black": 20,
    "white": 21,
    "little": 22,
}
tokenizer = SimpleTokenizer(vocab)

# load data
dataset = ITMDataset(DATA_JSON, tokenizer)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# model
model = MiniViLT(vocab_size=len(vocab)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# logs
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# csv
with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "loss", "acc"])

best_acc = 0

# main loop
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, total_correct = 0, 0

    for imgs, token_ids_batch, labels in tqdm(loader, desc=f"Epoch {epoch}"):
        imgs = [img.to(DEVICE) for img in imgs]
        token_ids_batch = torch.stack(token_ids_batch).to(DEVICE)
        labels = torch.tensor(labels).to(DEVICE)

        # 前向
        logits = torch.cat(
            [model(img, ids.unsqueeze(0)) for img, ids in zip(imgs, token_ids_batch)],
            dim=0,
        )

        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()

        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataset)
    acc = total_correct / len(dataset)

    print(f"📘 Epoch {epoch}: Loss={avg_loss:.4f}, Acc={acc:.4f}")

    # 保存最好模型
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best.pt"))

    # 写入日志
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, avg_loss, acc])
