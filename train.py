# train.py

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os, csv, json

from models.vilt import MiniViLT
from models.embedding import SimpleTokenizer
from data.itm_dataset import ITMDataset

# config
DATA_JSON = "data/flickr8k/flickr8k_data.json"
CHECKPOINT_DIR = "checkpoints"
LOG_FILE = "logs/training_log.csv"
EPOCHS = 200
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# vocab
with open("data/vocab.json") as f:
    vocab = json.load(f)
tokenizer = SimpleTokenizer(vocab)

# load data
dataset = ITMDataset(DATA_JSON, tokenizer)
tiny = torch.utils.data.Subset(dataset, indices=range(32))
# loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
loader = DataLoader(tiny, batch_size=BATCH_SIZE, shuffle=True)

# model
model = MiniViLT(vocab_size=len(vocab)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

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
        # imgs = [img.to(DEVICE) for img in imgs]
        imgs = imgs.to(DEVICE)
        # token_ids_batch = torch.stack(token_ids_batch).to(DEVICE)
        token_ids_batch = token_ids_batch.to(DEVICE)
        # labels = torch.tensor(labels).to(DEVICE)
        labels = labels.clone().detach().to(DEVICE)

        # 前向
        logits = model(imgs, token_ids_batch)

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
