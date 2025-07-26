# data/itm_dataset.py
import os
import json
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class ITMDataset(Dataset):
    def __init__(self, json_file, tokenizer, negative_ratio=0.5, max_len=32):
        with open(json_file, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.negative_ratio = negative_ratio

        self.transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),  # 将 PIL.Image 转为 Tensor
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if random.random() < self.negative_ratio:
            # Negative sample
            img_idx = idx
            txt_idx = random.randint(0, len(self.data) - 1)
            while txt_idx == img_idx:
                txt_idx = random.randint(0, len(self.data) - 1)
            item = self.data[img_idx]
            other = self.data[txt_idx]
            text = random.choice(other["captions"])
            label = 0
        else:
            # Positive sample
            item = self.data[idx]
            text = random.choice(item["captions"])
            label = 1

        img = Image.open(item["image_path"]).convert("RGB")
        img = self.transform(img)
        token_ids = self.tokenizer.tokenize(text, max_len=self.max_len)

        return img, token_ids, label
