import os
import json
from tqdm import tqdm
import pandas as pd

# 设置路径
DATA_DIR = "data/flickr8k"
IMG_DIR = os.path.join(DATA_DIR, "images")
CAPTION_FILE = os.path.join(DATA_DIR, "captions.txt")
OUTPUT_JSON = os.path.join(DATA_DIR, "flickr8k_data.json")


def build_caption_json():
    print("📚 加载数据中...")
    df = pd.read_csv(CAPTION_FILE)

    # group by image
    caption_dict = {}
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image = row["image"]
        caption = row["caption"]
        if image not in caption_dict:
            caption_dict[image] = []
        caption_dict[image].append(caption)

    # 构造样本列表
    samples = []
    for image_name, captions in caption_dict.items():
        samples.append(
            {"image_path": os.path.join(IMG_DIR, image_name), "captions": captions}
        )

    with open(OUTPUT_JSON, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"✅ 成功保存 {len(samples)} 条图文对到 {OUTPUT_JSON}")


if __name__ == "__main__":
    build_caption_json()
