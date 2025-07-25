# data/build_vocab.py

import json
from collections import Counter
from tqdm import tqdm

CAPTION_JSON = "data/flickr8k/flickr8k_data.json"
VOCAB_FILE = "data/vocab.json"
MIN_FREQ = 2


def build_vocab():
    with open(CAPTION_JSON, "r") as f:
        data = json.load(f)

    counter = Counter()
    for sample in tqdm(data, desc="🔍 统计词频"):
        for caption in sample["captions"]:
            tokens = caption.lower().strip().split()
            counter.update(tokens)

    # 添加特殊token
    vocab = {"[PAD]": 0, "[UNK]": 1}
    idx = 2
    for word, freq in counter.items():
        if freq >= MIN_FREQ:
            vocab[word] = idx
            idx += 1

    print(f"构建词表完成，共有 {len(vocab)} 个词")

    with open(VOCAB_FILE, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"词表已保存到 {VOCAB_FILE}")


if __name__ == "__main__":
    build_vocab()
