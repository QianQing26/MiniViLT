# MiniViLT

**MiniViLT** 是一个从零实现的轻量级 Vision-and-Language Transformer（ViLT）模型，支持在 **8GB 显存**环境下进行训练，面向图文匹配（Image-Text Matching, ITM）任务。
本项目完全基于 PyTorch 实现，核心模块（Transformer、多模态融合、Patch Embedding 等）均为自定义代码，无需 HuggingFace 等封装库。

---

## ✨ 特性

* **轻量实现**：适合入门学习和低显存训练。
* **结构规范**：模块化工程目录，便于扩展。
* **纯手工实现**：包括多头自注意力、TransformerBlock、patch embedding 等。
* **可训练可复现**：提供完整的数据预处理、训练、日志与checkpoint保存逻辑。

---

## 📂 项目结构

```
MiniViLT/
├── configs/             # 超参数配置（可选）
├── data/
│   ├── flickr8k/        # 数据集
│   ├── build_vocab.py   # 构建词表
│   ├── itm_dataset.py   # ITM任务数据集
│   └── prepare_flickr8k_from_csv.py
├── models/
│   ├── embedding.py     # 图像+文本编码模块
│   ├── transformer.py   # Transformer实现
│   └── vilt.py          # ViLT主模型
├── utils/               # 工具函数（可扩展）
├── checkpoints/         # 保存模型参数
├── logs/                # 训练日志 (training_log.csv)
├── train.py             # 训练主脚本
└── README.md
```

---

## 📦 环境依赖

* Python >= 3.8
* PyTorch >= 1.12
* torchvision
* tqdm
* pandas
* Pillow

安装示例：

```bash
pip install torch torchvision tqdm pandas pillow
```

---

## 📥 数据集准备

本项目使用 **Flickr8k** 数据集：

1. 下载图像与caption文件（或使用脚本）：

   ```
   vilt_tutorial/data/flickr8k/
       ├── images/
       └── captions.txt
   ```

2. 生成 JSON 格式数据：

   ```bash
   python data/prepare_flickr8k_from_csv.py
   ```

3. 构建词表：

   ```bash
   python data/build_vocab.py
   ```

---

## 🚀 训练

```bash
python train.py
```

训练过程会在控制台输出 Loss 和 Accuracy，并在以下位置保存日志和模型：

* `logs/training_log.csv`
* `checkpoints/best.pt`

---

## 📊 日志示例

```
epoch,loss,acc
1,0.7086,0.5008
2,0.6994,0.5105
3,0.6975,0.5046
...
```

---
