# Analgesic-peptide-prediction-based-on-deep-learning

# 蛋白质序列分类（镇痛肽预测）

基于 ESM‑2 蛋白质语言模型和 CNN‑LSTM‑Attention 架构的二分类项目。  
支持从 FASTA 序列文件出发，完成数据划分、特征提取、模型训练、评估与预测的全流程。

## ✨ 项目特点

- 使用 **ESM‑2** (facebook/esm2_t12_35M_UR50D) 提取氨基酸级别的嵌入特征
- 模型结构：**1D‑CNN + BiLSTM + 注意力机制**，参数量适中，适合小样本数据
- 提供完整的数据集划分、特征提取、训练、测试与预测脚本
- 支持 **per‑residue**（每个氨基酸一个向量）和 **global**（整个肽段平均池化）两种特征模式

## 📁 项目结构
```bash
.
├── data/ # 数据目录
│ ├── raw/ # 原始 FASTA（正负样本）
│ ├── processed/ # 划分后的 train/val/test.fasta
│ └── features/ # 提取的 ESM‑2 特征（pickle/npy）
├── src/ # 源代码
│ ├── data/
│ │ ├── split_data.py # 数据集划分
│ │ └── dataset.py # PyTorch Dataset
│ ├── features/
│ │ └── feature.py # ESM‑2 特征提取
│ ├── models/
│ │ └── model.py # 模型定义
│ ├── train.py # 训练脚本
│ └── predict.py # 预测与评估
├── checkpoints/ # 保存的模型权重（自动创建）
├── requirements.txt
└── README.md
```


## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆或下载本项目
cd peptide_classifier

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate   # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```


### 2. 准备原始数据
将您的正负样本 FASTA 文件放入 data/raw/ 目录：

positive100.fasta（正样本，标签为 1）

negative100.fasta（负样本，标签为 0）

文件编码需为 UTF‑16（如原脚本所述）。如果为 UTF‑8，请修改 split_data.py 中的 encoding 参数。


### 3. 划分数据集
```bash
cd src/data
python split_data.py
```
输出文件将保存在 data/processed/ 下：train.fasta, val.fasta, test.fasta。


### 4. 提取 ESM‑2 特征
```bash
cd ../features
python feature.py \
    --input_dir ../../data/processed \
    --output_dir ../../data/features/per_residue \
    --feature_type per_residue \
    --batch_size 8 \
    --device cuda   # 若无 GPU 可改为 cpu
```
参数说明：

--input_dir：包含 train/val/test.fasta 的目录

--output_dir：特征保存目录（会自动创建）

--feature_type：per_residue（每个位置保留）或 global（整条序列池化）

--model_name：可更换其他 ESM‑2 变体（如 facebook/esm2_t6_8M_UR50D）

首次运行会从 HuggingFace 下载模型（约 140 MB），请保持网络畅通。


### 5. 训练模型
```bash
cd ../..
python src/train.py \
    --features_dir data/features/per_residue \
    --output_dir checkpoints \
    --batch_size 16 \
    --epochs 100 \
    --lr 5e-4 \
    --max_length 128 \
    --device cuda
```
训练过程中会：

自动计算类别权重（平衡正负样本）

保存验证集 F1 最高的模型（best_model_epoch*.pth）

每 5 个 epoch 保存一次检查点

训练结束后在测试集上评估并生成 training_history.png


### 6. 评估与预测
使用训练好的最佳模型对测试集进行评估：

```bash
python src/predict.py \
    --model_path checkpoints/best_model_epoch14.pth \
    --features_dir data/features/per_residue \
    --max_length 128 \
    --device cuda
```
该脚本会：

输出测试集的整体准确率、各类别准确率

打印前 5 个样本的详细预测结果

若特征目录中包含真实标签，自动计算混淆矩阵（需自行扩展，当前脚本已支持）

如果要对单条序列进行预测，可以参考 predict.py 中的 predict_single() 方法。


## ⚙️ 主要参数配置
```bash
参数	默认值	说明
--max_length	128	序列最大长度（长于截断，短则填充）
--input_dim	480	ESM‑2_t12 的输出维度
--cnn_channels	64	卷积层通道数
--lstm_hidden	128	BiLSTM 隐藏单元数
--dropout	0.5	全局 Dropout 比率
--lr	5e-4	初始学习率
--weight_decay	1e-4	L2 正则化系数
完整参数请查看各脚本的 argparse 定义。
```


## 📝 依赖环境
```bash
Python 3.8+

PyTorch 1.9+

Transformers 4.20+

其他见 requirements.txt
```




