import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import os


class ProteinSequenceDataset(Dataset):
    def __init__(self, features_dir, split="train", max_length=128, pad_to_max=True):
        """
        加载ESM-2提取的特征数据

        Args:
            features_dir: 包含.pkl文件的目录
            split: 数据集划分 (train/val/test)
            max_length: 最大序列长度
            pad_to_max: 是否填充到固定长度
        """
        self.max_length = max_length
        self.pad_to_max = pad_to_max

        # 加载特征
        embeddings_path = os.path.join(features_dir, f"{split}_embeddings.pkl")
        metadata_path = os.path.join(features_dir, f"{split}_metadata.pkl")

        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"特征文件不存在: {embeddings_path}")

        # 加载特征和元数据
        with open(embeddings_path, "rb") as f:
            self.embeddings = pickle.load(f)

        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
            self.labels = metadata["labels"]

        print(f"加载 {split} 数据集: {len(self.embeddings)} 个样本")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        # 获取原始特征
        emb = self.embeddings[idx]  # 形状: (seq_len, feature_dim)

        # 确保是numpy数组
        if isinstance(emb, torch.Tensor):
            emb = emb.numpy()

        seq_len = emb.shape[0]

        # 处理不同长度序列
        if seq_len < self.max_length and self.pad_to_max:
            # 填充到max_length
            padding = np.zeros((self.max_length - seq_len, emb.shape[1]))
            emb_padded = np.vstack([emb, padding])
        elif seq_len > self.max_length:
            # 截断到max_length
            emb_padded = emb[:self.max_length, :]
        else:
            emb_padded = emb

        # 转换为tensor
        features_tensor = torch.FloatTensor(emb_padded)
        label_tensor = torch.LongTensor([self.labels[idx]])

        return features_tensor, label_tensor, seq_len