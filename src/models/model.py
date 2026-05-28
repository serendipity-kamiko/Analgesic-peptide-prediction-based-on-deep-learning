import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class SimplifiedProteinModel(nn.Module):
    def __init__(self, input_dim, cnn_channels=64, lstm_hidden=128,
                 num_classes=2, dropout=0.5):
        """
        简化的蛋白质序列分类模型
        移除Transformer层，减少参数数量，增强正则化

        Args:
            input_dim: 输入特征维度 (ESM-2: 480)
            cnn_channels: CNN输出通道数
            lstm_hidden: LSTM隐藏层维度
            num_classes: 分类数量
            dropout: Dropout率
        """
        super().__init__()

        # 1. 简化CNN部分
        self.cnn = nn.Sequential(
            # 第一层卷积
            nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 早期池化减少计算量
            nn.Dropout(dropout * 0.5),  # 较低dropout

            # 可选: 第二层轻量卷积
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 2. 单层BiLSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=1,  # 改为1层减少参数
            batch_first=True,
            bidirectional=True,
            dropout=0  # 单层LSTM不需要dropout
        )

        # 3. 注意力机制 (简化)
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),  # 减小中间维度
            nn.Tanh(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        # 4. 简化全连接层
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),  # 减小维度
            nn.BatchNorm1d(64),  # 添加BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),  # 较高dropout
            nn.Linear(64, num_classes)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """改进的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
                        # 设置LSTM forget gate偏置为1，有助于梯度流动
                        n = param.size(0)
                        param.data[n // 4:n // 2].fill_(1.0)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入特征 (batch, seq_len, input_dim)
        Returns:
            output: 分类输出 (batch, num_classes)
            attention_weights: 注意力权重 (batch, seq_len, 1)
        """
        batch_size, seq_len, feature_dim = x.shape

        # 1. CNN处理
        x_cnn = x.permute(0, 2, 1)  # [batch, feature_dim, seq_len]
        x_cnn = self.cnn(x_cnn)  # [batch, cnn_channels, new_seq_len]
        cnn_seq_len = x_cnn.shape[2]

        # 转回LSTM输入格式
        x_cnn = x_cnn.permute(0, 2, 1)  # [batch, new_seq_len, cnn_channels]

        # 2. BiLSTM处理
        lstm_out, _ = self.lstm(x_cnn)  # [batch, cnn_seq_len, lstm_hidden*2]

        # 3. 注意力机制
        attention_weights = self.attention(lstm_out)  # [batch, cnn_seq_len, 1]
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # [batch, lstm_hidden*2]

        # 4. 分类输出
        output = self.fc(context_vector)

        return output, attention_weights.squeeze(-1)

