"""
模块 2：BiLSTM 层
功能：捕捉股票序列的双向时间依赖关系，缓解长序列梯度衰减问题
输入维度：(batch_size, seq_len=10, features=8)
输出维度：(batch_size, seq_len=10, hidden_size*2=128)
"""

import torch
import torch.nn as nn


class BiLSTMLayer(nn.Module):
    """
    双向LSTM层 - 3层堆叠结构
    
    参数说明：
    - input_size: 输入特征维度 (8)
    - hidden_size: 每层隐藏单元数 (64)
    - num_layers: LSTM层数 (3)
    - dropout: dropout比例 (0.2)
    - bidirectional: 是否双向 (True)
    
    处理逻辑：
    1. 3层堆叠的双向LSTM
    2. 每层64个神经元，双向后输出128维
    3. 使用ReLU激活函数
    4. 添加残差连接和dropout防止过拟合
    """
    
    def __init__(self, input_size=8, hidden_size=64, num_layers=3, dropout=0.2):
        super(BiLSTMLayer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 双向LSTM层
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # ReLU激活函数
        self.relu = nn.ReLU()
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 残差连接的线性变换层（将输入8维映射到输出128维）
        self.residual_projection = nn.Linear(input_size, hidden_size * 2)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
    def forward(self, x):
        """
        前向传播
        
        输入：
        - x: (batch_size, seq_len, input_size) = (batch_size, 10, 8)
        
        输出：
        - 双向隐藏状态拼接: (batch_size, 10, 128)
        """
        # 保存输入用于残差连接
        residual = x  # (batch_size, seq_len, 8)
        
        # BiLSTM前向传播
        # lstm_out: (batch_size, seq_len, hidden_size*2) = (batch_size, 10, 128)
        # hidden: (num_layers*2, batch_size, hidden_size)
        lstm_out, (hidden, cell) = self.bilstm(x)
        
        # 应用ReLU激活函数
        lstm_out = self.relu(lstm_out)
        
        # 残差连接：将输入投影到与输出相同的维度
        residual_projected = self.residual_projection(residual)  # (batch_size, 10, 128)
        
        # 残差连接
        output = lstm_out + residual_projected
        
        # 层归一化
        output = self.layer_norm(output)
        
        # Dropout
        output = self.dropout(output)
        
        return output
    
    def init_hidden(self, batch_size, device):
        """
        初始化隐藏状态和细胞状态
        
        参数：
        - batch_size: 批大小
        - device: 设备 (cpu/cuda)
        
        返回：
        - hidden: (num_layers*2, batch_size, hidden_size)
        - cell: (num_layers*2, batch_size, hidden_size)
        """
        hidden = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        return hidden, cell
