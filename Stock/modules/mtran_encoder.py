"""
模块 3：改进 Transformer 编码器（MTRAN）
功能：提取股票序列的全局关联信息，关注重要特征（成交量、涨跌幅等）
输入维度：(batch_size, seq_len=10, hidden_size=128)
输出维度：(batch_size, seq_len=10, hidden_size=128)

更新：集成Autoformer的Auto-Correlation机制替换传统多头注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

# 添加auto_correlation模块
sys.path.append(os.path.dirname(__file__))
from auto_correlation import AutoCorrelationLayer


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    
    参数说明：
    - d_model: 模型维度 (128)
    - num_heads: 注意力头数 (8)
    - dropout: dropout比例 (0.1)
    
    注意力计算公式：
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
    其中 d_k = d_model / num_heads = 128 / 8 = 16
    """
    
    def __init__(self, d_model=128, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 16
        
        # 线性变换层：Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力计算
        
        输入：
        - Q, K, V: (batch_size, num_heads, seq_len, d_k)
        - mask: 注意力掩码
        
        输出：
        - attention_output: (batch_size, num_heads, seq_len, d_k)
        - attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        # 计算注意力分数: QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax归一化
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算注意力输出
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        输入：
        - x: (batch_size, seq_len, d_model) = (batch_size, 10, 128)
        
        输出：
        - output: (batch_size, seq_len, d_model) = (batch_size, 10, 128)
        """
        batch_size, seq_len, d_model = x.size()
        
        # 线性变换得到Q, K, V
        Q = self.w_q(x)  # (batch_size, seq_len, d_model)
        K = self.w_k(x)
        V = self.w_v(x)
        
        # 重塑为多头形式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # 现在形状为: (batch_size, num_heads, seq_len, d_k)
        
        # 计算注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 重塑回原始形状
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        # 最终线性变换
        output = self.w_o(attention_output)
        
        return output


class PositionwiseFeedForward(nn.Module):
    """
    位置前馈网络
    
    参数说明：
    - d_model: 模型维度 (128)
    - d_ff: 前馈网络隐藏层维度 (512)
    - dropout: dropout比例 (0.1)
    """
    
    def __init__(self, d_model=128, d_ff=512, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        前向传播：Linear -> ReLU -> Dropout -> Linear
        
        输入：
        - x: (batch_size, seq_len, d_model)
        
        输出：
        - output: (batch_size, seq_len, d_model)
        """
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层 - 集成Auto-Correlation机制

    结构：
    1. Auto-Correlation自相关机制 + 残差连接 + 层归一化
    2. 前馈网络 + 残差连接 + 层归一化
    3. 额外的残差连接用于增强梯度流动
    """

    def __init__(self, d_model=128, num_heads=8, d_ff=512, dropout=0.1, autocorr_factor=1):
        super(TransformerEncoderLayer, self).__init__()

        # 使用Auto-Correlation替换传统多头注意力
        self.auto_correlation = AutoCorrelationLayer(
            d_model=d_model, 
            n_heads=num_heads, 
            factor=autocorr_factor,
            dropout=dropout
        )
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)  # 额外的层归一化

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 额外的残差连接层（用于深层网络的梯度流动）
        self.residual_projection = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """
        前向传播 - 增强的残差连接架构

        输入：
        - x: (batch_size, seq_len, d_model)

        输出：
        - output: (batch_size, seq_len, d_model)
        """
        # 保存原始输入用于深层残差连接
        identity = x

        # 第一个子层：Auto-Correlation + 残差连接 + 层归一化
        autocorr_output = self.auto_correlation(x, mask)
        x = self.norm1(x + self.dropout(autocorr_output))

        # 第二个子层：前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        # 第三个子层：深层残差连接（跨越整个编码器层）
        # 这有助于在深层网络中保持梯度流动
        residual_output = self.residual_projection(identity)
        x = self.norm3(x + self.dropout(residual_output))

        return x


class MTRANEncoder(nn.Module):
    """
    改进的Transformer编码器（MTRAN）- 集成Auto-Correlation机制

    参数说明：
    - d_model: 模型维度 (128)
    - num_heads: 注意力头数 (8)
    - num_layers: 编码器层数 (6)
    - d_ff: 前馈网络隐藏层维度 (512)
    - dropout: dropout比例 (0.1)
    - autocorr_factor: Auto-Correlation的factor参数

    改进点：
    - 删除原生Transformer的Input Embedding模块
    - 直接接收BiLSTM的输出作为输入
    - 使用Auto-Correlation替换传统多头注意力
    - 增强的残差连接架构
    """

    def __init__(self, d_model=128, num_heads=8, num_layers=6, d_ff=512, dropout=0.1, autocorr_factor=1):
        super(MTRANEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # 6层Transformer编码器层堆叠（集成Auto-Correlation）
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, autocorr_factor)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # 全局残差连接（跨越整个编码器）
        self.global_residual = nn.Linear(d_model, d_model)
        self.global_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        前向传播 - 增强的残差连接架构

        输入：
        - x: BiLSTM输出 (batch_size, seq_len, d_model) = (batch_size, 10, 128)

        输出：
        - 含全局信息的张量: (batch_size, 10, 128)
        """
        # 保存输入用于全局残差连接
        global_identity = x

        # 应用dropout
        x = self.dropout(x)

        # 通过6层编码器层
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)

        # 全局残差连接（跨越整个编码器）
        # 这有助于在非常深的网络中保持梯度流动和特征传递
        global_residual_output = self.global_residual(global_identity)
        x = self.global_norm(x + self.dropout(global_residual_output))

        return x