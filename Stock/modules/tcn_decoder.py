"""
模块 4：TCN解码器（替换Transformer解码器）
功能：强化长时序列依赖捕捉，弥补Transformer序列信息弱的缺陷，避免未来信息泄露
输入维度：(batch_size, seq_len=10, hidden_size=128)
输出维度：(batch_size, seq_len=10, tcn_hidden=32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class Chomp1d(nn.Module):
    """
    裁剪层 - 用于因果卷积，确保不使用未来信息
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        裁剪右侧填充，保证因果性
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    TCN时间块 - 包含膨胀因果卷积、权重归一化、残差连接等
    
    参数说明：
    - n_inputs: 输入通道数
    - n_outputs: 输出通道数  
    - kernel_size: 卷积核大小 (7)
    - stride: 步长 (1)
    - dilation: 膨胀系数 (1, 2, 4, 8)
    - padding: 填充大小
    - dropout: dropout比例
    
    核心组件：
    1. 膨胀因果卷积
    2. 权重归一化
    3. ReLU激活
    4. Dropout
    5. 残差连接
    """
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # 第一个膨胀因果卷积层
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)  # 裁剪保证因果性
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # 第二个膨胀因果卷积层
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # 组合网络
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # 残差连接的降采样层（如果输入输出维度不同）
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
        # ReLU激活
        self.relu = nn.ReLU()
        
        self.init_weights()
    
    def init_weights(self):
        """
        初始化权重
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        """
        前向传播
        
        输入：
        - x: (batch_size, n_inputs, seq_len)
        
        输出：
        - output: (batch_size, n_outputs, seq_len)
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)  # 残差连接


class TemporalConvNet(nn.Module):
    """
    时间卷积网络（TCN）
    
    参数说明：
    - num_inputs: 输入通道数 (128)
    - num_channels: 各层通道数列表 [32, 32, 32, 32] (4层，每层32个神经元)
    - kernel_size: 卷积核大小 (7)
    - dropout: dropout比例 (0.2)
    
    膨胀系数：随层数指数增长 (1 → 2 → 4 → 8)
    """
    
    def __init__(self, num_inputs, num_channels, kernel_size=7, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1, 2, 4, 8
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # 计算填充大小以保持序列长度
            padding = (kernel_size - 1) * dilation_size
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 
                                   stride=1, dilation=dilation_size, 
                                   padding=padding, dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播
        
        输入：
        - x: (batch_size, num_inputs, seq_len)
        
        输出：
        - output: (batch_size, num_channels[-1], seq_len)
        """
        return self.network(x)


class TCNDecoder(nn.Module):
    """
    TCN解码器 - 替换Transformer解码器
    
    结构：
    1. 输入维度变换：(batch_size, seq_len, 128) -> (batch_size, 128, seq_len)
    2. TCN层：4层，每层32个神经元，卷积核大小7，膨胀系数1→2→4→8
    3. 全连接层：维度过渡
    4. Tanh激活函数
    5. 输出维度变换：(batch_size, 32, seq_len) -> (batch_size, seq_len, 32)
    
    参数说明：
    - input_size: 输入特征维度 (128)
    - num_channels: TCN各层通道数 [32, 32, 32, 32]
    - kernel_size: 卷积核大小 (7)
    - dropout: dropout比例 (0.2)
    """
    
    def __init__(self, input_size=128, num_channels=[32, 32, 32, 32], 
                 kernel_size=7, dropout=0.2):
        super(TCNDecoder, self).__init__()
        
        self.input_size = input_size
        self.output_size = num_channels[-1]  # 32
        
        # TCN网络
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        
        # 全连接层（可选的维度过渡层）
        self.fc = nn.Linear(num_channels[-1], num_channels[-1])
        
        # Tanh激活函数
        self.tanh = nn.Tanh()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        前向传播
        
        输入：
        - x: MTRAN编码器输出 (batch_size, seq_len, input_size) = (batch_size, 10, 128)
        
        输出：
        - TCN处理后的张量: (batch_size, seq_len, output_size) = (batch_size, 10, 32)
        """
        # 维度变换：(batch_size, seq_len, input_size) -> (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)  # (batch_size, 128, 10)
        
        # TCN前向传播
        tcn_out = self.tcn(x)  # (batch_size, 32, 10)
        
        # 维度变换回来：(batch_size, output_size, seq_len) -> (batch_size, seq_len, output_size)
        tcn_out = tcn_out.transpose(1, 2)  # (batch_size, 10, 32)
        
        # 全连接层
        fc_out = self.fc(tcn_out)  # (batch_size, 10, 32)
        
        # Tanh激活函数
        output = self.tanh(fc_out)
        
        # Dropout
        output = self.dropout(output)
        
        return output