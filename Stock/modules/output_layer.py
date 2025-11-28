"""
模块 5：输出层（全连接层）
功能：对TCN输出的高维特征降维，输出股票次日收盘价预测值
输入维度：(batch_size, seq_len=10, tcn_hidden=32)
输出维度：(batch_size, 1) - 次日收盘价预测结果
"""

import torch
import torch.nn as nn


class OutputLayer(nn.Module):
    """
    输出层 - 将TCN输出转换为股票价格预测
    
    参数说明：
    - input_size: TCN输出特征维度 (32)
    - seq_len: 序列长度 (10)
    - hidden_size: 中间隐藏层维度 (64，可选)
    - output_size: 输出维度 (1，单个预测值)
    - dropout: dropout比例 (0.1)
    
    处理逻辑：
    1. 将3D张量(batch_size, seq_len, tcn_hidden)展平为2D
    2. 通过全连接层降维
    3. 使用Tanh激活函数映射到合理价格波动范围
    4. 输出单个预测值
    """
    
    def __init__(self, input_size=32, seq_len=10, hidden_size=64, output_size=1, dropout=0.1):
        super(OutputLayer, self).__init__()
        
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 计算展平后的输入维度
        flattened_size = seq_len * input_size  # 10 * 32 = 320
        
        # 多层全连接网络
        self.fc_layers = nn.Sequential(
            # 第一层：展平维度 -> 隐藏层
            nn.Linear(flattened_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # 第二层：隐藏层 -> 输出层
            nn.Linear(hidden_size, output_size),
            nn.Tanh()  # Tanh激活函数，映射到[-1, 1]范围
        )
        
        # 可选：额外的归一化层
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """
        初始化网络权重
        """
        for module in self.fc_layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        输入：
        - x: TCN输出 (batch_size, seq_len, input_size) = (batch_size, 10, 32)
        
        输出：
        - 次日收盘价预测: (batch_size, 1)
        """
        batch_size = x.size(0)
        
        # 展平TCN输出：(batch_size, seq_len, input_size) -> (batch_size, seq_len * input_size)
        x_flattened = x.view(batch_size, -1)  # (batch_size, 320)
        
        # 通过全连接层
        output = self.fc_layers(x_flattened)  # (batch_size, 1)
        
        return output


class AdvancedOutputLayer(nn.Module):
    """
    高级输出层 - 使用注意力机制聚合时序信息
    
    这是一个可选的更复杂的输出层实现，使用注意力机制来
    自适应地聚合不同时间步的信息，而不是简单的展平操作。
    """
    
    def __init__(self, input_size=32, seq_len=10, hidden_size=64, output_size=1, dropout=0.1):
        super(AdvancedOutputLayer, self).__init__()
        
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )
    
    def forward(self, x):
        """
        前向传播 - 使用注意力机制
        
        输入：
        - x: TCN输出 (batch_size, seq_len, input_size) = (batch_size, 10, 32)
        
        输出：
        - 次日收盘价预测: (batch_size, 1)
        """
        # 计算注意力权重
        attention_weights = self.attention(x)  # (batch_size, seq_len, 1)
        
        # 加权聚合
        weighted_features = torch.sum(x * attention_weights, dim=1)  # (batch_size, input_size)
        
        # 输出预测
        output = self.output_layer(weighted_features)  # (batch_size, 1)
        
        return output
