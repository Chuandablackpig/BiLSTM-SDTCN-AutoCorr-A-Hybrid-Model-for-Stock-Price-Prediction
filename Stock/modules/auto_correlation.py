"""
Auto-Correlation机制模块
基于Autoformer的自相关机制，用于替换传统的多头注意力
主要特点：
1. 基于周期性依赖发现的自相关计算
2. 时间延迟聚合机制
3. 更适合时间序列数据的长期依赖建模
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation机制核心实现
    
    两个主要阶段：
    1. 基于周期的依赖发现 (period-based dependencies discovery)
    2. 时间延迟聚合 (time delay aggregation)
    
    参数说明：
    - factor: 控制top-k选择的因子，top_k = factor * log(length)
    - scale: 缩放因子
    - attention_dropout: dropout比例
    - output_attention: 是否输出注意力权重
    """
    
    def __init__(self, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        训练阶段的时间延迟聚合
        使用批归一化风格的设计
        
        输入：
        - values: (batch_size, num_heads, d_k, seq_len)
        - corr: 自相关结果
        
        输出：
        - delays_agg: 聚合后的结果
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        
        # 找到top-k个最重要的延迟
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        
        # 更新相关性权重
        tmp_corr = torch.softmax(weights, dim=-1)
        
        # 时间延迟聚合
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            # 根据延迟进行时间偏移
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        推理阶段的时间延迟聚合
        
        输入：
        - values: (batch_size, num_heads, d_k, seq_len)
        - corr: 自相关结果
        
        输出：
        - delays_agg: 聚合后的结果
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        
        # 初始化索引
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            .repeat(batch, head, channel, 1).to(values.device)
        
        # 找到top-k个最重要的延迟
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        
        # 更新相关性权重
        tmp_corr = torch.softmax(weights, dim=-1)
        
        # 时间延迟聚合
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask=None):
        """
        Auto-Correlation前向传播
        
        输入：
        - queries: (batch_size, seq_len, num_heads, d_k)
        - keys: (batch_size, seq_len, num_heads, d_k)
        - values: (batch_size, seq_len, num_heads, d_k)
        - attn_mask: 注意力掩码
        
        输出：
        - V: 自相关结果 (batch_size, seq_len, num_heads, d_k)
        - corr: 相关性权重（如果output_attention=True）
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        
        # 处理序列长度不匹配的情况
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # 阶段1: 基于周期的依赖发现
        # 使用FFT计算自相关
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, n=L, dim=-1)

        # 阶段2: 时间延迟聚合
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return V.contiguous(), corr.permute(0, 3, 1, 2)
        else:
            return V.contiguous(), None


class AutoCorrelationLayer(nn.Module):
    """
    Auto-Correlation层，包装AutoCorrelation机制
    类似于MultiHeadAttention的接口设计
    
    参数说明：
    - d_model: 模型维度
    - n_heads: 注意力头数
    - factor: AutoCorrelation的factor参数
    - dropout: dropout比例
    """
    
    def __init__(self, d_model, n_heads, factor=1, dropout=0.1, output_attention=False):
        super(AutoCorrelationLayer, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_keys = d_model // n_heads
        self.d_values = d_model // n_heads

        # AutoCorrelation核心机制
        self.inner_correlation = AutoCorrelation(
            factor=factor, 
            attention_dropout=dropout, 
            output_attention=output_attention
        )
        
        # 线性投影层
        self.query_projection = nn.Linear(d_model, self.d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, self.d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, self.d_values * n_heads)
        self.out_projection = nn.Linear(self.d_values * n_heads, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        前向传播
        
        输入：
        - x: (batch_size, seq_len, d_model)
        - mask: 注意力掩码
        
        输出：
        - output: (batch_size, seq_len, d_model)
        """
        B, L, _ = x.shape
        H = self.n_heads

        # 对于自注意力，queries, keys, values都来自同一个输入
        queries = keys = values = x
        
        # 线性投影
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, L, H, -1)
        values = self.value_projection(values).view(B, L, H, -1)

        # AutoCorrelation计算
        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            mask
        )
        
        # 重塑并输出投影
        out = out.view(B, L, -1)
        output = self.out_projection(out)
        output = self.dropout(output)

        return output
