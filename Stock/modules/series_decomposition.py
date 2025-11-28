"""
渐进式序列分解模块
基于Autoformer的Series Decomposition思想
功能：
1. 移动平均提取趋势组件
2. 残差提取季节性组件
3. 支持可学习的分解策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAverage(nn.Module):
    """
    移动平均模块，用于提取趋势组件
    """
    def __init__(self, kernel_size, stride=1):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        """
        输入: (batch_size, seq_len, features)
        输出: (batch_size, seq_len, features)
        """
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        
        # 转换维度以适应AvgPool1d: (batch_size, features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        # 转换回原始维度: (batch_size, seq_len, features)
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomposition(nn.Module):
    """
    序列分解模块
    将时间序列分解为趋势组件和季节性组件
    """
    def __init__(self, kernel_size=25):
        super(SeriesDecomposition, self).__init__()
        self.kernel_size = kernel_size
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x):
        """
        输入: (batch_size, seq_len, features)
        输出: 
        - seasonal: 季节性组件 (batch_size, seq_len, features)
        - trend: 趋势组件 (batch_size, seq_len, features)
        """
        # 提取趋势组件（移动平均）
        trend = self.moving_avg(x)
        
        # 提取季节性组件（残差）
        seasonal = x - trend
        
        return seasonal, trend


class AdaptiveSeriesDecomposition(nn.Module):
    """
    自适应序列分解模块
    支持可学习的分解策略
    """
    def __init__(self, kernel_size=25, features=128):
        super(AdaptiveSeriesDecomposition, self).__init__()
        self.kernel_size = kernel_size
        self.features = features
        
        # 基础移动平均
        self.moving_avg = MovingAverage(kernel_size, stride=1)
        
        # 可学习的权重调整
        self.trend_weight = nn.Parameter(torch.ones(1))
        self.seasonal_weight = nn.Parameter(torch.ones(1))
        
        # 特征级别的分解权重
        self.feature_weights = nn.Linear(features, features)
        
    def forward(self, x):
        """
        输入: (batch_size, seq_len, features)
        输出: 
        - seasonal: 季节性组件 (batch_size, seq_len, features)
        - trend: 趋势组件 (batch_size, seq_len, features)
        """
        # 基础移动平均提取趋势
        base_trend = self.moving_avg(x)
        
        # 特征级别的权重调整
        feature_weights = torch.sigmoid(self.feature_weights(x))
        
        # 加权趋势组件
        trend = base_trend * self.trend_weight * feature_weights
        
        # 加权季节性组件
        seasonal = (x - base_trend) * self.seasonal_weight * (1 - feature_weights)
        
        return seasonal, trend


class MultiScaleDecomposition(nn.Module):
    """
    多尺度分解模块
    使用不同的移动平均窗口捕获不同时间尺度的趋势
    """
    def __init__(self, kernel_sizes=[5, 15, 25], features=128):
        super(MultiScaleDecomposition, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.features = features
        
        # 多个移动平均模块
        self.moving_avgs = nn.ModuleList([
            MovingAverage(k) for k in kernel_sizes
        ])
        
        # 融合不同尺度的趋势
        self.trend_fusion = nn.Linear(len(kernel_sizes) * features, features)
        
        # 季节性组件权重
        self.seasonal_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        """
        输入: (batch_size, seq_len, features)
        输出: 
        - seasonal: 季节性组件 (batch_size, seq_len, features)
        - trend: 多尺度融合的趋势组件 (batch_size, seq_len, features)
        """
        batch_size, seq_len, features = x.shape
        
        # 提取多尺度趋势
        trends = []
        for moving_avg in self.moving_avgs:
            trend_i = moving_avg(x)
            trends.append(trend_i)
        
        # 拼接多尺度趋势
        multi_scale_trend = torch.cat(trends, dim=-1)  # (batch_size, seq_len, len(kernel_sizes) * features)
        
        # 融合多尺度趋势
        fused_trend = self.trend_fusion(multi_scale_trend)  # (batch_size, seq_len, features)
        
        # 计算季节性组件
        seasonal = (x - fused_trend) * self.seasonal_weight
        
        return seasonal, fused_trend