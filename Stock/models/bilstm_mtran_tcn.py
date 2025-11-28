"""
BiLSTM-MTRAN-TCN 混合神经网络模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.positional_encoding import PositionalEncoding
from modules.bilstm_layer import BiLSTMLayer
from modules.mtran_encoder import MTRANEncoder
from modules.tcn_decoder import TCNDecoder
from modules.output_layer import OutputLayer
from modules.series_decomposition import SeriesDecomposition
from core.logger import get_logger


class BiLSTM_MTRAN_TCN(nn.Module):
    """BiLSTM-MTRAN-TCN 混合神经网络模型"""
    
    def __init__(self, config=None):
        from core.config_manager import get_config
        if config is None:
            config = get_config()

        # 从配置中获取所有参数
        input_features = config.model.input_features
        seq_len = config.model.seq_len
        bilstm_hidden = config.model.bilstm_hidden
        bilstm_layers = config.model.bilstm_layers
        transformer_heads = config.model.transformer_heads
        transformer_layers = config.model.transformer_layers
        tcn_channels = config.model.tcn_channels
        tcn_kernel_size = config.model.tcn_kernel_size
        dropout = config.model.dropout
        decomp_kernel_size = config.model.decomp_kernel_size
        autocorr_factor = config.model.autocorr_factor
        super(BiLSTM_MTRAN_TCN, self).__init__()
        
        self.input_features = input_features
        self.seq_len = seq_len
        self.bilstm_hidden = bilstm_hidden

        self.positional_encoding = PositionalEncoding(
            d_model=input_features, max_len=seq_len, dropout=dropout
        )

        self.bilstm = BiLSTMLayer(
            input_size=input_features, hidden_size=bilstm_hidden,
            num_layers=bilstm_layers, dropout=dropout
        )

        self.series_decomp = SeriesDecomposition(kernel_size=decomp_kernel_size)

        self.mtran_encoder_seasonal = MTRANEncoder(
            d_model=bilstm_hidden * 2, num_heads=transformer_heads,
            num_layers=transformer_layers, d_ff=512, dropout=dropout,
            autocorr_factor=autocorr_factor
        )

        self.mtran_encoder_trend = MTRANEncoder(
            d_model=bilstm_hidden * 2, num_heads=transformer_heads,
            num_layers=transformer_layers, d_ff=512, dropout=dropout,
            autocorr_factor=autocorr_factor
        )

        self.tcn_decoder_seasonal = TCNDecoder(
            input_size=bilstm_hidden * 2, num_channels=tcn_channels,
            kernel_size=tcn_kernel_size, dropout=dropout
        )

        self.tcn_decoder_trend = TCNDecoder(
            input_size=bilstm_hidden * 2, num_channels=tcn_channels,
            kernel_size=tcn_kernel_size, dropout=dropout
        )

        self.fusion_layer = nn.Linear(tcn_channels[-1] * 2, tcn_channels[-1])
        self.fusion_dropout = nn.Dropout(dropout)

        self.output_layer = OutputLayer(
            input_size=tcn_channels[-1], seq_len=seq_len,
            hidden_size=64, output_size=1, dropout=dropout
        )

        self.init_weights()
    
    def init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x):
        """前向传播"""
        x = self.positional_encoding(x)
        x = self.bilstm(x)

        seasonal, trend = self.series_decomp(x)

        seasonal_encoded = self.mtran_encoder_seasonal(seasonal)
        seasonal_decoded = self.tcn_decoder_seasonal(seasonal_encoded)

        trend_encoded = self.mtran_encoder_trend(trend)
        trend_decoded = self.tcn_decoder_trend(trend_encoded)

        combined_features = torch.cat([seasonal_decoded, trend_decoded], dim=-1)
        fused_features = self.fusion_layer(combined_features)
        fused_features = self.fusion_dropout(fused_features)

        output = self.output_layer(fused_features)
        return output
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'BiLSTM-MTRAN-TCN',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_shape': f'(batch_size, {self.seq_len}, {self.input_features})',
            'output_shape': '(batch_size, 1)',
            'modules': {
                'positional_encoding': 'Sinusoidal Position Encoding',
                'bilstm': f'{self.bilstm.num_layers} layers, {self.bilstm_hidden} hidden units',
                'series_decomposition': 'Trend and Seasonal Decomposition',
                'mtran_encoder_seasonal': f'{self.mtran_encoder_seasonal.num_layers} layers (seasonal)',
                'mtran_encoder_trend': f'{self.mtran_encoder_trend.num_layers} layers (trend)',
                'tcn_decoder_seasonal': '4 layers, kernel_size=7 (seasonal)',
                'tcn_decoder_trend': '4 layers, kernel_size=7 (trend)',
                'fusion_layer': 'Feature Fusion Layer',
                'output_layer': 'Fully Connected with Tanh activation'
            }
        }


class ModelTrainer:
    """模型训练器"""

    def __init__(self, model, device='cuda', config=None):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.logger = get_logger("model_trainer")

        if config:
            lr = config.training.learning_rate
            weight_decay = config.training.weight_decay
        else:
            lr = 1e-5
            weight_decay = 0

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay
        )

        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.detach()

        avg_loss = (total_loss / num_batches).item()
        return avg_loss
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)

        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.detach()

        avg_loss = (total_loss / num_batches).item()
        return avg_loss
    
    def calculate_metrics(self, y_true, y_pred):
        """计算评价指标"""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}


if __name__ == "__main__":
    from core.config_manager import get_config

    logger = get_logger("model_test")
    config = get_config()

    model = BiLSTM_MTRAN_TCN(config)

    model_info = model.get_model_info()
    logger.log_model_info(model_info)

    batch_size = config.training.batch_size
    seq_len = config.model.seq_len
    features = config.model.input_features

    input_data = torch.randn(batch_size, seq_len, features)
    logger.info(f"输入维度: {input_data.shape}")

    with torch.no_grad():
        output = model(input_data)

    logger.info(f"输出维度: {output.shape}")
    logger.info(f"模型测试成功！")

    expected_output_dim = (batch_size, 1)
    logger.info(f"期望输出维度: {expected_output_dim}")
    logger.info(f"维度匹配: {output.shape == expected_output_dim}")
    logger.info(f"输出值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")