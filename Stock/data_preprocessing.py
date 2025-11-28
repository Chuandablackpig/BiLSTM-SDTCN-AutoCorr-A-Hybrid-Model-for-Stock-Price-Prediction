"""
数据预处理模块
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import sys
import os

warnings.filterwarnings('ignore')

from core.logger import get_logger


class StockDataset(Dataset):
    """股票数据集类"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class StockDataPreprocessor:
    """股票数据预处理器"""

    def __init__(self, config=None):
        from core.config_manager import get_config
        if config is None:
            config = get_config()

        self.config = config
        self.window_size = config.data.window_size
        self.target_column = config.data.target_column
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.logger = get_logger("data_preprocessor")

        self.feature_columns = [
            '收盘', '最高', '最低', '开盘', '涨跌额',
            '涨跌幅', '成交量', '成交额', 'MA5', 'MA10'
        ]
    
    def load_data(self, file_path):
        """加载股票数据"""
        self.logger.info(f"正在加载数据: {file_path}")

        df = pd.read_csv(file_path, encoding='utf-8')

        self.logger.info(f"原始数据形状: {df.shape}")
        self.logger.info(f"原始数据列名: {list(df.columns)}")
        self.logger.info(f"数据时间范围: {df['日期'].min()} 到 {df['日期'].max()}")

        df = self.clean_data(df)

        self.logger.info(f"清洗后数据形状: {df.shape}")
        return df
    
    def clean_data(self, df):
        """数据清洗"""
        df = df.copy()
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期').reset_index(drop=True)

        missing_columns = [col for col in self.feature_columns if col not in df.columns]
        if missing_columns:
            self.logger.warning(f"缺失特征列: {missing_columns}")
            column_mapping = {
                '收盘': ['close', 'Close', '收盘价'],
                '最高': ['high', 'High', '最高价'],
                '最低': ['low', 'Low', '最低价'],
                '开盘': ['open', 'Open', '开盘价'],
                '成交量': ['volume', 'Volume', '成交量'],
                '成交额': ['amount', 'Amount', '成交额', '成交金额']
            }

            for target_col, possible_names in column_mapping.items():
                if target_col not in df.columns:
                    for possible_name in possible_names:
                        if possible_name in df.columns:
                            df[target_col] = df[possible_name]
                            self.logger.info(f"映射列名: {possible_name} -> {target_col}")
                            break

        if '涨跌额' not in df.columns:
            df['涨跌额'] = df['收盘'].diff()
            self.logger.info("计算涨跌额")

        if '涨跌幅' not in df.columns:
            df['涨跌幅'] = df['收盘'].pct_change() * 100
            self.logger.info("计算涨跌幅")

        if 'MA5' not in df.columns:
            df['MA5'] = df['收盘'].rolling(window=5, min_periods=1).mean()
            self.logger.info("计算5日移动平均线")

        if 'MA10' not in df.columns:
            df['MA10'] = df['收盘'].rolling(window=10, min_periods=1).mean()
            self.logger.info("计算10日移动平均线")

        for col in self.feature_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df
    
    def create_sequences(self, df):
        """创建时间序列数据"""
        self.logger.info("正在创建时间序列数据...")

        feature_data = df[self.feature_columns].values
        target_data = df[self.target_column].values

        X, y = [], []

        for i in range(len(feature_data) - self.window_size):
            X.append(feature_data[i:i + self.window_size])
            y.append(target_data[i + self.window_size])

        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        self.logger.info(f"序列数据形状: X={X.shape}, y={y.shape}")
        return X, y
    
    def normalize_data(self, X_train, X_test, y_train, y_test):
        """Z-score标准化"""
        self.logger.info("正在进行时序Z-score标准化...")

        n_samples_train, n_timesteps, n_features = X_train.shape
        n_samples_test = X_test.shape[0]

        X_train_reshaped = X_train.reshape(-1, n_features)
        X_test_reshaped = X_test.reshape(-1, n_features)

        self.scaler_X.fit(X_train_reshaped)
        self.logger.info(f"特征均值: {self.scaler_X.mean_[:3]}...")
        self.logger.info(f"特征标准差: {self.scaler_X.scale_[:3]}...")

        X_train_scaled = self.scaler_X.transform(X_train_reshaped)
        X_test_scaled = self.scaler_X.transform(X_test_reshaped)

        X_train_scaled = X_train_scaled.reshape(n_samples_train, n_timesteps, n_features)
        X_test_scaled = X_test_scaled.reshape(n_samples_test, n_timesteps, n_features)

        self.scaler_y.fit(y_train)
        self.logger.info(f"目标均值: {self.scaler_y.mean_[0]:.6f}")
        self.logger.info(f"目标标准差: {self.scaler_y.scale_[0]:.6f}")

        y_train_scaled = self.scaler_y.transform(y_train)
        y_test_scaled = self.scaler_y.transform(y_test)

        self.logger.info("时序标准化完成")
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled
    
    def inverse_transform_target(self, y_scaled):
        """反标准化目标变量"""
        return self.scaler_y.inverse_transform(y_scaled)
    
    def prepare_data(self, file_path, test_size=0.2, random_state=42):
        """完整的数据准备流程"""
        df = self.load_data(file_path)

        total_len = len(df)
        train_size = int(total_len * (1 - test_size))

        self.logger.info(f"数据时间范围: {df['日期'].min()} 到 {df['日期'].max()}")
        self.logger.info(f"训练集时间范围: {df['日期'].iloc[0]} 到 {df['日期'].iloc[train_size-1]}")
        self.logger.info(f"测试集时间范围: {df['日期'].iloc[train_size]} 到 {df['日期'].iloc[-1]}")

        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()

        X_train, y_train = self.create_sequences(train_df)
        X_test, y_test = self.create_sequences(test_df)

        self.logger.info(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
        self.logger.info(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")

        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = self.normalize_data(
            X_train, X_test, y_train, y_test
        )

        train_dataset = StockDataset(X_train_scaled, y_train_scaled)
        test_dataset = StockDataset(X_test_scaled, y_test_scaled)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            pin_memory=self.config.data.pin_memory,
            num_workers=self.config.data.num_workers,
            persistent_workers=self.config.data.persistent_workers
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            pin_memory=self.config.data.pin_memory,
            num_workers=self.config.data.num_workers,
            persistent_workers=self.config.data.persistent_workers
        )

        data_info = {
            'total_samples': len(X_train) + len(X_test),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': len(self.feature_columns),
            'window_size': self.window_size,
            'feature_names': self.feature_columns,
            'data_shape': {
                'input': f"(batch_size, {self.window_size}, {len(self.feature_columns)})",
                'output': "(batch_size, 1)"
            }
        }

        self.logger.log_section("数据准备完成")
        for key, value in data_info.items():
            self.logger.info(f"{key}: {value}")

        return train_loader, test_loader, data_info


if __name__ == "__main__":
    from core.config_manager import get_config
    config = get_config()
    logger = get_logger("data_preprocessing_test")

    preprocessor = StockDataPreprocessor(config)
    data_path = config.paths.stock_data_file

    try:
        train_loader, test_loader, data_info = preprocessor.prepare_data(
            file_path=data_path,
            test_size=0.2,
            random_state=42
        )

        logger.log_section("数据加载器测试")
        for batch_idx, (data, target) in enumerate(train_loader):
            logger.info(f"批次 {batch_idx + 1}:")
            logger.info(f"  输入形状: {data.shape}")
            logger.info(f"  目标形状: {target.shape}")
            if batch_idx >= 2:
                break

        logger.info("数据预处理成功！")
        logger.info(f"训练批次数: {len(train_loader)}")
        logger.info(f"测试批次数: {len(test_loader)}")

    except Exception as e:
        logger.log_exception(e, "数据预处理")