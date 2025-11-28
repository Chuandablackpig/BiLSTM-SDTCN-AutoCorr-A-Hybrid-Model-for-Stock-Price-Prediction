"""
股票预测模型训练脚本
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from models.bilstm_mtran_tcn import BiLSTM_MTRAN_TCN, ModelTrainer
from data_preprocessing import StockDataPreprocessor
from financial_metrics import FinancialMetricsCalculator
from core.config_manager import get_config
from core.logger import get_logger, setup_logger


class StockPredictionTrainer:
    """股票预测模型训练器"""

    def __init__(self, config_manager=None):
        self.config = config_manager or get_config()
        self.logger = get_logger("trainer")

        self.device = torch.device(self.config.training.device)
        self.model = None
        self.trainer = None
        self.preprocessor = None
        self.train_loader = None
        self.test_loader = None
        self.data_info = None

        self.financial_calculator = FinancialMetricsCalculator(risk_free_rate=0.03)
        self.train_history = {'train_loss': [], 'val_loss': [], 'epochs': []}

        self.logger.info(f"使用设备: {self.device}")
    
    def setup_data(self):
        """设置数据"""
        self.logger.log_section("数据准备阶段")

        self.preprocessor = StockDataPreprocessor(self.config)

        self.train_loader, self.test_loader, self.data_info = self.preprocessor.prepare_data(
            file_path=self.config.paths.stock_data_file,
            test_size=self.config.data.test_size,
            random_state=self.config.data.random_state
        )

        self.logger.log_data_info(self.data_info)
        self.logger.info("数据准备完成")
    
    def setup_model(self):
        """设置模型"""
        self.logger.log_section("模型初始化阶段")

        self.model = BiLSTM_MTRAN_TCN(self.config)

        self.trainer = ModelTrainer(self.model, device=self.device, config=self.config)

        model_info = self.model.get_model_info()
        self.logger.log_model_info(model_info)
        self.logger.info("模型初始化完成")
    
    def train_model(self):
        """训练模型"""
        epochs = self.config.training.epochs
        save_interval = self.config.training.save_interval
        validation_interval = self.config.training.validation_interval
        early_stopping_patience = self.config.training.early_stopping_patience

        self.logger.log_section(f"模型训练阶段 (共{epochs}轮)")
        self.logger.info(f"保存间隔: {save_interval}, 验证间隔: {validation_interval}")

        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        last_val_loss = float('inf')

        for epoch in range(1, epochs + 1):
            train_loss = self.trainer.train_epoch(self.train_loader)

            if epoch % validation_interval == 0 or epoch == 1:
                val_loss = self.trainer.validate(self.test_loader)
                last_val_loss = val_loss

                # 记录训练历史
                self.train_history['epochs'].append(epoch)
                self.train_history['train_loss'].append(train_loss)
                self.train_history['val_loss'].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model('best_model.pth')
                else:
                    patience_counter += validation_interval

            if epoch % 20 == 0 or epoch == 1:
                elapsed_time = time.time() - start_time
                train_metrics = self._calculate_training_metrics()

                self.logger.log_training_progress(
                    epoch, epochs, train_loss, last_val_loss, elapsed_time, train_metrics
                )

            if epoch % save_interval == 0 and epoch > 0:
                self.save_model(f'model_epoch_{epoch}.pth')

            if patience_counter >= early_stopping_patience:
                self.logger.info(f"早停触发！在第{epoch}轮停止训练")
                break

        total_time = time.time() - start_time
        self.logger.info(f"训练完成！总用时: {total_time:.1f}秒")
        self.logger.info(f"最佳验证损失: {best_val_loss:.6f}")

    def _calculate_training_metrics(self):
        """计算训练阶段的评价指标"""
        self.model.eval()
        y_true_scaled_list = []
        y_pred_scaled_list = []
        y_true_original_list = []
        y_pred_original_list = []

        with torch.no_grad():
            sample_count = 0
            max_samples = 500

            for data, target in self.train_loader:
                if sample_count >= max_samples:
                    break

                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                output = self.model(data)

                # 标准化数据
                y_true_scaled_list.extend(target.cpu().numpy().flatten())
                y_pred_scaled_list.extend(output.cpu().numpy().flatten())

                # 反标准化到原始数据
                target_original = self.preprocessor.inverse_transform_target(target.cpu().numpy()).flatten()
                output_original = self.preprocessor.inverse_transform_target(output.cpu().numpy()).flatten()

                y_true_original_list.extend(target_original)
                y_pred_original_list.extend(output_original)

                sample_count += len(target)

        y_true_original = np.array(y_true_original_list)
        y_pred_original = np.array(y_pred_original_list)

        self.model.train()

        return {}

    def evaluate_model(self):
        """评估模型"""
        self.logger.log_section("模型评估阶段")

        self.model.eval()
        y_true_scaled_tensors = []
        y_pred_scaled_tensors = []
        y_true_original_tensors = []
        y_pred_original_tensors = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                output = self.model(data)

                y_true_scaled_tensors.append(target.cpu())
                y_pred_scaled_tensors.append(output.cpu())

                target_original = self.preprocessor.inverse_transform_target(target.cpu().numpy())
                output_original = self.preprocessor.inverse_transform_target(output.cpu().numpy())

                y_true_original_tensors.append(torch.from_numpy(target_original))
                y_pred_original_tensors.append(torch.from_numpy(output_original))

        y_true_scaled = torch.cat(y_true_scaled_tensors, dim=0).numpy().flatten()
        y_pred_scaled = torch.cat(y_pred_scaled_tensors, dim=0).numpy().flatten()
        y_true_original = torch.cat(y_true_original_tensors, dim=0).numpy().flatten()
        y_pred_original = torch.cat(y_pred_original_tensors, dim=0).numpy().flatten()

        mse = mean_squared_error(y_true_scaled, y_pred_scaled)
        mae = mean_absolute_error(y_true_scaled, y_pred_scaled)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_scaled, y_pred_scaled)

        metrics = {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R²': r2}
        self.logger.log_metrics(metrics, "评估结果（标准化数据）")

        financial_metrics = self.financial_calculator.calculate_all_metrics(
            y_true_original, y_pred_original
        )

        self.financial_calculator.print_financial_metrics(financial_metrics)
        all_metrics = {**metrics, **financial_metrics}

        # 绘制时间序列对比图
        self.plot_timeseries_comparison(y_true_original, y_pred_original, show_plots=False)

        return all_metrics, y_true_original, y_pred_original

    def plot_timeseries_comparison(self, y_true, y_pred, show_plots=False):
        """绘制时间序列对比图"""
        save_path = self.config.paths.results_dir
        self.logger.log_section("时间序列对比图")

        os.makedirs(save_path, exist_ok=True)

        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']

        plt.ioff()

        plt.figure(figsize=(15, 6))
        n_show = min(200, len(y_true))
        x_axis = range(n_show)

        plt.plot(x_axis, y_true[:n_show], label='True Values', color='blue', linewidth=1)
        plt.plot(x_axis, y_pred[:n_show], label='Predicted Values', color='red', linewidth=1)
        plt.xlabel('Time Steps')
        plt.ylabel('Stock Price')
        plt.title('Stock Price Prediction Time Series')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'{save_path}/prediction_timeseries.png', dpi=150, bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close()

        self.logger.info(f"时间序列对比图已保存到: {save_path}/prediction_timeseries.png")
    
    def save_model(self, filename):
        """保存模型"""
        save_path = os.path.join(self.config.paths.models_dir, filename)
        os.makedirs(self.config.paths.models_dir, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'train_history': self.train_history,
            'data_info': self.data_info
        }

        torch.save(checkpoint, save_path)

    def load_model(self, filename):
        """加载模型"""
        load_path = os.path.join(self.config.paths.models_dir, filename)
        checkpoint = torch.load(load_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_history = checkpoint['train_history']

        self.logger.info(f"模型已从 {load_path} 加载")
    
    def run_full_pipeline(self):
        """运行完整的训练流程"""
        self.logger.log_section("股票预测模型训练流程")
        self.logger.log_config(self.config.get_config_dict())

        self.setup_data()
        self.setup_model()
        self.train_model()

        metrics, y_true, y_pred = self.evaluate_model()


        results_df = pd.DataFrame({
            'True_Values': y_true,
            'Predictions': y_pred,
            'Error': y_true - y_pred
        })
        results_df.to_csv(self.config.paths.prediction_results_file, index=False)

        self.logger.log_section("训练流程完成")
        self.logger.info(f"模型文件保存在: {self.config.paths.models_dir}")
        self.logger.info(f"结果图表保存在: {self.config.paths.results_dir}")
        self.logger.info(f"预测结果保存在: {self.config.paths.prediction_results_file}")

        return metrics


def main():
    """主函数"""
    config = get_config()
    logger = setup_logger(config)

    if not torch.cuda.is_available():
        logger.error("CUDA不可用！此项目需要GPU支持。请检查CUDA安装和GPU驱动。")
        raise RuntimeError("CUDA不可用")

    device = torch.device('cuda')
    logger.info(f"使用设备: {device}")
    logger.info(f"GPU型号: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    trainer = StockPredictionTrainer(config)

    try:
        metrics = trainer.run_full_pipeline()
        # logger.log_metrics(metrics, "最终结果")

    except Exception as e:
        logger.log_exception(e, "训练过程")
        raise


if __name__ == "__main__":
    main()