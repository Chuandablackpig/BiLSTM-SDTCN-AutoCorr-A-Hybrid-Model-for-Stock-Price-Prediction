"""
金融评价指标计算器
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.logger import get_logger


class FinancialMetricsCalculator:
    """金融评价指标计算器"""

    def __init__(self, risk_free_rate=0.03):
        self.risk_free_rate = risk_free_rate
        self.logger = get_logger("financial_metrics")
    
    def calculate_annual_return(self, y_true, y_pred):
        """计算年收益率"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        trading_days_per_year = 252

        true_annual_return = (y_true[-1] / y_true[0]) ** (trading_days_per_year / len(y_true)) - 1
        pred_annual_return = (y_pred[-1] / y_pred[0]) ** (trading_days_per_year / len(y_pred)) - 1

        return true_annual_return, pred_annual_return
    
    def calculate_sharpe_ratio(self, y_true, y_pred):
        """计算夏普比率"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        true_daily_returns = np.diff(y_true) / y_true[:-1]
        pred_daily_returns = np.diff(y_pred) / y_pred[:-1]

        true_daily_returns = true_daily_returns[np.isfinite(true_daily_returns)]
        pred_daily_returns = pred_daily_returns[np.isfinite(pred_daily_returns)]

        trading_days_per_year = 252

        true_annual_return = np.mean(true_daily_returns) * trading_days_per_year
        pred_annual_return = np.mean(pred_daily_returns) * trading_days_per_year

        true_annual_volatility = np.std(true_daily_returns) * np.sqrt(trading_days_per_year)
        pred_annual_volatility = np.std(pred_daily_returns) * np.sqrt(trading_days_per_year)

        true_sharpe = (true_annual_return - self.risk_free_rate) / true_annual_volatility if true_annual_volatility != 0 else 0
        pred_sharpe = (pred_annual_return - self.risk_free_rate) / pred_annual_volatility if pred_annual_volatility != 0 else 0

        return true_sharpe, pred_sharpe
    
    def calculate_all_metrics(self, y_true, y_pred):
        """计算所有金融指标"""
        true_annual_return, pred_annual_return = self.calculate_annual_return(y_true, y_pred)
        true_sharpe, pred_sharpe = self.calculate_sharpe_ratio(y_true, y_pred)

        return {
            'True_Annual_Return': true_annual_return,
            'Pred_Annual_Return': pred_annual_return,
            'True_Sharpe_Ratio': true_sharpe,
            'Pred_Sharpe_Ratio': pred_sharpe,
            'Annual_Return_Diff': abs(true_annual_return - pred_annual_return),
            'Sharpe_Ratio_Diff': abs(true_sharpe - pred_sharpe)
        }
    
    def print_financial_metrics(self, metrics):
        """打印金融指标结果"""
        self.logger.log_section("金融评价指标")
        self.logger.info(f"真实年收益率: {metrics['True_Annual_Return']:.4f} ({metrics['True_Annual_Return']*100:.2f}%)")
        self.logger.info(f"预测年收益率: {metrics['Pred_Annual_Return']:.4f} ({metrics['Pred_Annual_Return']*100:.2f}%)")
        self.logger.info(f"年收益率差异: {metrics['Annual_Return_Diff']:.4f} ({metrics['Annual_Return_Diff']*100:.2f}%)")
        self.logger.info(f"真实夏普比率: {metrics['True_Sharpe_Ratio']:.4f}")
        self.logger.info(f"预测夏普比率: {metrics['Pred_Sharpe_Ratio']:.4f}")
        self.logger.info(f"夏普比率差异: {metrics['Sharpe_Ratio_Diff']:.4f}")


if __name__ == "__main__":
    logger = get_logger("financial_metrics_test")

    np.random.seed(42)
    n_days = 252

    initial_price = 3000
    daily_returns = np.random.normal(0.0005, 0.02, n_days)

    y_true = [initial_price]
    for ret in daily_returns:
        y_true.append(y_true[-1] * (1 + ret))
    y_true = np.array(y_true)

    noise = np.random.normal(0, 10, len(y_true))
    y_pred = y_true + noise

    calculator = FinancialMetricsCalculator(risk_free_rate=0.03)
    metrics = calculator.calculate_all_metrics(y_true, y_pred)
    calculator.print_financial_metrics(metrics)