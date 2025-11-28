"""
统一日志系统
替换所有print语句，提供结构化的日志输出
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class Logger:
    """统一日志管理器"""

    _instances = {}

    def __new__(cls, name: str = "stock_prediction", config=None):
        if name not in cls._instances:
            cls._instances[name] = super().__new__(cls)
        return cls._instances[name]

    def __init__(self, name: str = "stock_prediction", config=None):
        if hasattr(self, '_initialized'):
            return

        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # 避免重复添加handler
        if not self.logger.handlers:
            self._setup_logger(config)

        self._initialized = True

    def _setup_logger(self, config=None):
        """设置日志配置"""
        # 默认配置
        if config is None:
            log_level = "INFO"
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            log_dir = "logs"
            file_handler = True
            console_handler = True
            max_bytes = 10485760  # 10MB
            backup_count = 5
        else:
            log_level = config.logging.level
            log_format = config.logging.format
            log_dir = config.paths.logs_dir
            file_handler = config.logging.file_handler
            console_handler = config.logging.console_handler
            max_bytes = config.logging.max_bytes
            backup_count = config.logging.backup_count

        # 创建日志目录
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # 设置日志级别
        level = getattr(logging, log_level.upper(), logging.INFO)
        self.logger.setLevel(level)

        # 创建格式器
        formatter = logging.Formatter(log_format)

        # 控制台处理器
        if console_handler:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # 文件处理器（带轮转）
        if file_handler:
            log_file = os.path.join(log_dir, f"{self.name}.log")
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def debug(self, message: str, *args, **kwargs):
        """调试级别日志"""
        self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        """信息级别日志"""
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """警告级别日志"""
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """错误级别日志"""
        self.logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        """严重错误级别日志"""
        self.logger.critical(message, *args, **kwargs)

    def log_section(self, title: str, level: str = "INFO"):
        """记录分节日志"""
        separator = "=" * 50
        getattr(self, level.lower())(f"\n{separator}")
        getattr(self, level.lower())(f"{title}")
        getattr(self, level.lower())(f"{separator}")

    def log_config(self, config_dict: dict, title: str = "配置信息"):
        """记录配置信息"""
        self.log_section(title)
        for section, params in config_dict.items():
            self.info(f"[{section.upper()}]")
            if isinstance(params, dict):
                for key, value in params.items():
                    self.info(f"  {key}: {value}")
            else:
                self.info(f"  {params}")

    def log_metrics(self, metrics: dict, title: str = "评估指标"):
        """记录评估指标"""
        self.log_section(title)
        for metric, value in metrics.items():
            if isinstance(value, float):
                self.info(f"{metric}: {value:.6f}")
            else:
                self.info(f"{metric}: {value}")

    def log_model_info(self, model_info: dict):
        """记录模型信息"""
        self.log_section("模型信息")
        for key, value in model_info.items():
            if key == 'modules' and isinstance(value, dict):
                self.info(f"{key}:")
                for module_name, module_info in value.items():
                    self.info(f"  {module_name}: {module_info}")
            else:
                self.info(f"{key}: {value}")

    def log_training_progress(self, epoch: int, total_epochs: int,
                            train_loss: float, val_loss: float,
                            elapsed_time: float, metrics: dict = None):
        """记录训练进度"""
        progress_msg = (f"Epoch {epoch:3d}/{total_epochs} | "
                       f"Train Loss: {train_loss:.6f} | "
                       f"Val Loss: {val_loss:.6f} | "
                       f"Time: {elapsed_time:.1f}s")
        self.info(progress_msg)

        if metrics:
            metrics_msg = "    Metrics - " + " | ".join([
                f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in metrics.items()
            ])
            self.info(metrics_msg)

    def log_data_info(self, data_info: dict):
        """记录数据信息"""
        self.log_section("数据信息")
        for key, value in data_info.items():
            self.info(f"{key}: {value}")

    def log_exception(self, exception: Exception, context: str = ""):
        """记录异常信息"""
        if context:
            self.error(f"异常发生在 {context}: {str(exception)}")
        else:
            self.error(f"异常: {str(exception)}")

        # 记录详细的异常堆栈
        import traceback
        self.debug("异常堆栈:")
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                self.debug(line)


class LoggerMixin:
    """日志混入类，为其他类提供日志功能"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self.__class__.__name__)

    def log_debug(self, message: str, *args, **kwargs):
        self.logger.debug(message, *args, **kwargs)

    def log_info(self, message: str, *args, **kwargs):
        self.logger.info(message, *args, **kwargs)

    def log_warning(self, message: str, *args, **kwargs):
        self.logger.warning(message, *args, **kwargs)

    def log_error(self, message: str, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)

    def log_section(self, title: str, level: str = "INFO"):
        self.logger.log_section(title, level)


# 全局日志实例
_global_logger = None


def setup_logger(config=None) -> Logger:
    """设置全局日志器"""
    global _global_logger
    _global_logger = Logger("stock_prediction", config)
    return _global_logger


def get_logger(name: str = "stock_prediction") -> Logger:
    """获取日志器实例"""
    global _global_logger
    if _global_logger is None:
        _global_logger = Logger(name)
    return _global_logger


def log_info(message: str, *args, **kwargs):
    """快捷信息日志函数"""
    get_logger().info(message, *args, **kwargs)


def log_error(message: str, *args, **kwargs):
    """快捷错误日志函数"""
    get_logger().error(message, *args, **kwargs)


def log_warning(message: str, *args, **kwargs):
    """快捷警告日志函数"""
    get_logger().warning(message, *args, **kwargs)


def log_debug(message: str, *args, **kwargs):
    """快捷调试日志函数"""
    get_logger().debug(message, *args, **kwargs)


def log_section(title: str, level: str = "INFO"):
    """快捷分节日志函数"""
    get_logger().log_section(title, level)


if __name__ == "__main__":
    # 测试日志系统
    logger = get_logger()

    logger.log_section("日志系统测试")
    logger.info("这是一条信息日志")
    logger.warning("这是一条警告日志")
    logger.error("这是一条错误日志")
    logger.debug("这是一条调试日志")

    # 测试配置日志
    test_config = {
        "model": {"input_features": 10, "seq_len": 20},
        "training": {"epochs": 500, "batch_size": 36}
    }
    logger.log_config(test_config, "测试配置")

    # 测试指标日志
    test_metrics = {"MSE": 0.001234, "MAE": 0.005678, "R2": 0.987654}
    logger.log_metrics(test_metrics, "测试指标")

    print("日志测试完成，请检查logs目录下的日志文件")