"""
统一配置管理系统
实现所有参数的规范化管理
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional


@dataclass
class ModelConfig:
    """模型配置参数"""
    input_features: int = 10
    seq_len: int = 20
    bilstm_hidden: int = 64
    bilstm_layers: int = 3
    transformer_heads: int = 8
    transformer_layers: int = 6
    tcn_channels: List[int] = None
    tcn_kernel_size: int = 7
    dropout: float = 0.2
    decomp_kernel_size: int = 25
    autocorr_factor: int = 1

    def __post_init__(self):
        if self.tcn_channels is None:
            self.tcn_channels = [32, 32, 32, 32]


@dataclass
class TrainingConfig:
    """训练配置参数"""
    epochs: int = 1000
    batch_size: int = 36
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    early_stopping_patience: int = 100
    save_interval: int = 100
    validation_interval: int = 10
    optimizer: str = "adam"
    loss_function: str = "mse"
    device: str = "cuda"


@dataclass
class DataConfig:
    """数据配置参数"""
    window_size: int = 20
    test_size: float = 0.2
    random_state: int = 42
    target_column: str = "收盘"
    feature_columns: List[str] = None
    num_workers: int = 10
    pin_memory: bool = True
    persistent_workers: bool = True

    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = [
                '收盘', '最高', '最低', '开盘', '涨跌额',
                '涨跌幅', '成交量', '成交额', 'MA5', 'MA10'
            ]


@dataclass
class PathConfig:
    """路径配置参数"""
    project_root: str = ""
    data_dir: str = ""
    results_dir: str = ""
    models_dir: str = ""
    logs_dir: str = ""
    stock_data_file: str = ""
    prediction_results_file: str = ""

    def __post_init__(self):
        if not self.project_root:
            self.project_root = self._detect_environment()

        root = Path(self.project_root)
        self.data_dir = str(root / "data")
        self.results_dir = str(root / "results")
        self.models_dir = str(root / "saved_models")
        self.logs_dir = str(root / "logs")
        self.stock_data_file = str(root / "data" / "上证综指_20120201_to_20250228.csv")
        self.prediction_results_file = str(root / "results" / "prediction_results.csv")

    def _detect_environment(self) -> str:
        current_dir = Path.cwd()
        if (current_dir / "core" / "config_manager.py").exists():
            return str(current_dir)

        # 向上查找项目根目录
        for parent in current_dir.parents:
            if (parent / "core" / "config_manager.py").exists():
                return str(parent)

        # 如果在AutoDL环境但找不到项目文件，使用默认路径
        if os.path.exists("/root/autodl-tmp") or "autodl-tmp" in os.getcwd():
            return "/root/autodl-tmp"

        # 检查环境变量
        if "AUTODL_ROOT" in os.environ:
            return os.environ["AUTODL_ROOT"]

        # 默认本地开发环境路径
        return "D:/PyCharm/PyCharmProjects/Stock"


@dataclass
class LogConfig:
    """日志配置参数"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler: bool = True
    console_handler: bool = True
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5


class ConfigManager:
    """统一配置管理器"""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.paths = PathConfig()
        self.logging = LogConfig()

        if config_file and os.path.exists(config_file):
            self.load_config(config_file)

        self._create_directories()

    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            self.paths.data_dir,
            self.paths.results_dir,
            self.paths.models_dir,
            self.paths.logs_dir
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def load_config(self, config_file: str):
        """从文件加载配置"""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"配置文件不存在: {config_file}")

        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.endswith('.json'):
                config_data = json.load(f)
            elif config_file.endswith(('.yml', '.yaml')):
                config_data = yaml.safe_load(f)
            else:
                raise ValueError("配置文件格式不支持，请使用 .json 或 .yaml 格式")

        # 更新配置
        if 'model' in config_data:
            self.model = ModelConfig(**config_data['model'])
        if 'training' in config_data:
            self.training = TrainingConfig(**config_data['training'])
        if 'data' in config_data:
            self.data = DataConfig(**config_data['data'])
        if 'paths' in config_data:
            self.paths = PathConfig(**config_data['paths'])
        if 'logging' in config_data:
            self.logging = LogConfig(**config_data['logging'])

    def save_config(self, config_file: str):
        """保存配置到文件"""
        config_data = {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'paths': asdict(self.paths),
            'logging': asdict(self.logging)
        }

        # 只有当配置文件包含目录路径时才创建目录
        config_dir = os.path.dirname(config_file)
        if config_dir:
            os.makedirs(config_dir, exist_ok=True)

        with open(config_file, 'w', encoding='utf-8') as f:
            if config_file.endswith('.json'):
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            elif config_file.endswith(('.yml', '.yaml')):
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError("配置文件格式不支持，请使用 .json 或 .yaml 格式")

    def update_config(self, section: str, **kwargs):
        """更新配置参数"""
        if section == 'model':
            for key, value in kwargs.items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)
        elif section == 'training':
            for key, value in kwargs.items():
                if hasattr(self.training, key):
                    setattr(self.training, key, value)
        elif section == 'data':
            for key, value in kwargs.items():
                if hasattr(self.data, key):
                    setattr(self.data, key, value)
        elif section == 'paths':
            for key, value in kwargs.items():
                if hasattr(self.paths, key):
                    setattr(self.paths, key, value)
        elif section == 'logging':
            for key, value in kwargs.items():
                if hasattr(self.logging, key):
                    setattr(self.logging, key, value)
        else:
            raise ValueError(f"未知的配置节: {section}")

    def get_config_dict(self) -> Dict[str, Any]:
        """获取完整配置字典"""
        return {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'paths': asdict(self.paths),
            'logging': asdict(self.logging)
        }

    def print_config(self):
        """打印当前配置"""
        print("=== 当前配置信息 ===")
        print("\n[模型配置]")
        for key, value in asdict(self.model).items():
            print(f"  {key}: {value}")

        print("\n[训练配置]")
        for key, value in asdict(self.training).items():
            print(f"  {key}: {value}")

        print("\n[数据配置]")
        for key, value in asdict(self.data).items():
            print(f"  {key}: {value}")

        print("\n[路径配置]")
        for key, value in asdict(self.paths).items():
            print(f"  {key}: {value}")

        print("\n[日志配置]")
        for key, value in asdict(self.logging).items():
            print(f"  {key}: {value}")
        print("==================")


# 全局配置实例
config_manager = ConfigManager()


def get_config() -> ConfigManager:
    """获取全局配置管理器"""
    return config_manager


def load_config_from_file(config_file: str) -> ConfigManager:
    """从文件加载配置"""
    return ConfigManager(config_file)


def set_server_environment(server_type: str = "autodl") -> ConfigManager:
    """设置服务器环境配置

    Args:
        server_type: 服务器类型，支持 'autodl', 'custom'

    Returns:
        ConfigManager: 配置管理器实例
    """
    global config_manager

    if server_type == "autodl":
        # 设置AutoDL服务器环境变量
        os.environ["AUTODL_ROOT"] = "/root/autodl-tmp"
        # 重新创建配置管理器以应用新路径
        config_manager = ConfigManager()
    elif server_type == "custom":
        # 允许用户自定义服务器路径
        custom_path = input("请输入服务器项目根路径 (默认: /root/autodl-tmp/): ").strip()
        if not custom_path:
            custom_path = "/root/autodl-tmp"

        # 创建自定义路径配置
        config_manager = ConfigManager()
        config_manager.paths.project_root = custom_path
        config_manager.paths.__post_init__()
        config_manager._create_directories()
    else:
        raise ValueError(f"不支持的服务器类型: {server_type}")

    return config_manager


def get_environment_info() -> dict:
    """获取当前环境信息"""
    return {
        "os_name": os.name,
        "current_dir": str(Path.cwd()),
        "project_root": config_manager.paths.project_root,
        "autodl_tmp_exists": os.path.exists("~/autodl-tmp"),
        "autodl_env_var": os.environ.get("AUTODL_ROOT", "未设置"),
        "is_server_env": "autodl-tmp" in config_manager.paths.project_root
    }


if __name__ == "__main__":
    # 测试配置管理器
    print("=== 环境检测测试 ===")
    env_info = get_environment_info()
    for key, value in env_info.items():
        print(f"{key}: {value}")

    print("\n=== 默认配置 ===")
    config = ConfigManager()
    config.print_config()

    # 测试服务器环境设置
    print("\n=== 测试服务器环境设置 ===")
    try:
        # 模拟设置AutoDL环境
        print("设置AutoDL服务器环境...")
        server_config = set_server_environment("autodl")
        print("服务器环境配置:")
        print(f"项目根路径: {server_config.paths.project_root}")
        print(f"数据目录: {server_config.paths.data_dir}")
        print(f"模型目录: {server_config.paths.models_dir}")
    except Exception as e:
        print(f"服务器环境设置测试失败: {e}")

    # 测试保存配置
    config.save_config("config_test.json")
    print("\n配置已保存到 config_test.json")

    # 测试加载配置
    config2 = ConfigManager("config_test.json")
    print("\n从文件加载的配置:")
    config2.print_config()