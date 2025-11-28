import akshare as ak
import sys
import os

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 导入新的配置系统
from core.config_manager import get_config
from core.logger import get_logger

config = get_config()
logger = get_logger("stock_data_fetcher")

def get_index_data(index_code, index_name, start_date, end_date):
    """获取指定指数的数据并保存到CSV文件"""
    file_path = config.paths.data_dir

    try:
        # 获取指数数据
        df = ak.index_zh_a_hist(
            symbol=index_code,
            start_date=start_date,
            end_date=end_date,
        )

        # 记录获取到的字段
        logger.info(f"{index_name} 包含的字段: {df.columns.tolist()}")

        # 保存到CSV文件
        filename = f"{file_path}/{index_name}_{start_date}_to_{end_date}.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info(f"{index_name} 数据已保存至 {filename}，共 {len(df)} 条记录")

        return df
    except Exception as e:
        logger.error(f"获取 {index_name} 数据时出错: {str(e)}")
        return None

# 配置参数
start_date = "20000201"  # 开始日期
end_date = "20250228"    # 结束日期

# 指数信息：代码和名称
indexes = [
    {"code": "000001", "name": "上证综指"},
    {"code": "399001", "name": "深证成指"},
    {"code": "000300", "name": "沪深300"},
    {"code": "399006", "name": "创业板指"},
    {"code": "000906", "name": "中证800"}
]

# 逐个获取指数数据
logger.log_section("开始获取股票指数数据")

for index in indexes:
    logger.info(f"正在获取 {index['name']} 数据...")
    get_index_data(
        index_code=index["code"],
        index_name=index["name"],
        start_date=start_date,
        end_date=end_date
    )

logger.info("所有指数数据获取完成！")
