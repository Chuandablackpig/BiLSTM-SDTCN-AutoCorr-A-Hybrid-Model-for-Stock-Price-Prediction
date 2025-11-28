import yfinance as yf
import sys
import os
proxy = 'http://127.0.0.1:33210'
os.environ['HTTP_PROXY'] = proxy
os.environ['HTTPS_PROXY'] = proxy

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 导入新的配置系统
from core.config_manager import get_config
from core.logger import get_logger

config = get_config()
logger = get_logger("us_index_data_fetcher")

def get_index_data(ticker, index_name, start_date, end_date):
    """获取指定美股指数的数据并保存到CSV文件"""
    file_path = config.paths.data_dir

    try:
        # yfinance 日期格式是 YYYY-MM-DD
        df = yf.download(ticker, start=start_date, end=end_date)

        # 记录获取到的字段
        logger.info(f"{index_name} 包含的字段: {df.columns.tolist()}")

        # yfinance 返回的索引是 datetime，我们可以重置为普通列
        df.reset_index(inplace=True)

        # 保存到CSV文件
        filename = f"{file_path}/{index_name}_{start_date}_to_{end_date}.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info(f"{index_name} 数据已保存至 {filename}，共 {len(df)} 条记录")

        return df
    except Exception as e:
        logger.error(f"获取 {index_name} 数据时出错: {str(e)}")
        return None

# 配置参数
start_date = "2012-02-01"
end_date = "2025-02-28"

# 美股指数信息：代码(跟踪ETF)和名称
indexes = [
    {"code": "SPY", "name": "标普500"},
    {"code": "QQQ", "name": "纳斯达克100"},
    {"code": "DIA", "name": "道琼斯工业平均指数"},
    {"code": "IWM", "name": "罗素2000小盘股指数"}
]

# 逐个获取指数数据
logger.log_section("开始获取美股指数数据")

for index in indexes:
    logger.info(f"正在获取 {index['name']} 数据...")
    get_index_data(
        ticker=index["code"],
        index_name=index["name"],
        start_date=start_date,
        end_date=end_date
    )

logger.info("所有美股指数数据获取完成！")
