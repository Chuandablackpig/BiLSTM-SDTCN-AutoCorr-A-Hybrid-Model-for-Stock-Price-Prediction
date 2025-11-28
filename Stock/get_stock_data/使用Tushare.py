import tushare as ts
pro = ts.pro_api("YOUR_TOKEN")  # 替换为实际Token

# 获取沪深300指数日数据（2015-01-01至2025-07-26）
df = pro.index_daily(
    ts_code="000300.SH",  # 沪深300指数代码
    start_date="20150101",
    end_date="20250726"
)

# 保存为CSV或直接处理
df.to_csv("hs300_tushare.csv", index=False)