import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter

def plot_stock_index(index_name):
    """
    绘制单个指数的时序图
    :param index_name: 指数名称（用于拼接文件路径、标题、保存路径）
    """
    # 1. 拼接数据文件路径
    file_path = f'D:/PyCharm/PyCharmProjects/Stock/data/{index_name}_20120201_to_20250228.csv'
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"⚠️ 文件 {file_path} 不存在，跳过 {index_name} 的绘图。")
        return

    # 2. 日期列转换为 datetime 类型
    data['日期'] = pd.to_datetime(data['日期'])

    # 3. 绘图配置（清晰度、字体）
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']

    # 4. 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))

    # 5. 绘制折线（更深的橙色 + 合适线宽）
    ax.plot(data['日期'], data['收盘'], color='#E67E22', linewidth=2.5)

    # 6. 标题与标签（英文标题，根据指数名映射）
    title_map = {
        "上证综指": "Shanghai Composite Index",
        "中证800": "CSI 800 Index",
        "创业板指": "ChiNext Index",
        "沪深300": "CSI 300 Index",
        "深证成指": "Shenzhen Component Index"
    }
    ax.set_title(title_map.get(index_name, index_name))  # 若无映射则用原名称
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')

    # 7. Y 轴：保留两位小数
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # 8. X 轴：每 6 个月显示一个标签（平衡密度与可读性）
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))

    # 9. 网格：仅水平 + 浅灰细线
    ax.grid(axis='y', color='lightgrey', linestyle='-', linewidth=0.5)

    # 10. 日期标签旋转（避免重叠）
    plt.xticks(rotation=45)

    # 11. 紧凑布局
    plt.tight_layout()

    # 12. 保存图表
    save_path = f'D:/PyCharm/PyCharmProjects/Stock/Pic/{index_name}股票时序图.png'
    plt.savefig(save_path)
    print(f"✅ 已生成 {index_name} 的时序图，保存至：{save_path}")

    # 13. 关闭当前图表（避免多个图表重叠）
    plt.close(fig)


# ------------------- 批量执行 ------------------- #
# 定义需要绘图的指数列表
index_list = ["上证综指", "中证800", "创业板指", "沪深300", "深证成指"]

# 循环生成每个指数的时序图
for index_name in index_list:
    plot_stock_index(index_name)