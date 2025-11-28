import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

# ------------------- 1. 读取并处理数据 -------------------
data = pd.read_csv('C:/Users/范炜鋆/Desktop/申请/科研/Stock/结果/0.94/0.94-shuffle关/prediction_results.csv')  # 替换为实际CSV路径

true_values = data['True_Values']   # 全量实际值
predictions = data['Predictions']   # 全量预测值
x = range(1, len(true_values) + 1)  # X轴为「1~数据总数」的序号


# ------------------- 2. 图表样式配置 -------------------
plt.rcParams['figure.dpi'] = 300  # 高清分辨率
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']  # 英文标签字体
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 6))  # 窄长型图表，匹配示例比例


# ------------------- 3. 绘制双折线 -------------------
# 实际值：深蓝色、实线、线宽2
ax.plot(x, true_values, color='#1F77B4', linewidth=2, label='actual')
# 预测值：深橙色、实线、线宽2
ax.plot(x, predictions, color='#FF7F0E', linewidth=2, label='predicted_value')


# ------------------- 4. 坐标轴调整（核心：解决X轴拥挤） -------------------
## Y轴：保留2位小数 + 动态范围
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 保留2位小数
y_min = min(min(true_values), min(predictions)) - 50      # 上下留50缓冲
y_max = max(max(true_values), max(predictions)) + 50
ax.set_ylim(y_min, y_max)

## X轴：用MultipleLocator控制刻度间隔（每20个点显示1个刻度，可按需调整）
ax.xaxis.set_major_locator(MultipleLocator(20))  # 关键：每20个数据点显示1个刻度
ax.set_xlim(1, len(x))  # X轴范围固定为「1 ~ 数据总数」


# ------------------- 5. 标签与图例 -------------------
ax.set_title('Shanghai Composite Index', fontsize=14, pad=15)
ax.set_xlabel('')  # 隐藏X轴文字标签（示例图风格）
ax.set_ylabel('Value', fontsize=12)

# 图例放在图表下方，避免遮挡线条
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)


# ------------------- 6. 网格与边框优化 -------------------
ax.grid(axis='y', color='lightgrey', linewidth=0.5, linestyle='-', alpha=0.7)  # 仅水平网格
ax.spines['top'].set_visible(False)    # 隐藏顶部边框
ax.spines['right'].set_visible(False)  # 隐藏右侧边框


# ------------------- 7. 保存与显示 -------------------
plt.tight_layout()  # 紧凑布局，避免标签截断
plt.savefig('true_vs_predicted_plot.png', bbox_inches='tight')  # 保存图片
plt.show()