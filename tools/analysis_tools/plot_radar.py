import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.text import Text

sns.set_style("whitegrid")  # 白底黑线
colors = sns.color_palette()
colors_d = sns.color_palette("dark")

# 最外圈标注数据
datasets = [
    "1%\n(Data)", "5%\n(Data)", "10%\n(Data)", "20%\n(Data)",
    "MobileNetV2\n(Backbone)", "ResNet-18\n(Backbone)", "Swin-S\n(Backbone)", "Swin-B\n(Backbone)",
    "DETR\n(Model)", "FCOS\n(Model)", "RetinaNet\n(Model)",
    "1x\n(Schedule)", "12k\n(Schedule)", "2x\n(Schedule)"
]

# 多边形数据
results = {
    "Unaligned Pre-training": [
        8.8,  19.1, 24.2, 26.5,
        30.1, 34.5, 46.6, 48.8,
        35.0, 36.6, 36.3,
        38.3, 27.2, 38.8
    ],
    "Aligned Pre-training (Ours)": [
        12.4, 22.4, 27.4, 30.1,
        31.3, 35.7, 47.5, 49.6,
        37.3, 37.5, 37.3,
        39.4, 30.5, 39.8
    ],
}

# 自适应调整坐标轴尺度
span = 10   # 坐标轴区域
start = []  # 每个轴的起始值(中心点值)
scale = []  # 每个轴的缩放比例
for data in zip(*list(results.values())):
    sc = (max(data) - min(data) + 2) / span
    st = min(data) - 1.5
    scale.append(sc)
    start.append(st)
start = start + start[:1]  # 首尾相接
scale = scale + scale[:1]  # 首尾相接

# === 初始化图 ===
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, polar=True)
plt.ylim(0, span)   # 调整坐标轴范围
plt.yticks([])

# hack: 偏移量，控制标注位置，避免标注重合
angle = 3  # 【按需调整】
text_dx = angle * np.pi / 180

line_styles = ['--o', 'o-', 'o-']

# === 绘制多边形 ===
lab = {}  # 记录图例
angles = np.linspace(0, 2*np.pi, len(datasets), endpoint=False)  # 等分360度
x_values = np.concatenate((angles, angles[:1]))  # 首尾相接
for i, (name, sts) in enumerate(results.items()):
    # x_values 为极坐标角度，y_values 为对应极轴上的数值
    y_values = np.concatenate((sts, sts[:1]))  # 首尾相接

    # 给围住的区域填色，画框线
    ax.fill(x_values, (y_values - np.array(start)) / np.array(scale), c=colors[i+1], alpha=0.25)
    lab[name] = ax.plot(x_values, (y_values - np.array(start)) / np.array(scale), line_styles[i], c=colors[i+1], linewidth=2)[0]

    # 给每个点标记数值，稍微偏移避免重合
    for j, (x, y, st, sc) in enumerate(zip(x_values[:-1], y_values[:-1], start[:-1], scale[:-1])):
        plt.text(x + text_dx * (-1) ** i, (y - st) / sc, y, c=colors_d[i+1], fontsize=14)

# 外围标注
ax.xaxis.set_tick_params(pad=25)      # 让最外面那圈标签出去点
ax.set_thetagrids(angles * 180 / np.pi, datasets, fontsize=16)
ax.spines['polar'].set_color('none')  # 去掉外框

# 显示图例
plt.legend(list(lab.values()), list(lab.keys()),
           ncol=len(results),  # 几列并排
           loc="upper center", bbox_to_anchor=(0.5, -0.1),  # 控制图例位置，loc 对齐该坐标
           fontsize=16)
plt.tight_layout()
plt.savefig('radar.pdf')
plt.savefig('radar.png')