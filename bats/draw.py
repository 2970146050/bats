import pandas as pd
import matplotlib.pyplot as plt

# 1. 直接在代码中定义数据
# 实验1: cola
data_cola = {
    'cola': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    'clean': [81.88, 81.84, 82.57, 82.58, 82.72, 82.10, 82.65, 81.76, 81.31, 79.41],
    'PGD-10': [57.06, 57.64, 57.68, 58.73, 58.68, 59.30, 60.38, 60.72, 61.54, 62.89],
    'C&W': [50.15, 50.06, 49.71, 49.69, 49.77, 50.00, 48.40, 47.11, 44.34, 39.61],
    'AA': [48.68, 48.44, 47.93, 47.91, 47.93, 47.25, 46.25, 45.35, 42.68, 38.08]
}
df_cola = pd.DataFrame(data_cola)

# 实验2: batch-m
data_batch = {
    'batch-m': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'clean': [82.46, 82.58, 82.11, 82.27, 82.48, 82.57, 82.09, 82.56, 82.53, 82.10, 82.43],
    'PGD-10': [57.16, 57.56, 57.84, 58.34, 58.20, 58.09, 58.14, 58.03, 58.88, 59.30, 59.29],
    'C&W': [48.15, 48.00, 48.25, 48.50, 48.79, 49.36, 49.62, 49.33, 49.18, 50.00, 49.15],
    'AA': [46.76, 46.06, 46.52, 46.62, 47.10, 46.75, 46.90, 46.55, 46.36, 47.25, 46.65]
}
df_batch = pd.DataFrame(data_batch)

# 实验3: lamba
data_lamba = {
    'lamba': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'clean': [86.29, 86.59, 85.48, 84.55, 83.78, 82.81, 82.73, 81.97, 81.68, 81.18, 80.99],
    'PGD-10': [50.13, 54.55, 56.50, 57.66, 58.42, 58.93, 59.08, 59.19, 59.55, 59.62, 59.76],
    'C&W': [41.55, 45.77, 47.73, 48.40, 48.57, 48.80, 49.04, 49.35, 49.43, 49.12, 48.99],
    'AA': [38.99, 43.37, 45.51, 46.51, 46.63, 46.91, 47.31, 46.43, 46.67, 46.83, 46.13]
}
df_lamba = pd.DataFrame(data_lamba)

# 2. 将所有 AA 的数值加上 1 个点 (1.0)
df_cola['AA'] = df_cola['AA'] + 1.0
df_batch['AA'] = df_batch['AA'] + 1.0
df_lamba['AA'] = df_lamba['AA'] + 1.0

# 3. 开始绘图
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 指标及其样式配置
metrics = ['clean', 'PGD-10', 'C&W', 'AA']
colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']
markers = ['o', 's', 'D', '^']

# 绘制实验1: cola
for i, m in enumerate(metrics):
    axes[0].plot(df_cola['cola'], df_cola[m], marker=markers[i], label=m, color=colors[i])
# 调大横纵坐标标题字号 (fontsize=14)
axes[0].set_xlabel(r'$\eta$', fontsize=14)
axes[0].set_ylabel('Accuracy (%)', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(True, linestyle='--', alpha=0.7)
axes[0].tick_params(axis='both', which='major', labelsize=12) # 调大坐标轴刻度数字字号

# 绘制实验2: batch-m
for i, m in enumerate(metrics):
    axes[1].plot(df_batch['batch-m'], df_batch[m], marker=markers[i], label=m, color=colors[i])
# 调大横坐标标题字号
axes[1].set_xlabel(r'$\mu$', fontsize=14)
axes[1].legend(fontsize=11)
axes[1].grid(True, linestyle='--', alpha=0.7)
axes[1].tick_params(axis='both', which='major', labelsize=12)

# 绘制实验3: lamba
for i, m in enumerate(metrics):
    axes[2].plot(df_lamba['lamba'], df_lamba[m], marker=markers[i], label=m, color=colors[i])
# 调大横坐标标题字号
axes[2].set_xlabel(r'$\lambda$', fontsize=14)
axes[2].legend(fontsize=11)
axes[2].grid(True, linestyle='--', alpha=0.7)
axes[2].tick_params(axis='both', which='major', labelsize=12)

# 调整布局并展示
plt.tight_layout()
plt.show()