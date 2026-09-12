import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 全局样式与排版设置
# ==========================================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8

# ==========================================
# 2. 精确构建数据 (数学公式拟合，拒绝随机波动)
# ==========================================

# --- 子图 (a) 的 KDE 曲线精确拟合 ---
x = np.linspace(1.1, 2.6, 500)

def gaussian(x, mu, sig):
    return (1.0 / (np.sqrt(2.0 * np.pi) * sig)) * np.exp(-0.5 * ((x - mu) / sig)**2)

# 蓝线 (Easy): 由两个主峰和一个用于平滑波谷的隐藏峰组成
y_blue = (0.42 * gaussian(x, 1.64, 0.12) +
          0.68 * gaussian(x, 2.09, 0.11) +
          0.15 * gaussian(x, 1.86, 0.15))

# 红线 (Hard): 单个极高极窄的峰，左侧尾巴略宽
y_red = 1.15 * gaussian(x, 2.19, 0.10)

# --- 子图 (b) 的折线轨迹精确拟合 ---
epochs = np.arange(110)
blue_mean = np.zeros(110)
red_mean = np.zeros(110)

# 蓝线走势精调
blue_mean[0] = 1.8
for i in range(1, 20):
    blue_mean[i] = 1.8 - 0.65 * (i / 19.0)**0.8   # 前20轮快速下降到 1.15
for i in range(20, 85):
    blue_mean[i] = 1.15 + 0.8 * ((i - 19) / 65.0)**1.1  # 20-85轮平滑上升到 1.95
for i in range(85, 110):
    blue_mean[i] = 1.95 - 0.06 * ((i - 85) / 24.0)**1.5 # 85轮后微降到 1.89

# 红线走势精调
red_mean[0] = 4.8
for i in range(1, 110):
    red_mean[i] = 2.05 + 0.12 * (i / 109.0)**0.7 # 1轮后瞬间掉到2.05，然后极缓慢上升到2.17

# 误差棒及微小噪点 (模拟真实训练的抖动)
np.random.seed(42)
blue_mean[1:] += np.random.normal(0, 0.02, 109)
red_mean[1:] += np.random.normal(0, 0.02, 109)

blue_err = np.random.uniform(0.12, 0.18, size=110)
blue_err[0] = 1.4 # 精确还原原图中第0轮蓝线极其夸张的向下误差棒

red_err = np.random.uniform(0.1, 0.15, size=110)
red_err[0] = 0.1

# ==========================================
# 3. 开始绘图
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

# ------------------------------------------
# 绘制子图 (a) : 密度曲线
# ------------------------------------------
ax1 = axes[0]
# 绘制蓝线与填充
ax1.plot(x, y_blue, color='darkblue', linewidth=1.5)
ax1.fill_between(x, y_blue, color='blue', alpha=0.3, label='Easy/Robust Samples')
# 绘制红线与填充
ax1.plot(x, y_red, color='darkred', linewidth=1.5)
ax1.fill_between(x, y_red, color='red', alpha=0.3, label='Hard/Vulnerable Samples')

ax1.set_title('Loss Distribution in Late Training Stage', pad=10)
ax1.set_xlabel('Pure Cross Entropy Loss (No Label Smoothing)\n\n(a)', fontsize=11)
ax1.set_ylabel('Density')

# 图例边框调整
legend1 = ax1.legend(loc='upper left', frameon=True, edgecolor='black')
legend1.get_frame().set_linewidth(0.8)

ax1.grid(True, linestyle='--', color='lightgray', alpha=0.8)
ax1.set_xlim(1.1, 2.6)
ax1.set_ylim(0, 5.4)

# ------------------------------------------
# 绘制子图 (b) : 轨迹图
# ------------------------------------------
ax2 = axes[1]
ax2.errorbar(epochs, blue_mean, yerr=blue_err, fmt='-', color='blue',
             ecolor='blue', elinewidth=1.0, capsize=2, linewidth=1.5, label='Easy Samples Avg Loss')
ax2.errorbar(epochs, red_mean, yerr=red_err, fmt='-', color='red',
             ecolor='red', elinewidth=1.0, capsize=2, linewidth=1.5, label='Hard Samples Avg Loss')

ax2.set_title('Loss Trajectory over Epochs', pad=10)
ax2.set_xlabel('Epochs\n\n(b)', fontsize=11)
ax2.set_ylabel('Average Pure Loss')

# 图例边框调整
legend2 = ax2.legend(loc='upper right', frameon=True, edgecolor='black')
legend2.get_frame().set_linewidth(0.8)

ax2.grid(True, linestyle='--', color='lightgray', alpha=0.8)
ax2.set_xlim(-5, 115)
ax2.set_ylim(0.8, 5.1)

# ==========================================
# 4. 布局调整与保存
# ==========================================
plt.tight_layout(pad=3.0)

# 保存高质量矢量图与位图
plt.savefig('loss_analysis_figure_perfect.pdf', format='pdf', bbox_inches='tight')
plt.savefig('loss_analysis_figure_perfect.png', format='png', bbox_inches='tight', dpi=300)

plt.show()