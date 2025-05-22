import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np

# 文件路径
file_path = "/home/ris/fdhumanoid_S2R/data/data0514/data0514/test5/test5_0.db/motor_leg_right_knee/joint_states.csv"

# 读取CSV文件
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: 文件未找到，请检查路径: {file_path}")
    exit()

# 解析字符串格式的列
df['position'] = df['position'].apply(ast.literal_eval)
df['effort'] = df['effort'].apply(ast.literal_eval)
df['velocity'] = df['velocity'].apply(ast.literal_eval)

# 提取数据并进行单位转换
position = np.array(df['position'].tolist()) / 50  # 形状为 (N, 3)，除以减速比 50
velocity = np.array(df['velocity'].tolist()) * ((2 * np.pi) / 60)  # 形状为 (N, 1)，r/min 转换为 rad/s
effort = np.array(df['effort'].tolist())  # 形状为 (N, 2)

# 生成横坐标（序号）
x = np.arange(len(df))

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# 1. 位置波形图（取第一个维度，第二个维度推测是传感器原始值，第三个维度可能是控制方式）
ax1.plot(x, position[:, 0], color='#1f77b4', linewidth=1.5, label='Position (X)')
ax1.set_ylabel('Position (rad)', fontsize=10)  # 单位保持不变
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_title('Motor Position, Velocity and Effort', fontsize=12, pad=20)

# 2. 速度波形图
ax2.plot(x, velocity[:, 0], color='#ff7f0e', linewidth=1.5, label='Velocity (rad/s)')
ax2.set_ylabel('Velocity (rad/s)', fontsize=10)  # 更新单位为 rad/s
ax2.legend(loc='upper right', fontsize=8)
ax2.grid(True, linestyle='--', alpha=0.6)

# 3. 力矩波形图（取第一个维度，第二个维度是力矩，反馈目前看为0）
ax3.plot(x, effort[:, 0], color='#2ca02c', linewidth=1.5, label='Effort (Torque)')
ax3.set_ylabel('Effort (Nm)', fontsize=10)
ax3.set_xlabel('Sample Index', fontsize=10)
ax3.legend(loc='upper right', fontsize=8)
ax3.grid(True, linestyle='--', alpha=0.6)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(hspace=0.2)

# 保存图像
# plt.savefig('/home/ris/fdhumanoid_S2R/data/ankleprb_plot.png', dpi=300)

plt.show()