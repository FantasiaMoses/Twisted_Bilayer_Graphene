# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 石墨烯晶格打开能隙后高对称点能带路径图

# 1. 物理参数设置
t = 2.8  # 跃迁能 (eV)
a = 1.42  # 碳原子间距 (Angstrom)
M = 0.6  # 质量项/势能差 (eV)，此值决定能隙大小为 1.2 eV

# 定义高对称点坐标 (基于之前的倒格矢推导)
Gamma = np.array([0, 0])
M_point = np.array([0, 2 * np.pi / (3 * a)])
K1 = np.array([2 * np.pi / (3 * np.sqrt(3) * a), 2 * np.pi / (3 * a)])

# 2. 构建连续路径: Gamma -> M -> K1 -> Gamma
n_points = 300
path1 = np.linspace(Gamma, M_point, n_points)
path2 = np.linspace(M_point, K1, n_points)
path3 = np.linspace(K1, Gamma, n_points)
full_path = np.vstack((path1, path2, path3))

# 计算累积距离作为横轴
distances = [0]
for i in range(1, len(full_path)):
    d = np.linalg.norm(full_path[i] - full_path[i - 1])
    distances.append(distances[-1] + d)


# 3. 能量函数计算 (带质量项的特征值求解结果)
def get_energy(kx, ky, mass):
    # 紧束缚模型结构因子 f(k) 的模平方
    term1 = 2 * np.cos(np.sqrt(3) * kx * a)
    term2 = 4 * np.cos(np.sqrt(3) * kx * a / 2) * np.cos(3 * ky * a / 2)
    f_sq = (t ** 2) * (3 + term1 + term2)

    # 能量 E = ±sqrt(|f(k)|^2 + M^2)
    return np.sqrt(np.abs(f_sq) + mass ** 2)


# 4. 执行计算
e_up = [get_energy(k[0], k[1], M) for k in full_path]
e_down = [-x for x in e_up]

# 5. 绘图
plt.figure(figsize=(10, 7))
plt.plot(distances, e_up, color='#1f77b4', lw=2.5, label='Conduction Band')
plt.plot(distances, e_down, color='#d62728', lw=2.5, label='Valence Band')

# 添加高对称点分割线和标签
v_lines = [distances[0], distances[n_points - 1], distances[2 * n_points - 1], distances[-1]]
for v in v_lines:
    plt.axvline(x=v, color='gray', linestyle='--', alpha=0.5)

plt.xticks(v_lines, [r'$\Gamma$', 'M', '$K_1$', r'$\Gamma$'], fontsize=12)
plt.yticks(fontsize=10)
plt.ylabel('Energy (eV)', fontsize=13)
plt.title(f'Graphene Band Structure with Mass Term ($M = {M}$ eV)', fontsize=14)

# 突出显示能隙区域
gap_center = distances[2 * n_points - 1]
plt.annotate('', xy=(gap_center, M), xytext=(gap_center, -M),
             arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=2))
plt.text(gap_center + 0.05, 0, f'Band Gap = {2 * M} eV',
         color='darkgreen', fontweight='bold', va='center')

plt.axhline(0, color='black', lw=1, alpha=0.2)  # 费米能级
plt.grid(True, alpha=0.2)
plt.legend(frameon=True)
plt.tight_layout()

plt.show()