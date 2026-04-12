# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 石墨烯晶格高对称点能带路径图

# 1. 参数设置
t = 2.8  # 跃迁能 (eV)
a = 1.42  # 原子间距 (Angstrom)

# 定义高对称点坐标
Gamma = np.array([0, 0])
M = np.array([0, 2 * np.pi / (3 * a)])
K1 = np.array([2 * np.pi / (3 * np.sqrt(3) * a), 2 * np.pi / (3 * a)])


# 2. 能量函数 (严格匹配你提供的结构: 3 + term1 + term2)
def get_energy(kx, ky):
    # term1 对应 2 * cos(sqrt(3) * kx * a)
    term1 = 2 * np.cos(np.sqrt(3) * kx * a)
    # term2 对应 4 * cos(sqrt(3)*kx*a/2) * cos(3*ky*a/2)
    term2 = 4 * np.cos(np.sqrt(3) * kx * a / 2) * np.cos(3 * ky * a / 2)

    f_sq = 3 + term1 + term2
    # 使用 abs 避免浮点误差导致的微小负数开方
    return t * np.sqrt(np.abs(f_sq))


# 3. 构建三段完整路径: Gamma -> M -> K1 -> Gamma
n_points = 200
path1 = np.linspace(Gamma, M, n_points)
path2 = np.linspace(M, K1, n_points)
path3 = np.linspace(K1, Gamma, n_points)
full_path = np.vstack((path1, path2, path3))

# 计算累积距离作为横轴坐标
distances = [0]
for i in range(1, len(full_path)):
    d = np.linalg.norm(full_path[i] - full_path[i - 1])
    distances.append(distances[-1] + d)

# 4. 计算能量
energies_up = []
energies_down = []
for k in full_path:
    e = get_energy(k[0], k[1])
    energies_up.append(e)
    energies_down.append(-e)

# 5. 绘图
plt.figure(figsize=(12, 7))
plt.plot(distances, energies_up, color='blue', lw=2, label='Conduction Band')
plt.plot(distances, energies_down, color='red', lw=2, label='Valence Band')

# 添加高对称点分割线
plt.axvline(x=distances[0], color='black', linestyle='--', alpha=0.5)
plt.axvline(x=distances[n_points - 1], color='black', linestyle='--', alpha=0.5)
plt.axvline(x=distances[2 * n_points - 1], color='black', linestyle='--', alpha=0.5)
plt.axvline(x=distances[-1], color='black', linestyle='--', alpha=0.5)

# 设置横轴刻度标签
plt.xticks([distances[0], distances[n_points - 1], distances[2 * n_points - 1], distances[-1]],
           [r'$\Gamma$', 'M', '$K_1$', r'$\Gamma$'])

plt.ylabel('Energy (eV)', fontsize=12)
plt.title('Graphene Band Structure Path: $\Gamma$ - M - $K_1$ - $\Gamma$', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

# 设置 y 轴显示范围，让图像更好看
plt.ylim(-10, 10)
plt.axhline(0, color='black', lw=1, alpha=0.3)  # 费米面

plt.show()