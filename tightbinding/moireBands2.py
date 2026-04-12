# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# ================= 1. 参数定义 (PRX 2018，保持不变) =================
theta = 1.05
u = 0.0797       # w0, eV
u_prime = 0.0975 # w1, eV
a = 0.246        # 石墨烯晶格常数, nm
N = 5            # 倒格矢截断: -N 到 N
hv = 2.1354 * a  # ħv_F ≈ 0.525 eV·nm
valley = 1

def rotate(v, alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.dot(np.array([[c, -s], [s, c]]), v)

# 单层石墨烯倒格矢 (未旋转)
a1_star = 2 * np.pi * np.array([1, -1/np.sqrt(3)]) / a
a2_star = 2 * np.pi * np.array([0, 2/np.sqrt(3)]) / a

theta_rad = theta / 180.0 * np.pi
# 两层旋转后的Dirac点位置 (谷K)
K_1 = -valley * (2*rotate(a1_star, -theta_rad/2) + rotate(a2_star, -theta_rad/2)) / 3
K_2 = -valley * (2*rotate(a1_star,  theta_rad/2) + rotate(a2_star,  theta_rad/2)) / 3

# 莫尔倒格矢
G1_M = rotate(a1_star, -theta_rad/2) - rotate(a1_star, theta_rad/2)
G2_M = rotate(a2_star, -theta_rad/2) - rotate(a2_star, theta_rad/2)

# ================= 2. 构建倒格矢列表 (仿照Lattice函数逻辑) =================
L = []
invL = {}   # 字典映射 (m,n) -> 索引
idx = 0
for i in range(-N, N+1):
    for j in range(-N, N+1):
        L.append((i, j))
        invL[(i, j)] = idx
        idx += 1
n_sites = len(L)   # = (2N+1)^2
dim = 4 * n_sites  # 每个倒格矢点有4个轨道 (两层×两个子晶格)

# Pauli矩阵
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)

# 层间耦合矩阵 (PRX 2018 定义)
phi = 2 * np.pi / 3
omega = np.exp(1j * phi)
T1 = np.array([[u, u_prime], [u_prime, u]], dtype=complex)
T2 = np.array([[u, u_prime * np.conj(omega)], [u_prime * omega, u]], dtype=complex)
T3 = np.array([[u, u_prime * omega], [u_prime * np.conj(omega), u]], dtype=complex)

def build_hamiltonian(k_vec):
    """构造连续模型哈密顿量 (显式厄米，仿照zihaophys代码风格)"""
    H = np.zeros((dim, dim), dtype=complex)
    off = 2 * n_sites   # 层2在H中的起始索引

    for i, (m, n) in enumerate(L):
        G = m * G1_M + n * G2_M

        # ----- 层1 对角线块 (狄拉克哈密顿量) -----
        p1 = k_vec - K_1 + G
        H[2*i:2*i+2, 2*i:2*i+2] = -hv * (valley * p1[0] * sigma_x + p1[1] * sigma_y)

        # ----- 层2 对角线块 (狄拉克哈密顿量) -----
        p2 = k_vec - K_2 + G
        H[off+2*i:off+2*i+2, off+2*i:off+2*i+2] = -hv * (valley * p2[0] * sigma_x + p2[1] * sigma_y)

        # ----- 层间耦合 (同一倒格矢点) -----
        # 上三角: 层1 -> 层2
        H[2*i:2*i+2, off+2*i:off+2*i+2] += T1
        # 下三角: 层2 -> 层1 (共轭转置)
        H[off+2*i:off+2*i+2, 2*i:2*i+2] += T1.conj().T

        # ----- 耦合到相邻倒格矢 (m-1, n) -----
        if (m-1, n) in invL:
            j = invL[(m-1, n)]
            H[2*i:2*i+2, off+2*j:off+2*j+2] += T2
            H[off+2*j:off+2*j+2, 2*i:2*i+2] += T2.conj().T

        # ----- 耦合到相邻倒格矢 (m-1, n-1) -----
        if (m-1, n-1) in invL:
            j = invL[(m-1, n-1)]
            H[2*i:2*i+2, off+2*j:off+2*j+2] += T3
            H[off+2*j:off+2*j+2, 2*i:2*i+2] += T3.conj().T

    return H

# ================= 3. 高对称路径 (保持您的原始路径定义) =================
Gamma = np.array([0.0, 0.0])
M_pos = -K_1 + G2_M   # 您的原始M点定义 (若需要标准M点可改为 (G1_M+G2_M)/2)
nodes = [K_1, Gamma, M_pos, K_2]
labels = [r'$K$', r'$\Gamma$', r'$M$', r'$K$']
n_seg = 50   # 每段插值点数

k_vecs = []
k_dist = [0.0]
node_idx = [0]

for s in range(len(nodes)-1):
    for i in range(n_seg):
        if s > 0 and i == 0:
            continue
        frac = i / (n_seg - 1)
        k = nodes[s] * (1 - frac) + nodes[s+1] * frac
        k_vecs.append(k)
        if len(k_vecs) > 1:
            k_dist.append(k_dist[-1] + np.linalg.norm(k_vecs[-1] - k_vecs[-2]))
    node_idx.append(len(k_dist)-1)

print("正在计算能带 (仿照zihaophys逻辑，已修正错误)...")
bands = np.array([np.linalg.eigvalsh(build_hamiltonian(k)) for k in k_vecs])

# ================= 4. 绘图 (保留您的样式) =================
plt.figure(figsize=(8,6))
bands_mev = bands * 1000   # eV -> meV
mid = dim // 2
# 绘制中心附近的8条能带 (低能区，您也可改为全部绘制)
for i in range(mid-4, mid+4):
    plt.plot(k_dist, bands_mev[:, i], color='b', lw=1.5)

# 标记高对称点
for idx in node_idx:
    plt.axvline(k_dist[idx], color='k', alpha=0.3)
plt.xticks([k_dist[i] for i in node_idx], labels, fontsize=12)
plt.ylabel("Energy (meV)", fontsize=12)
plt.ylim(-300, 300)
plt.title(f"TBG Bands at {theta}$^\circ$ (continuum model, corrected)", fontsize=12)
plt.axhline(0, color='r', linestyle='--', alpha=0.5)
plt.xlim(0, max(k_dist))
plt.tight_layout()
plt.show()