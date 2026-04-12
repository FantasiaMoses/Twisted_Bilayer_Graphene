# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# ================= 1. 参数定义 (PRX 2018) =================
theta = 1.05
u = 0.0797 * 1000  # w0
u_prime = 0.0975 * 1000  # w1
a = 0.246
N = 5
theta_rad = theta / 180.0 * np.pi
hv = 2.1354 * a * 1000  # ~ 0.525 meV*nm
valley = 1
KDens  = 100            #density of k points

def rotate(v, alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.dot(np.array([[c, -s], [s, c]]), v)


# 莫尔倒格矢定义
a1_star = 2 * np.pi * np.array([1, -1 / np.sqrt(3)]) / a
a2_star = 2 * np.pi * np.array([0, 2 / np.sqrt(3)]) / a
# dirac point
K_1 = -valley * (2 * rotate(a1_star, -theta_rad/2) + rotate(a2_star, -theta_rad/2)) / 3
K_2 = -valley * (2 * rotate(a1_star, theta_rad/2) + rotate(a2_star, theta_rad/2)) / 3

# 莫尔倒格矢
G1_M = rotate(a1_star, -theta_rad/2) - rotate(a1_star, theta_rad/2)
G2_M = rotate(a2_star, -theta_rad/2) - rotate(a2_star, theta_rad/2)

# ================= 2. 构建哈密顿量 =================
L = []
for i in range(-N, N + 1):
    for j in range(-N, N + 1):
        L.append([i, j])
n_sites = len(L)
invL = {tuple(pos): idx for idx, pos in enumerate(L)}
dim = 4 * n_sites

sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
phi = 2 * np.pi / 3
omega = np.exp(1j * phi)
T1 = np.array([[u, u_prime], [u_prime, u]], dtype=complex)
T2 = np.array([[u, u_prime * np.conj(omega)], [u_prime * omega, u]], dtype=complex)
T3 = np.array([[u, u_prime * omega], [u_prime * np.conj(omega), u]], dtype=complex)


def build_hamiltonian(kx, ky):
    H = np.zeros((dim, dim), dtype=complex)
    off = 2 * n_sites
    k_vec = np.array([kx, ky])

    for i in range(n_sites):
        m, n = L[i]
        G = m * G1_M + n * G2_M

        # 层内项：使用相对位移确保上下对称
        p1 = rotate(k_vec + G - K_1, theta_rad/2)
        H[2*i:2*i+2, 2*i:2*i+2] = -hv * (valley * p1[0] * sigma_x + p1[1] * sigma_y)

        p2 = rotate(k_vec + G - K_2, -theta_rad/2)
        H[off+2*i:off+2*i+2, off+2*i:off+2*i+2] = -hv * (valley * p2[0] * sigma_x + p2[1] * sigma_y)

        # 层间耦合填充
        # T1 通道 (m, n) -> (m, n)
        H[2*i:2*i+2, off+2*i:off+2*i+2] = T1
        H[off+2*i:off+2*i+2, 2*i:2*i+2] = T1.conj().T

        # T2 通道 (m, n) -> (m+1, n)
        if (m+1, n) in invL:
            j2 = invL[(m+1, n)]
            H[2*i:2*i+2, off+2*j2:off+2*j2+2] = T2
            H[off+2*j2:off+2*j2+2, 2*i:2*i+2] = T2.conj().T

        # T3 通道 (m, n) -> (m+1, n+1)
        if (m+1, n+1) in invL:
            j3 = invL[(m+1, n+1)]
            H[2*i:2*i+2, off+2*j3:off+2*j3+2] = T3
            H[off+2*j3:off+2*j3+2, 2*i:2*i+2] = T3.conj().T

    return H


# ================= 3. 修正后的路径与计算 =================
# 莫尔 K 点矢量
K_vec  = K_2
Kp_vec = K_1
Gamma_vec = (K_1 + K_2) / 2 + rotate(K_1 - K_2, np.pi/2) * (np.sqrt(3)/2)
M_vec = (K_1 + K_2) / 2

# 采样路径
n_seg = KDens
path_K_G = np.linspace(K_vec, Gamma_vec, n_seg, endpoint=False)
path_G_M = np.linspace(Gamma_vec, M_vec, n_seg, endpoint=False)
path_M_Kp = np.linspace(M_vec, Kp_vec, n_seg, endpoint=True)

full_path = np.vstack([path_K_G, path_G_M, path_M_Kp])
AllK = len(full_path)
E = np.zeros((AllK, 4 * n_sites))

for idx, k_pos in enumerate(full_path):
    # 直接传入坐标，不再做基矢乘法
    H = build_hamiltonian(k_pos[0], k_pos[1])
    # 建议只取中间能带以观察平带
    eigvals = np.linalg.eigvalsh(H)
    E[idx, :] = np.real(eigvals)



#能带偏移修正
# 计算 M_vec (即 mBZ 中心) 处的哈密顿量，找到平带的中点
H_ref = build_hamiltonian(M_vec[0], M_vec[1])
ref_eigvals = np.linalg.eigvalsh(H_ref)
# 找到最靠近 0 的两个值的平均值（假设它们是平带）
mid_index = len(ref_eigvals) // 2
E_offset = (ref_eigvals[mid_index] + ref_eigvals[mid_index-1]) / 2

# 在绘图前减去这个偏移
E = E - E_offset

# ================= 4. 绘图优化 (全量能带展示) =================
plt.figure(figsize=(12, 8))

# 1. 遍历并绘制所有 484 条能带
# E 的每一列代表一条能带
for j in range(E.shape[1]):
    plt.plot(E[:, j], color='black', linewidth=0.5, alpha=0.7)

# 2. 突出显示费米面 (0 meV)
plt.axhline(0, color='blue', linestyle='--', linewidth=1, alpha=0.8)

# 3. 设置 y 轴范围为 -300 到 300 meV
plt.ylim(-10, 10)

# 4. 坐标轴与标题
plt.title(r"TBG Moire Bands at $\theta=1.05^\circ$ ($-10 \sim 10$ meV)", fontsize=18)
plt.ylabel('Energy (meV)', fontsize=16)

# 标记高对称点位置
x_labels = [0, len(path_K_G), len(path_K_G) + len(path_G_M), AllK]
plt.xticks(x_labels, ['K', r'$\Gamma$', 'M', "K'"], fontsize=18)

# 添加垂直线区分路径
for x in x_labels:
    plt.axvline(x, color='gray', linestyle=':', alpha=0.5)

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()




