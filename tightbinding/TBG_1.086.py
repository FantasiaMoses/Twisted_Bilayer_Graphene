# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# ================= 1. 参数定义 =================
theta = 1.086  # °
u = 0.06 * 1000     # mev  intra
u_prime = 0.11 * 1000  # mev  inter
a = 0.246   # nm
N = 5
theta_rad = theta / 180.0 * np.pi
hv = 2.365 * a * 1000  # meV*nm
valley = -1
KDens  = 100            #density of k points
L_M = a / ( 2 * np.sin(theta_rad/2) )
L_M1 = np.array([0, -1]) * L_M
L_M2 = np.array([np.sqrt(3) / 2, -1/2]) * L_M
def rotate(v, alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.dot(np.array([[c, -s], [s, c]]), v)


# 基矢
a1_star = 2 * np.pi * np.array([1, -1 / np.sqrt(3)]) / a
a2_star = 2 * np.pi * np.array([0, 2 / np.sqrt(3)]) / a
# dirac point
K_1 = valley * np.array ([ 2 * np.pi / np.sqrt(3),  2 * np.pi / 3]) / L_M
K_2 = valley * np.array ([ 2 * np.pi / np.sqrt(3), - 2 * np.pi / 3]) / L_M

# 莫尔倒格矢
G1_M = np.array ([- 2 * np.pi / np.sqrt(3), - 2 * np.pi]) / L_M
G2_M = np.array ([4 * np.pi / np.sqrt(3), 0]) / L_M

print(L_M * G1_M / np.pi)
print(L_M * G2_M / np.pi * np.sqrt(3) / 2)



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
omega = np.exp(1j * valley * phi)
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

        # 层内项
        p1 = rotate(k_vec + G - K_1, theta_rad/2 )
        H[2*i:2*i+2, 2*i:2*i+2] = -hv * (valley * p1[0] * sigma_x + p1[1] * sigma_y)

        p2 = rotate(k_vec + G - K_2, -theta_rad/2 )
        H[off+2*i:off+2*i+2, off+2*i:off+2*i+2] = -hv * (valley * p2[0] * sigma_x + p2[1] * sigma_y)

        # 层间耦合填充
        # T1 通道 (m, n) -> (m, n)
        H[2*i:2*i+2, off+2*i:off+2*i+2] = T1
        H[off+2*i:off+2*i+2, 2*i:2*i+2] = T1.conj().T

        # T2 通道 (m, n) -> (m+1, n)
        if (m+1*valley, n) in invL:
            j2 = invL[(m+1*valley, n)]
            H[2*i:2*i+2, off+2*j2:off+2*j2+2] = T2
            H[off+2*j2:off+2*j2+2, 2*i:2*i+2] = T2.conj().T

        # T3 通道 (m, n) -> (m+1, n+1)
        if (m+1*valley, n+1*valley) in invL:
            j3 = invL[(m+1*valley, n+1*valley)]
            H[2*i:2*i+2, off+2*j3:off+2*j3+2] = T3
            H[off+2*j3:off+2*j3+2, 2*i:2*i+2] = T3.conj().T

    return H


# ================= 3. 路径与计算 =================
# 莫尔 K 点矢量

K_vec  =  np.array ([ 2 * np.pi / np.sqrt(3),  2 * np.pi / 3]) / L_M
Kp_vec =  np.array ([ 2 * np.pi / np.sqrt(3), - 2 * np.pi / 3]) / L_M
Gamma_vec = (Kp_vec + K_vec) / 2 + rotate(K_vec - Kp_vec, np.pi/2) * (np.sqrt(3)/2)
M_vec = (Kp_vec + K_vec) / 2

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
    print(E[idx, 2*n_sites-1:2*n_sites+1])



# ================= 能带偏移修正 (以 K_vec 狄拉克点为基准) =================
# 在莫尔狄拉克点 K_vec 处计算哈密顿量
H_ref = build_hamiltonian(K_vec[0], K_vec[1])
ref_eigvals = np.real(np.linalg.eigvalsh(H_ref))

# 寻找距离 0 最近的能级。由于此处两条平带发生狄拉克简并，
# 这个能量值就是最完美的电荷中性点 (CNP) 参考零点。
zero_idx = np.argmin(np.abs(ref_eigvals))
E_offset = ref_eigvals[zero_idx]

# 将所有计算出的能带减去狄拉克点的能量
E = E - E_offset

# ================= 4. 绘图 =================
plt.figure(figsize=(12, 8))

# 1. 遍历并绘制所有 484 条能带
# E 的每一列代表一条能带
for j in range(E.shape[1]):
    plt.plot(E[:, j], color='black', linewidth=0.5, alpha=0.7)

# 2. 突出显示费米面 (0 meV)
plt.axhline(0, color='blue', linestyle='--', linewidth=1, alpha=0.8)

# 3. 设置 y 轴范围为 -300 到 300 meV
plt.ylim(-300, 300)

# 4. 坐标轴与标题
plt.title(f"TBG Full Moire Bands at $\\theta = {theta}^\circ valley={valley}$ ($-300 \sim 300$ meV)", fontsize=18)
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




