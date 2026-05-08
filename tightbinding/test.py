# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# ================= 1. 参数定义 =================
theta = 1.086473511635319  # °
u = 0.06 * 1000     # mev  intra
u_prime = 0.11 * 1000  # mev  inter
a = 0.246   # nm
N = 5
theta_rad = theta / 180.0 * np.pi
hv = 2.36464 * a * 1000  # meV*nm
valley = -1
KDens  = 100            #density of k points
delta_c2b = 20.0  # meV
mu = -10.0 # mev
e = 1.602 * 10e-19  # C
h = 1.054571 * 10e-34 # J*S
L_M = a / ( 2 * np.sin(theta_rad/2) )
L_M1 = np.array([0, -1]) * L_M
L_M2 = np.array([np.sqrt(3) / 2, -1/2]) * L_M
hbar2_over_2m = 38.0998

mu = -10.0  # meV

e = 1.602176634e-19      # C
hbar = 1.054571817e-34   # J·s
mu_B = 9.2740100783e-24  # J/T

meV_to_J = 1.602176634e-22
nm_to_m = 1e-9

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
sigma_z = np.array([[1, 0], [0, -1]])
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
        p1 = rotate(k_vec + G - K_1, theta_rad/2 * 0)
        H[2*i:2*i+2, 2*i:2*i+2] = -hv * (valley * p1[0] * sigma_x + p1[1] * sigma_y) + delta_c2b * sigma_z

        p2 = rotate(k_vec + G - K_2, -theta_rad/2 * 0)
        H[off+2*i:off+2*i+2, off+2*i:off+2*i+2] = -hv * (valley * p2[0] * sigma_x + p2[1] * sigma_y) + delta_c2b * sigma_z

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


# ================= 3.轨道磁矩计算配置 =================
# 3.1 寻找基准点偏移 (CNP)
H_ref = build_hamiltonian(K_1[0], K_1[1])
ref_eigvals = np.real(np.linalg.eigvalsh(H_ref))
mid_idx = dim // 2
E_offset = (ref_eigvals[mid_idx - 1] + ref_eigvals[mid_idx]) / 2.0

# 3.2 生成莫尔布里渊区(mBZ)的二维积分网格
Nk = 15  # 900个k点
k_mesh = []
for i in range(Nk):
    for j in range(Nk):
        k_point = (i / Nk) * G1_M + (j / Nk) * G2_M
        k_mesh.append(k_point)
num_k = len(k_mesh)

# 3.3 要扫描的截断能带数 n_cut (1 到 225)
n_cut_list = np.arange(1, 60, 1)
M_sym_array = np.zeros(len(n_cut_list))
dk = 1e-5  # 有限差分步长


# ================= 4. 对每个 K 点进行矩阵积分 =================
for idx, (kx, ky) in enumerate(k_mesh):
    print(idx)

    # ===== diagonalize =====
    H = build_hamiltonian(kx, ky)
    evals, evecs = np.linalg.eigh(H)
    evals -= E_offset

    # ===== 构造 dH/dk =====
    dH_dx = (build_hamiltonian(kx + dk, ky) - build_hamiltonian(kx - dk, ky)) / (2 * dk)
    dH_dy = (build_hamiltonian(kx, ky + dk) - build_hamiltonian(kx, ky - dk)) / (2 * dk)

    # ===== 投影到本征基 =====
    Vx = evecs.conj().T @ dH_dx @ evecs
    Vy = evecs.conj().T @ dH_dy @ evecs

    # ===== 对每个 n_cut =====
    for i_cut, n_c in enumerate(n_cut_list):

        occ_idx = np.arange(mid_idx - n_c, mid_idx)
        emp_idx = np.arange(mid_idx, mid_idx + n_c)

        # ===== 构造 projector =====
        U_occ = evecs[:, occ_idx]
        U_emp = evecs[:, emp_idx]

        P = U_occ @ U_occ.conj().T
        Q = U_emp @ U_emp.conj().T

        # ===== 构造 dP/dk（用有限差分）=====
        # 注意：这里必须重新算 P(k±dk)

        def build_P(kx_, ky_, occ_idx_):
            H_ = build_hamiltonian(kx_, ky_)
            evals_, evecs_ = np.linalg.eigh(H_)
            evals_ -= E_offset
            U_occ_ = evecs_[:, occ_idx_]
            return U_occ_ @ U_occ_.conj().T

        P_x_plus  = build_P(kx + dk, ky, occ_idx)
        P_x_minus = build_P(kx - dk, ky, occ_idx)
        dP_dx = (P_x_plus - P_x_minus) / (2 * dk)

        P_y_plus  = build_P(kx, ky + dk, occ_idx)
        P_y_minus = build_P(kx, ky - dk, occ_idx)
        dP_dy = (P_y_plus - P_y_minus) / (2 * dk)

        # ===== dQ = - dP（因为 P+Q ≈ I）=====
        dQ_dx = -dP_dx
        dQ_dy = -dP_dy

        # ===== 计算 W_xy =====
        W = np.trace(
            P @ dP_dx @ dQ_dy @ (H - mu)
        )

        # ===== 计算 N_xy =====
        N = np.trace(
            Q @ dP_dx @ dQ_dy @ (H - mu)
        )

        M_k = -np.imag(W - N)

        M_sym_array[i_cut] += M_k

print("计算完成，正在绘制 Fig. 2(b)...")

M_sym_array /= num_k
A_BZ = np.abs(np.cross(G1_M, G2_M))
M_sym_array *= A_BZ
A_M = np.sqrt(3)/2 * L_M**2
M_sym_array /= (2 * np.pi) ** 2
M_sym_array /= A_M
M_sym_array = M_sym_array * meV_to_J * (nm_to_m ** 2)
M_sym_array *= (e / hbar)
M_sym_array /= mu_B

# ================= 5. 绘制 Fig. 2(b) 结果 =================
plt.figure(figsize=(7, 5))

# 绘制对称截断（黑色，平滑收敛）
plt.plot(n_cut_list, M_sym_array, color='black', marker='o', markersize=4,
         linestyle='-', linewidth=1.5, label=r'$M_{orb}$')

# 美化图表，匹配论文风格
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.xlabel(r'$n_{cut}$', fontsize=16)
plt.ylabel(r'$M_{orb} / N_{cell} (\mu_B)$', fontsize=16)
plt.title('Convergence of Orbital Magnetization at $\mu = -10$ meV', fontsize=14)

# 限制纵轴范围，方便清楚看到红色剧烈振荡和黑色完美收敛的对比

plt.ylim(-10, 10)
plt.xlim(0, max(n_cut_list))

plt.legend(fontsize=14, loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()