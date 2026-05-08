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
h = 1.054571 * 10e-34 # J*S
L_M = a / ( 2 * np.sin(theta_rad/2) )
L_M1 = np.array([0, -1]) * L_M
L_M2 = np.array([np.sqrt(3) / 2, -1/2]) * L_M
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
        p1 = rotate(k_vec + G - K_1, theta_rad/2 )
        H[2*i:2*i+2, 2*i:2*i+2] = -hv * (valley * p1[0] * sigma_x + p1[1] * sigma_y) + delta_c2b * sigma_z

        p2 = rotate(k_vec + G - K_2, -theta_rad/2 )
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
print(f"开始计算轨道磁矩... 共 {num_k} 个 K 点。这可能需要一两分钟。")

for idx, (kx, ky) in enumerate(k_mesh):
    print(idx)

    # --- 关键修正 1：把每个 K 点只需要算 1 次的物理量提出来 ---
    # a. 对角化当前 K 点的哈密顿量
    H = build_hamiltonian(kx, ky)
    evals, evecs = np.linalg.eigh(H)
    evals -= E_offset  # 将费米面平移到 0

    # b. 用有限差分计算 H 对 px, py 的导数
    dH_dx = (build_hamiltonian(kx + dk, ky) - build_hamiltonian(kx - dk, ky)) / (2 * dk)
    dH_dy = (build_hamiltonian(kx, ky + dk) - build_hamiltonian(kx, ky - dk)) / (2 * dk)

    # c. 投影导数矩阵元到本征态基底上： <u_n | \partial_x H | u_\alpha>
    Vx = evecs.conj().T @ dH_dx @ evecs
    Vy = evecs.conj().T @ dH_dy @ evecs

    # --- 关键修正 2：内层循环只做纯粹的“切片求和” ---
    for i, n_c in enumerate(n_cut_list):
        # 截断的占据态索引：从 mid - n_c 到 mid - 1
        occ_slice = slice(mid_idx - n_c, mid_idx)
        # 截断的空态索引：从 mid 到 mid + n_c
        emp_slice = slice(mid_idx, mid_idx + n_c)

        E_n = evals[occ_slice][:, None]      # 变成列向量
        E_alpha = evals[emp_slice][None, :]  # 变成行向量
        dE = E_n - E_alpha                   # 自动广播生成 (E_n - E_alpha) 矩阵
        # (注：由于 occ 和 emp 严格位于费米面两侧，dE 绝对不可能为 0，删除了不需要的 fill_diagonal)

        # 提取速度矩阵区块
        V_x_na = Vx[occ_slice, emp_slice]
        V_y_an = Vy[emp_slice, occ_slice]

        V_y_na = Vy[occ_slice, emp_slice]
        V_x_an = Vx[emp_slice, occ_slice]

        # 计算并取虚部求和
        W_matrix = - (V_x_na * V_y_an) / (dE ** 2) * (E_n - mu)
        W_xy_im = np.imag(np.sum(W_matrix))

        N_matrix = - (V_y_na * V_x_an) / (dE ** 2) * (E_alpha - mu)
        N_xy_im = np.imag(np.sum(N_matrix))

        # 累加当前 n_cut 下的值
        val_M_orb = W_xy_im - N_xy_im
        M_sym_array[i] += val_M_orb



# --- 关键修正 3：所有循环结束后，统一取平均并转换单位！ ---
print("K 空间积分完成，正在转换物理单位...")
M_sym_array /= num_k
A_BZ = np.abs(np.cross(G1_M, G2_M))
M_sym_array *= A_BZ
A_M = np.sqrt(3)/2 * L_M**2
M_sym_array *= A_M
M_sym_array /= (2 * np.pi) ** 2
M_sym_array = M_sym_array * meV_to_J * (nm_to_m ** 2)
M_sym_array *= (- e / hbar)
M_sym_array /= mu_B
print("计算完成，正在绘制 Fig. 2(b)...")

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