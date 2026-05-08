# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# ================= 1. 参数定义 =================
theta = 1.086473511635319  # °
u = 0.06 * 1000     # meV
u_prime = 0.11 * 1000  # meV
a = 0.246   # nm
N = 5
theta_rad = theta / 180.0 * np.pi
hv = 2.36464 * a * 1000  # meV*nm
valley = -1
delta_c2b = 20.0  # meV
mu = -10.0  # meV

e = 1.602176634e-19      # C
hbar = 1.054571817e-34   # J·s
mu_B = 38.0998  # mev * nm^2

meV_to_J = 1.602176634e-22
nm_to_m = 1e-9

L_M = a / (2 * np.sin(theta_rad / 2))

def rotate(v, alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.dot(np.array([[c, -s], [s, c]]), v)

# Dirac points
K_1 = valley * np.array([2 * np.pi / np.sqrt(3), 2 * np.pi / 3]) / L_M
K_2 = valley * np.array([2 * np.pi / np.sqrt(3), -2 * np.pi / 3]) / L_M

# moire reciprocal vectors
G1_M = np.array([-2 * np.pi / np.sqrt(3), -2 * np.pi]) / L_M
G2_M = np.array([4 * np.pi / np.sqrt(3), 0]) / L_M

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

        p1 = rotate(k_vec + G - K_1, +theta_rad / 2)
        p2 = rotate(k_vec + G - K_2, -theta_rad / 2)

        H[2*i:2*i+2, 2*i:2*i+2] = -hv * (valley * p1[0] * sigma_x + p1[1] * sigma_y) + delta_c2b * sigma_z
        H[off+2*i:off+2*i+2, off+2*i:off+2*i+2] = -hv * (valley * p2[0] * sigma_x + p2[1] * sigma_y) + delta_c2b * sigma_z

        # interlayer
        H[2*i:2*i+2, off+2*i:off+2*i+2] = T1
        H[off+2*i:off+2*i+2, 2*i:2*i+2] = T1.conj().T

        if (m+valley, n) in invL:
            j2 = invL[(m+valley, n)]
            H[2*i:2*i+2, off+2*j2:off+2*j2+2] = T2
            H[off+2*j2:off+2*j2+2, 2*i:2*i+2] = T2.conj().T

        if (m+valley, n+valley) in invL:
            j3 = invL[(m+valley, n+valley)]
            H[2*i:2*i+2, off+2*j3:off+2*j3+2] = T3
            H[off+2*j3:off+2*j3+2, 2*i:2*i+2] = T3.conj().T

    return H


# ================= 3. k-mesh =================
Nk = 30
k_mesh = []
for i in range(Nk):
    for j in range(Nk):
        k_mesh.append((i/Nk)*G1_M + (j/Nk)*G2_M)

num_k = len(k_mesh)

# 截断
n_cut_list = np.arange(1, 242, 1)
M_sym_array = np.zeros(len(n_cut_list))
m_sym_array = np.zeros(len(n_cut_list))

# ================= 4. 主循环（优化版） =================
print("开始计算...")

for idx, (kx, ky) in enumerate(k_mesh):
    print(idx)

    H = build_hamiltonian(kx, ky)
    evals, evecs = np.linalg.eigh(H)

    mid_idx = dim // 2
    E_offset = (evals[mid_idx-1] + evals[mid_idx]) / 2
    evals -= E_offset

    # 一次性算 dH/dk
    dk = 1e-5
    dH_dx = (build_hamiltonian(kx+dk, ky) - build_hamiltonian(kx-dk, ky)) / (2*dk)
    dH_dy = (build_hamiltonian(kx, ky+dk) - build_hamiltonian(kx, ky-dk)) / (2*dk)

    Vx = evecs.conj().T @ dH_dx @ evecs
    Vy = evecs.conj().T @ dH_dy @ evecs

    for i_cut, n_c in enumerate(n_cut_list):

        occ = np.arange(mid_idx - n_c, mid_idx)
        emp = np.arange(mid_idx, mid_idx + n_c)


        # ===== 快速构造 dP（关键优化）=====
        E_n = evals[occ][:, None]
        E_a = evals[emp][None, :]
        denom = E_n - E_a

        denom[np.abs(denom) < 1e-12] = np.inf

        Vx_na = Vx[np.ix_(occ, emp)]
        Vy_an = Vy[np.ix_(emp, occ)]

        Vy_na = Vy[np.ix_(occ, emp)]
        Vx_an = Vx[np.ix_(emp, occ)]

        W_mat = -(Vx_na * Vy_an.T) / (denom ** 2) * (E_n - mu)
        N_mat = -(Vy_na * Vx_an.T) / (denom ** 2) * (E_a - mu)

        M_sym_array[i_cut] += -np.imag(np.sum(W_mat - N_mat))
        m_sym_array[i_cut] += np.imag(np.sum(W_mat + N_mat))


# ================= 5. 结果 =================
M_sym_array /= num_k
M_sym_array /= mu_B
m_sym_array /= num_k
m_sym_array /= mu_B
print(M_sym_array)
print(m_sym_array)

plt.figure(figsize=(7,5))
#plt.plot(n_cut_list, M_sym_array, 'k-o')
plt.plot(n_cut_list, m_sym_array, 'k-o')
plt.xlabel(r'$n_{cut}$')
plt.ylabel(r'$m_{SR}\ / N_{cell} (\mu_B))$')
plt.title('self-rotation mSR')
plt.grid()
plt.show()