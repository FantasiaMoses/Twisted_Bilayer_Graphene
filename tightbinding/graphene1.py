# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 石墨烯晶格能带色散关系

#parameter
a=1.0
a1=a/2*np.array([3,np.sqrt(3)])
a2=a/2*np.array([3,-np.sqrt(3)])
N=1000
kx_start=ky_start=-1*np.pi/a
kx_end=ky_end=1*np.pi/a
kx=np.linspace(kx_start,kx_end,N)
ky=np.linspace(ky_start,ky_end,N)
kxx,kyy=np.meshgrid(kx,ky)
eigen_val_matrix = np.zeros([N,N,2])
delta=0#onsite energy

def Hamiltonian_NN(kx,ky,delta):
    H_NN = np.zeros([2,2],dtype=complex) #2*2 matrix for Hamiltonian
    k = np.array([kx,ky])
    H_NN[0,0]=delta #diagonal element
    H_NN[0,1]=1+np.exp(-1j*k.dot(a1))+np.exp(-1j*k.dot(a2))
    H_NN[1,0]=np.conjugate(H_NN[0,1])
    H_NN[1,1]=-delta #diagonal element
    return H_NN

for i in range(N):
    for j in range(N):
        H_NN = Hamiltonian_NN(kxx[i,j],kyy[i,j],delta)
        eigen_val,eigen_wave = np.linalg.eig(H_NN)
        eigen_val = np.sort(np.real(eigen_val))
        eigen_val_matrix[i,j,:]=eigen_val

fig = plt.figure(figsize=(5,8),dpi=100,layout='constrained')
ax = fig.add_subplot(projection='3d')
ax.plot_surface(kxx, kyy, eigen_val_matrix[:,:,0],cmap=cm.RdYlGn, linewidth=0, antialiased=False)
ax.plot_surface(kxx, kyy, eigen_val_matrix[:,:,1],cmap=cm.RdYlGn, linewidth=0, antialiased=False)
ax.set_aspect('equal')
plt.show()