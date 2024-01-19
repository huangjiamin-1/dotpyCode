##########################################################################################################################
# 注释
# 本代码力求解释DBF中的FFT方法与(Bartlett beamforming)直接角度搜索算法的一致性
# 注意：FFT的方法与最大能量的DBF方法是不同的两类算法 
# 第一次用github
# 在均匀阵列下，dbf与fft的结果是一致的，但是在非均匀阵列条件下，结果是不一致的，dbf的方式是遵循了MF的理论，但是fft只是dbf在均匀阵列下的快速实现

import numpy as np
import math
import matplotlib.pyplot as plt

f0 = 10e9
c = 3e8
d = 0.5*c/f0
M = 30
y = np.zeros(shape=(1,M),dtype=complex,order='C')
print("阵列的个数" + str(y.shape))


# 产生信号
for idx in range(M):
    noise = 0 #np.random.normal(0, 0.5, 1)
    y[:,idx] = np.exp(1j*2*np.pi*d/(c/f0)*idx*np.sin(math.radians(20))) + np.exp(1j*2*np.pi*d/(c/f0)*idx*np.sin(math.radians(30))) + noise

# 进行遍历方式的DBF 
degree = range(-90,90,1)
print(len(degree))
z = np.zeros(shape=(1,len(degree)),dtype=complex)

print(z.shape)
for idx in range(len(degree)):
    a = np.exp(1j*2*np.pi*d/(c/f0)*np.arange(M)*np.sin(math.radians(degree[idx])))
    z[:,idx] = np.dot(y[0,:], np.conj(a))

z = z.flatten(order='C')

plt.figure(1)
plt.plot(degree,10*np.log10(np.abs(z)),'g')

plt.figure(1)
fft_sig = np.fft.fftshift(np.fft.fft(y,len(degree)))
lambda_f0 = c/f0
degr2 = np.arcsin(np.arange(-90,90)/180*2)*180/np.pi
plt.plot(degr2,10*np.log10(np.abs(fft_sig.flatten(order='c'))),'r')

# 以下是关于最大会功率的DBF算法
R = np.conj(y.T)@y
print(R.shape)
PDBF = np.zeros(shape=(len(degree)),dtype=complex)
n = np.arange(M)
for idx in range(-90,90,1):
    a_theta = np.exp(-1j*2*np.pi*n*f0/c*np.sin(math.radians(idx))*d)
    PDBF[idx+90] = np.conj(a_theta)@R@a_theta.T/np.dot(a_theta,np.conj(a_theta.T))

plt.figure(1)
plt.plot(degree,10*np.log10(np.abs(PDBF)),'b')
plt.legend(['Bartlett beamforming', 'fft', 'RDBF'])
plt.title('no-FFT&FFT&RDBF_Rmatrix')
plt.show()