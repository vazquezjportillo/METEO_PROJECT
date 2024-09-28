import numpy as np 
import matplotlib.pyplot as plt
import time

error = np.zeros(8)
for i in range(1, 8):

    # Parameters
    Lx = 2
    Ly = 1
    Nx = 2**i
    Ny = Nx
    sin = np.sin
    cos = np.cos
    pi = np.pi

    # Grid
    x = np.linspace(0,Lx,Nx,endpoint=False)
    y = np.linspace(0,Ly,Ny,endpoint=False)

    X,Y = np.meshgrid(x,y,indexing='ij')
    T = np.zeros((Nx,Ny))
    T1 = sin(4*pi*X)*cos(2*pi*Y)
    T2 = sin(12*pi*X + 5)*cos(6*pi*Y - 3)
    T_analytical = T1 + T2
    F = (-(4*pi)**2 - (2*pi)**2)*T1 + (-(12*pi)**2 - (6*pi)**2)*T2

    #####################################################################

    def transform(T):
        return np.fft.fft2(T)

    def inverse_transform(T_hat):
        return np.fft.ifft2(T_hat).real

    def X_derivative(kx,ky,T_hat):
        return 1j*(kx*T_hat.T).T

    def Y_derivative(kx,ky,T_hat):
        return 1j*ky*T_hat

    def wavenumbers():
        kx = np.fft.fftfreq(Nx)*Nx*(2*pi/Lx)
        ky = np.fft.fftfreq(Ny)*Ny*(2*pi/Ly)
        return kx, ky



    kx,ky=wavenumbers()
    T_hat=transform(T)
    F_hat=transform(F)

    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    ksq = KX**2 + KY**2
    ksq[0,0] = 1

    T_hat = -F_hat/ksq   
    T=inverse_transform(T_hat)

    error[i] = np.linalg.norm(T_analytical - T)
    
Nx_values = [2**i for i in range(1, 8)]

plt.plot(Nx_values, error[1:], marker='o', linestyle='-', color='b', label='Error Norm')
plt.xlabel('Nx')
plt.ylabel('Error Norm')
plt.title('Error Norm vs Nx')
plt.grid(True)
plt.legend()
plt.show()