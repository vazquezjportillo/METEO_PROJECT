import numpy as np

def poisson_fourier(Lx,Ly,Nx,Ny,F):
    
    kx = np.fft.rfftfreq(Nx)*Nx*(2*np.pi/Lx)
    ky = np.fft.fftfreq(Ny)*Ny*(2*np.pi/Ly)
    F_hat=np.fft.rfft2(F,axes=(1,0)) #real axis last
    
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    
    ksq = KX**2 + KY**2
    ksq[0,0] = 1

    T_hat = -F_hat/ksq   
    T_hat[0,0] = 0
    
    return np.fft.irfft2(T_hat,axes=(1,0))
