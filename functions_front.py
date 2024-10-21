import numpy as np 

def inverse_transform(F_hat,Nx,Ny,Nz,aliasing=False):
    if aliasing:
        F_hat2=np.zeros((3*Nx//2,Ny,Nz),dtype=np.complex128)
        F_hat2[:Nx//2,:,:]=F_hat[:Nx//2,:,:]
        F_hat2[-Nx//2+1:,:,:]=F_hat[-Nx//2+1:,:,:]
        F_hat = np.zeros((F_hat2.shape[0],3*F_hat2.shape[1]//2,F_hat2.shape[2]),dtype=np.complex128)
        F_hat[:,:F_hat2.shape[1]//2,:]=F_hat2[:,:F_hat2.shape[1]//2,:]
        F_hat[:,-F_hat2.shape[1]//2:,:]=F_hat2[:,-F_hat2.shape[1]//2:,:]
        
        return np.fft.irfft2(F_hat,axes=(1,0))
    else:
        return np.fft.irfft2(F_hat,axes=(1,0))

def inverse_transform2D(F_hat,Nx,Ny,aliasing=False):
    if aliasing:
        F_hat2=np.zeros((3*Nx//2,Ny),dtype=np.complex128)
        F_hat2[:Nx//2,:]=F_hat[:Nx//2,:]
        F_hat2[-Nx//2+1:,:]=F_hat[-Nx//2+1:,:]
        F_hat = np.zeros((F_hat2.shape[0],3*F_hat2.shape[1]//2),dtype=np.complex128)
        F_hat[:,:F_hat2.shape[1]//2]=F_hat2[:,:F_hat2.shape[1]//2]
        F_hat[:,-F_hat2.shape[1]//2:]=F_hat2[:,-F_hat2.shape[1]//2:]
        
        return np.fft.irfft2(F_hat,axes=(1,0))
    else:
        return np.fft.irfft2(F_hat,axes=(1,0))
    
def transform(F, Nx, Ny):
    F_hat = np.fft.rfft2(F, axes=(1, 0))

    zero_array_x = np.zeros_like(F_hat[:1,:])
    F_hat_x = np.concatenate((F_hat[:Nx//2, :], zero_array_x, F_hat[-Nx//2+1:, :]), axis=0)
    
    zero_array_y = np.zeros_like(F_hat_x[:,:1])
    F_hat_xy = np.concatenate((F_hat_x[:, :Ny//2], zero_array_y, F_hat_x[:, -Ny//2+1:]), axis=1)
    
    return F_hat_xy
    
def wavenumbers(Nx,Ny,Lx,Ly):
    kx = np.fft.fftfreq(Nx)*Nx*(2*np.pi/Lx)
    ky = np.fft.fftfreq(Ny)*Ny*(2*np.pi/Ly)
    return kx, ky

