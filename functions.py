import numpy as np 

def inverse_transform(F_hat,aliasing=False):
    if aliasing:
        F_hat2=np.zeros((3*F_hat.shape[0]//2,F_hat.shape[1],F_hat.shape[2]),dtype=np.complex128)
        F_hat2[:F_hat.shape[0]//2,:,:]=F_hat[:F_hat.shape[0]//2,:,:]
        F_hat2[-F_hat.shape[0]//2+1:,:,:]=F_hat[-F_hat.shape[0]//2+1:,:,:]
        return np.fft.irfft2(F_hat,axes=(1,0))
    else:
        return np.fft.irfft2(F_hat,axes=(1,0))
    
def transform(F):
    return np.fft.rfft2(F,axes=(1,0)) 
    
def X_derivative(kx,ky,Nx,T_hat):
    return 1j*(kx*T_hat)

def Y_derivative(kx,ky,Nx,T_hat):
    return 1j*ky*T_hat

def wavenumbers(Nx,Ny,Lx,Ly):
    kx = np.fft.fftfreq(Nx)*Nx*(2*np.pi/Lx)
    ky = np.fft.fftfreq(Ny)*Ny*(2*np.pi/Ly)
    return kx, ky