import numpy as np 

def inverse_transform(F_hat,dealiasing=False):
    PP=F_hat.shape 
    Nx,Ny=PP[0],PP[1]   
    if dealiasing:
        if len(PP)==3:        
            F_hat2=np.zeros((3*Nx//2,3*Ny//2,PP[2]),dtype=np.complex128)
        else:
            F_hat2=np.zeros((3*Nx//2,3*Ny//2),dtype=np.complex128)
        F_hat2[:Nx,:Ny//2]=F_hat[:,:Ny//2]
        F_hat2[:Nx,-Ny//2:]=F_hat[:,Ny//2:]
    else:
        F_hat2=F_hat
    return np.fft.irfft2(F_hat2,axes=(1,0))
     
def transform(F,dealiasing=False):
    F_hat=np.fft.rfft2(F, axes=(1, 0))

    if dealiasing:
        PP=F_hat.shape
        Nx,Ny=PP[0],PP[1]
        kx_max=(2*Nx)//3+1
        F_hat=F_hat[:kx_max]
        Ny=(Ny*2)//3
        
        if len(PP)==2:
            F_hat2=np.zeros((kx_max,Ny),dtype=np.complex128)
        else:
            F_hat2=np.zeros((kx_max,Ny,PP[2]),dtype=np.complex128)

        F_hat2[:,:Ny//2]=F_hat[:,:Ny//2]
        F_hat2[:,Ny//2:]=F_hat[:,-Ny//2:]
        
    else:
        F_hat2=F_hat 
    PP=F_hat2.shape
    F_hat2[:,PP[1]//2]=0  
    F_hat2[-1]=0       
    return F_hat2

    

def wavenumbers(Nx,Ny,Lx,Ly):
    kx = np.fft.rfftfreq(Nx)*Nx*(2*np.pi/Lx)
    ky = np.fft.fftfreq(Ny)*Ny*(2*np.pi/Ly)
    return kx, ky


if __name__ == "__main__":
    Nx=2**8
    Ny=2**5
    Nz=3
    
    A=np.random.rand(Nx,Ny,Nz)    
    A_hat=transform(A, False)
    
    A2=inverse_transform(A_hat,True)
    A_hat2=transform(A2,True)
    print(np.allclose(A_hat,A_hat2))



#pyfftw
#ray

