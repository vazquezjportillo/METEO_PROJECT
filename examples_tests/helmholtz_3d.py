import numpy as np 
from line_profiler import LineProfiler
from scipy.linalg import solve_banded
import time
import matplotlib.pyplot as plt

#region INTRODUCTION
#####################################################################
# Helmholtz equation in 3D
# Laplace(T) + epsilon*T = f(x,y,z)
#
# Previously knowing the analytical solution T, we can compute f(x,y,z)
# for testing our implementation.
#
# Spectral methods for X and Y where X real FFT and central finite 
# differences for Z. 
# Esquema:
# (epsilon - (kx**2 + ky**2))*T_hat1 + (T_hat2 - 2*T_hat1 + T_hat0)/(dz**2)= F_hat1
#####################################################################
#endregion

# Parameters
Lx = 2;             Ly = 2;             Lz = 1
Xc = Lx/2;          Yc = Ly/2;          Zc = Lz/2
Nx = 2**6;          Ny = 2**6;          Nz = 3200
alpha = 0.01;       beta = 0.01;        gamma = 0.01
epsilon = 1

# Grid
x = np.linspace(0,Lx,Nx,endpoint=False)
y = np.linspace(0,Ly,Ny,endpoint=False)
z = np.linspace(0,Lz,Nz)
dz = z[1]-z[0]
X,Y,Z = np.meshgrid(x,y,z,indexing='ij')


# Functions definition
expo = np.exp(-(X-Xc)**2/alpha - (Y-Yc)**2/beta - (Z-Zc)**2/gamma)

X[:] = (-2/alpha + 4*(X-Xc)**2/alpha**2 - 2/beta + 4*(Y-Yc)**2/beta**2 - \
     2/gamma + 4*(Z-Zc)**2/gamma**2)*expo + 2 + epsilon*(expo + Z**2)

Y[:] = expo + Z**2

F=X
T_analytical=Y
Z=0
#####################################################################

def transform(T):
    return np.fft.rfft2(T,axes=(1,0)) 
#Real axe is the X axe (1,0). For having the Y axe: (0,1).

def inverse_transform(T_hat):
    return np.fft.irfft2(T_hat,axes=(1,0))

def wavenumbers():
    kx = np.fft.rfftfreq(Nx)*Nx*(2*np.pi/Lx)
    ky = np.fft.fftfreq(Ny)*Ny*(2*np.pi/Ly)
    return kx, ky  

# You can also define a normalized coordinate 
# x_star = 2*pi*x/Lx and y_star = 2*pi*y/Ly
# BUT YOU NEED TO BE CAREFUL WITH THE DERIVATIVES

#####################################################################

def helmholtz_3d(F,Nx,Ny,Nz,epsilon):
    
    kx,ky = wavenumbers()
    F_hat = transform(F)


    # Esquema:
    # [epsilon-(kx**2 + ky**2)]*I + D]*T_hat= F_hat
    # We need to include in D the CFD for Z:
    # (T_hat2 - 2*T_hat1 + T_hat0)/(dz**2)= F_hat1
    
    # Precompute the banded form of D
    D_banded=np.ones((3,Nz))/dz**2
    D_banded[1]*=-2
    
    D_banded[1,[0,-1]]=1
    D_banded[0,1]=0
    D_banded[2,-2]=0
    base=D_banded[1].copy()
        
    F_hat[:,:,[0,-1]]=0
    F_hat[0,0,-1]=Nx*Ny*1

    for i in range(Nx//2+1):  # We only need to compute the first half of the spectrum
        for j in range(Ny):        
            D_banded[1, 1:-1] = base[1:-1] + (epsilon - kx[i]**2 - ky[j]**2) 
            F_hat[i,j] = solve_banded((1,1), D_banded, F_hat[i, j])
    return inverse_transform(F_hat)

#####################################################################

t1=time.time()
T = helmholtz_3d(F,Nx,Ny,Nz,epsilon)
t2=time.time()
print('Elapsed time:',t2-t1)
print('maximum error:',np.max(np.abs(T-T_analytical)))



# lp = LineProfiler()
# lp_wrapper = lp(helmholtz_3d)
# lp_wrapper(F,Nx,Ny,Nz,epsilon)
# lp.print_stats()


####################################################################

# level = 100

# fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# # Plot the analytical solution
# contour1 = axs[0].contourf(X[:,:,level], Y[:,:,level], T_analytical[:,:,level],np.linspace(0,1.25,20))
# fig.colorbar(contour1, ax=axs[0])
# axs[0].set_title('Analytical Solution')

# # Plot the computed solution
# contour2 = axs[1].contourf(X[:,:,level], Y[:,:,level], T[:,:,level],np.linspace(0,1.25,20))
# fig.colorbar(contour2, ax=axs[1])
# axs[1].set_title('Spectral Methods Solution')

# plt.tight_layout()
# plt.show()