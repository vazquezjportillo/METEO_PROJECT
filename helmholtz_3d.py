import numpy as np 
from line_profiler import LineProfiler
from scipy.linalg import solve_banded


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
Lx = 2;             Ly = 1;             Lz = 1
Xc = Lx/2;          Yc = Ly/2;          Zc = Lz/2
Nx = 2**6;          Ny = 2**6;          Nz = 200
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
T_analytical = expo + Z**2
F = (-2/alpha + 4*(X-Xc)**2/alpha**2 - 2/beta + 4*(Y-Yc)**2/beta**2 - \
     2/gamma + 4*(Z-Zc)**2/gamma**2)*expo + 2 + epsilon*(expo + Z**2)

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
    T_hat = np.zeros((Nx//2+1,Ny,Nz),dtype=complex)

    # Esquema:
    # [epsilon-(kx**2 + ky**2)]*I + D]*T_hat= F_hat
    # We need to include in D the CFD for Z:
    # (T_hat2 - 2*T_hat1 + T_hat0)/(dz**2)= F_hat1
    D = np.diag(-2/dz**2*np.ones(Nz)) + np.diag(1/dz**2*np.ones(Nz-1),1) + \
        np.diag(1/dz**2*np.ones(Nz-1),-1) 
    # Defines a tridiagonal matrix with -2/dz**2 in the diagonal and 1/dz**2 in the upper and lower diagonals 

    # BC:
    D[[0,-1],]=0  # First and last rows are zero
    D[0,0]=1
    D[-1,-1]=1
    
        # Precompute the banded form of D
    D_banded = np.zeros((3, Nz))
    D_banded[0, 1:] = np.diag(D, k=1)    # Upper diagonal, shifted right
    D_banded[1, :] = np.diag(D)          # Main diagonal
    D_banded[2, :-1] = np.diag(D, k=-1)  # Lower diagonal, shifted left


    I = np.eye(Nz)
    I[[0,-1]]=0
 
    F_hat[:,:,[0,-1]]=0
    F_hat[0,0,-1]=Nx*Ny*1

    for i in range(Nx//2+1):  # We only need to compute the first half of the spectrum
        for j in range(Ny):        
            D_banded[1, :] = np.diag(D) + (epsilon - kx[i]**2 - ky[j]**2) * np.diag(I)
            T_hat[i,j] = solve_banded((1,1), D_banded, F_hat[i, j])

    return inverse_transform(T_hat)

#####################################################################

T = helmholtz_3d(F,Nx,Ny,Nz,epsilon)
print('maximum error:',np.max(np.abs(T-T_analytical)))

lp = LineProfiler()
lp_wrapper = lp(helmholtz_3d)
lp_wrapper(F,Nx,Ny,Nz,epsilon)
lp.print_stats()