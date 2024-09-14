import numpy as np 
import matplotlib.pyplot as plt
import time
    
# Parameters
Lx = 2
Ly = 1
Lz = 1
Xc = Lx/2
Yc = Ly/2
Zc = Lz/2
Nx = 2**6
Ny = 2**6
Nz = 2**6
alpha = 0.01
beta = 0.01
gamma = 0.01
epsilon = 1
dz = 0.001 

# Grid
x = np.linspace(0,Lx,Nx,endpoint=False)
y = np.linspace(0,Ly,Ny,endpoint=False)
z = np.linspace(0,Lz,Nz,endpoint=False)

X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
T = np.zeros((Nx,Ny))
# T1 = np.sin(4*np.pi*X)*np.cos(2*np.pi*Y) # wavenumber (4,1)
# T2 = np.sin(12*np.pi*X + 5)*np.cos(6*np.pi*Y - 3) # wavenumber (12,3)
# T_analytical = T1 + T2
# F = (-(4*np.pi)**2 - (2*np.pi)**2)*T1 + (-(12*np.pi)**2 - (6*np.pi)**2)*T2 + epsilon*(T1 + T2)

expo = np.exp(-(X-Xc)**2/alpha - (Y-Yc)**2/beta - (Z-Zc)**2/gamma)
T_analytical = expo + Z**2
F = (-2/alpha + 4*(X-Xc)**2/alpha**2 - 2/beta + 4*(Y-Yc)**2/beta**2 - 2/gamma + 4*(Z-Zc)**2/gamma**2)*expo + 2 + epsilon*(expo + Z**2)
#####################################################################

def transform(T):
    return np.fft.fftn(T)

def inverse_transform(T_hat):
    return np.fft.ifftn(T_hat).real

def wavenumbers():
    kx = np.fft.fftfreq(Nx)*Nx*(2*np.pi/Lx)
    ky = np.fft.fftfreq(Ny)*Ny*(2*np.pi/Ly)
    kz = np.fft.fftfreq(Nz)*Nz*(2*np.pi/Lz)  
    return kx, ky, kz  

#You can also define a normalized coordinate x_star = 2*pi*x/Lx and y_star = 2*pi*y/Ly
#BUT YOU NEED TO BE CAREFUL WITH THE DERIVATIVES


kx,ky,kz=wavenumbers()
F_hat=transform(F)
T_hat=np.zeros((Nx,Ny,Nz),dtype=complex)



KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

ksq = KX**2 + KY**2 
b = epsilon - ksq - 2/dz**2
a = 1/dz**2

for i in range(Nx):
    for j in range(Ny):
        A = np.zeros((Nz, Nz))
        
        for k in range(1, Nz-1):
            A[k, k-1] = a          
            A[k, k] = b[i,j,k] 
            A[k, k+1] = a           
        
        # Apply boundary conditions
        A[0, 0] = 1
        A[-1, -1] = 1
        A[0, 1:] = 0
        A[-1, :-1] = 0
        
        T_hat_slice = np.linalg.solve(A, F_hat[i, j, :])
        T_hat[i,j,:] = T_hat_slice

T=inverse_transform(T_hat)
#####################################################################


fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot the analytical solution
contour1 = axs[0].contourf(X[:,:,40], Y[:,:,40], T_analytical[:,:,40])
fig.colorbar(contour1, ax=axs[0])
axs[0].set_title('Analytical Solution')

# Plot the computed solution
contour2 = axs[1].contourf(X[:,:,40], Y[:,:,40], T[:,:,40])
fig.colorbar(contour2, ax=axs[1])
axs[1].set_title('Spectral Methods Solution')

plt.tight_layout()
plt.show()

# Additional diagnostics
print("Maximum and minimum values in the computed solution:")
print(f"Max: {np.max(T)}")
print(f"Min: {np.min(T)}")

# Additional diagnostics
print("Maximum and minimum values in the analytic solution:")
print(f"Max: {np.max(T_analytical)}")
print(f"Min: {np.min(T_analytical)}")