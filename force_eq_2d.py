import numpy as np 
import matplotlib.pyplot as plt
import time
    
# Parameters
Lx = 2
Ly = 1
Nx = 2**8
Ny = 2**6
sin = np.sin
cos = np.cos
pi = np.pi

# Grid
x = np.linspace(0,Lx,Nx,endpoint=False)
y = np.linspace(0,Ly,Ny,endpoint=False)

X,Y = np.meshgrid(x,y,indexing='ij')
T = np.zeros((Nx,Ny))
T1 = sin(4*pi*X)*cos(2*pi*Y) # wavenumber (4,1)
T2 = sin(12*pi*X + 5)*cos(6*pi*Y - 3) # wavenumber (12,3)
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

#You can also define a normalized coordinate x_star = 2*pi*x/Lx and y_star = 2*pi*y/Ly
#BUT YOU NEED TO BE CAREFUL WITH THE DERIVATIVES


kx,ky=wavenumbers()
F_hat=transform(F)

KX, KY = np.meshgrid(kx, ky, indexing='ij')

ksq = KX**2 + KY**2
ksq[0,0] = 1

T_hat = -F_hat/ksq   
T_hat[0,0] = 0

T=inverse_transform(T_hat)
#####################################################################


fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot the analytical solution
contour1 = axs[0].contourf(X, Y, T_analytical)
fig.colorbar(contour1, ax=axs[0])
axs[0].set_title('Analytical Solution')

# Plot the computed solution
contour2 = axs[1].contourf(X, Y, T)
fig.colorbar(contour2, ax=axs[1])
axs[1].set_title('Spectral Methods Solution')

plt.tight_layout()
plt.show()