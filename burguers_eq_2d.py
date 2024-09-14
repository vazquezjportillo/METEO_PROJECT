import numpy as np 
import matplotlib.pyplot as plt
import time

# Parameters
Lx = 6
Ly = 6
nu = 0.01
Nx = 2**8
Ny = 2**8

# Grid
x = np.linspace(0,Lx,Nx,endpoint=False)
y = np.linspace(0,Ly,Ny,endpoint=False)

X,Y = np.meshgrid(x,y,indexing='ij')
u = np.zeros((Nx,Ny))

# Initial condition
u = np.exp(-((X-Lx/2)**2 + (Y-Ly/2)**2)/2)
u0 = u

plt.contourf(X, Y, u, levels=50, cmap='viridis')
plt.colorbar(label='u')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Contour plot of u')
plt.show()
#####################################################################

def transform(u):
    return np.fft.fft2(u)

def inverse_transform(u_hat):
    return np.fft.ifft2(u_hat).real

def X_derivative(kx,ky,u_hat):
    return 1j*(kx*u_hat.T).T

def Y_derivative(kx,ky,u_hat):
    return 1j*ky*u_hat

def wavenumbers():
    kx = np.fft.fftfreq(Nx)*Nx*(2*np.pi/Lx)
    ky = np.fft.fftfreq(Ny)*Ny*(2*np.pi/Ly)  
    return kx, ky

kx,ky=wavenumbers()

u_hat=transform(u)

KX, KY = np.meshgrid(kx, ky, indexing='ij')
ksq = KX**2 + KY**2

dt=0.01
t=0
tmax=1
t1=time.time()
A = 1 + nu*dt*(ksq**2)
while t<tmax:
    nlt = inverse_transform(u_hat)*inverse_transform(X_derivative(kx,ky,u_hat) + Y_derivative(kx,ky,u_hat))
    nlt_hat = transform(nlt)
    u_hat /= A 
    u_hat -= nlt_hat*dt/A
    t+=dt
t2=time.time()
print('Elapsed time:',t2-t1)

u=inverse_transform(u_hat)


# 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, u, cmap='viridis')

# Add a color bar
fig.colorbar(surf, ax=ax, label='u')

# Labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('u')
ax.set_title('3D Surface plot of u')

plt.show()

