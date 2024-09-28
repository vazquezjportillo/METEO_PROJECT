import numpy as np 
import matplotlib.pyplot as plt
import time

# Parameters
Lx = 6
nu = 0.01
Nx = 2**8
dt=0.001
t=0
tmax=10

# Grid
X = np.linspace(-Lx,Lx,Nx,endpoint=False)
u = np.zeros(Nx)

# Initial condition
u = np.exp(-(X)**2/2)
u0 = u
#####################################################################

def transform(u):
    u_hat = np.fft.fft(u)
    return np.concatenate((u_hat[:Nx//2],[0],u_hat[-Nx//2+1:]))


# To deal with aliasing we are going to:
# 1. Go to Spectral space
# 2. Extend the spectral espace up to 3/2 (Nx//2)
# 4. Return to physical space
# 5. Compute the derivative in physical space
# 6. Return to spectral space
# 7. Truncate the spectral space to Nx//2


def inverse_transform(u_hat,aliasing=False):
    if aliasing:
        u_hat2=np.zeros(3*Nx//2,dtype=np.complex128)
        u_hat2[:Nx//2]=u_hat[:Nx//2]
        u_hat2[-Nx//2+1:]=u_hat[-Nx//2+1:]
        return np.fft.ifft(u_hat2).real
    else:
        return np.fft.ifft(u_hat).real

def X_derivative(kx,u_hat):
    return 1j*kx*u_hat

def wavenumbers():
    return np.fft.fftfreq(Nx)*Nx*(2*np.pi/Lx)
    
kx=wavenumbers()

u_hat=transform(u)


t1=time.time()
A = 1 + nu*dt*kx**2
while t<tmax:
    nlt = inverse_transform(u_hat,True)*inverse_transform(X_derivative(kx,u_hat),True)
    nlt_hat = transform(nlt)
    u_hat -= nlt_hat*dt
    u_hat /= A 
    t+=dt
t2=time.time()
print('Elapsed time:',t2-t1)

u=inverse_transform(u_hat)

# plt.figure()
plt.plot(X, u, label='u')
plt.plot(X, u0, 'r', label='u0')
plt.legend()
plt.show()