import numpy as np 
import matplotlib.pyplot as plt
import time

# Parameters
Lx = 6
nu = 0.01
Nx = 2**8

# Grid
X = np.linspace(-Lx,Lx,Nx,endpoint=False)
u = np.zeros(Nx)

# Initial condition
u = np.exp(-(X)**2/2)
u0 = u
#####################################################################

def transform(u):
    return np.fft.fft(u)

def inverse_transform(u_hat):
    return np.fft.ifft(u_hat).real

def X_derivative(kx,u_hat):
    return 1j*(kx*u_hat.T).T

def wavenumbers():
    kx = kx = np.fft.fftfreq(Nx)*Nx*(2*np.pi/Lx)
    return kx

kx=wavenumbers()

u_hat=transform(u)

dt=0.0001
t=0
tmax=1
t1=time.time()
A = 1 + nu*dt*(kx**2)
while t<tmax:
    nlt = inverse_transform(u_hat)*inverse_transform(X_derivative(kx,u_hat))
    nlt_hat = transform(nlt)
    u_hat /= A 
    u_hat -= nlt_hat*dt/A
    t+=dt
t2=time.time()
print('Elapsed time:',t2-t1)

u=inverse_transform(u_hat)

# plt.figure()
plt.plot(X, u, label='u')
plt.plot(X, u0, 'r', label='u0')
plt.legend()
plt.show()