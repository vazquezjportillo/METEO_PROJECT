import numpy as np 
import matplotlib.pyplot as plt
import time

# Parameters
Lx = 2*np.pi
Ly = 2*np.pi
alpha = 1.0
beta = 2
Nx = 2**8
Ny = Nx

# Grid
x = np.linspace(0,Lx,Nx,endpoint=False)
y = np.linspace(0,Ly,Ny,endpoint=False)

X,Y = np.meshgrid(x,y,indexing='ij')
T = np.zeros((Nx,Ny))

# Initial condition
R=np.sqrt((X-np.pi)**2+(Y-1)**2)
T[R<0.5]=1

# plt.contourf(X,Y,T);plt.colorbar()
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
    kx = np.fft.fftfreq(Nx)*Nx
    ky = np.fft.fftfreq(Ny)*Ny
    return kx,ky

def basic_eq(dt,alpha,beta,KX,KY):
    return 1+dt*alpha*(KX**2+KY**2)

def linear_term(dt,alpha,beta,KX,KY):
    return 1+dt*(alpha*(KX**2+KY**2) + 1j * beta*(KX+KY))
    
kx,ky=wavenumbers()

T_hat=transform(T)


dt=0.9*8/(alpha*(Nx**2+Ny**2))*100
t=0
tmax=1
KX,KY=np.meshgrid(kx,ky,indexing='ij')
S=linear_term(dt,alpha,beta,KX,KY)
t1=time.time()
while t<tmax:
    T_hat /=  S
    t+=dt
t2=time.time()
print('Elapsed time:',t2-t1)

T=inverse_transform(T_hat)

# plt.figure()
plt.contourf(X,Y,T);plt.colorbar();plt.show()