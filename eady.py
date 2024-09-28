import numpy as np 
import matplotlib.pyplot as plt
from functions import transform, inverse_transform, wavenumbers, X_derivative, Y_derivative
from scipy.linalg import solve_banded
from mpl_toolkits.mplot3d import Axes3D

# Parameters
Lx = 2;             Ly = 2;             Lz = 100
Nx = 2**4;          Ny = 2**2;          Nz = 4
Xc = Lx/2;          Yc = Ly/2;          Zc = Lz/2
alpha = 0.1;       kappa = 0.1;        gamma = 10    
f0 = 0.01        
rho0 = 1
Er = 7.2921e-5
Omega = 6371000
N2 = 1            
nu = 1 # Damping term in potential vorticity
Umax = 3
t = 0;              tmax = 1;          dt=0.1

# Grid
x = np.linspace(0,Lx,Nx,endpoint=False)
y = np.linspace(0,Ly,Ny,endpoint=False)
z = np.linspace(0,Lz,Nz)
dz = z[1]-z[0]
X,Y,Z = np.meshgrid(x,y,z,indexing='ij')

def beta(y):
    return 2*Omega/Er*np.cos(y)

q00 = -np.ones((Nx,Ny,Nz))*2*np.exp(-(X-Xc)**2/alpha-(Y-Yc)**2/kappa)*(1/alpha + 1/kappa + 2*(X-Xc)**2/alpha**2 + 2*(Y-Yc)**2/kappa**2) 
q00[:,:,0] = 0
q00[:,:,-1] = -y*Umax/Lz

kx,ky = wavenumbers(Nx,Ny,Lx,Ly)
psi_hat = transform(np.zeros((Nx,Ny,Nz)))[:Nx//2+1,:,:]
q00_hat = transform(q00)[:Nx//2+1,:,:]
q0_hat = q00_hat.copy()
q_hat = q0_hat.copy()

D_banded=np.ones((3,Nz))*(f0**2/N2)/dz**2
D_banded[1]*=-2

D_banded[1,0] = -1/dz
D_banded[1,-1] = 1/dz
D_banded[0,0] = 1/dz
D_banded[2,-1] = -1/dz
D_banded[0,-1] = 0
D_banded[2,0] = 0
base=D_banded[1].copy()

KX,KY = np.meshgrid(kx[:Nx//2+1],ky,indexing='ij')
KX = np.repeat(KX[:, :, np.newaxis], Nz, axis=2)
KY = np.repeat(KY[:, :, np.newaxis], Nz, axis=2)

def nonlinear_term(q0_hat):
    
    for i in range(Nx//2+1):  # We only need to compute the first half of the spectrum
        for j in range(Ny):        
            D_banded[1, 1:-1] = base[1:-1] + (- kx[i]**2 - ky[j]**2)
            psi_hat[i,j] = solve_banded((1,1), D_banded, q0_hat[i, j])
            
    v_hat = -X_derivative(KX,KY,Nx,psi_hat)  
    u_hat = Y_derivative(KX,KY,Nx,psi_hat)

    nlt = inverse_transform(u_hat,True)*inverse_transform(X_derivative(KX,KY,Nx,q0_hat),True) + \
        inverse_transform(v_hat,True)*(inverse_transform(Y_derivative(KX,KY,Nx,q0_hat),True) + beta(Y))
        
    return transform(nlt), v_hat, u_hat 



for i in range(1, Nz - 1):  # Skip the first and last point
    q0_hat = q00_hat[:,:,i] + dt * (
        nonlinear_term(q00_hat)[0][:,:,i] + 
        nu * (-KX**2 - KY**2) * q00_hat[:,:,i] + 
        nu * (q00_hat[:,:,i+1] - 2 * q00_hat[:,:,i] + q00_hat[:,:,i-1]) / dz**2
    )

for n in range(1, tmax//dt - 1):
    for i in range(1, Nz - 1):  # Skip the first and last point
        # Update using the Adams-Bashforth formula
        q_hat[:,:,i] = q0_hat[:,:,i] + (dt/2)*(
            3*(
                nonlinear_term(q0_hat)[0] + 
                nu*(-KX**2 -KY**2)*q0_hat + 
                nu*(q0_hat[:,:,i+1] - 2*q0_hat[0:,:,i] + q0_hat[:,:, i-1]) /dz**2
            ) - (
                nonlinear_term(q00_hat)[0] + 
                nu*(-KX**2 -KY**2)*q00_hat + 
                nu*(q00_hat[:,:,i+1] - 2*q00_hat[0:,:,i] + q00_hat[:,:, i-1]) /dz**2
            )
        )
        q00_hat = q0_hat.copy()
        q0_hat = q_hat.copy()
















