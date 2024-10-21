import numpy as np 
import matplotlib.pyplot as plt
from functions_front import transform, inverse_transform, wavenumbers, inverse_transform2D
from scipy.linalg import solve_banded

############################################################################
# Parameters
############################################################################

Lx = 9e6;           Ly = 9e6;                 Lz = 1e4
Nx = 2**6;          Ny = 2**2;                  Nz = 32
t = 0;              tmax = 3600*24*4;         dt=1000

N = 0.02
nu = 0     # Damping term in potential vorticity
f0=1e-4
Umax=20
Ld=N*Lz/f0
Bu = Ld/Lx
print('Ld=',Ld)
print('Bu=',Bu)
print('Ro=',Umax/f0/Lx)
print('Fr=',Umax/N/Lz)
#####################
#Normalization
Xc = Lx/2;          Yc= Ly/2;               Zc = Lz/2
alphax = Lx/20;     alphay = Ly/20;       alphaz = Lz/100
############################################################################

# Grid
x = np.linspace(0,Lx,Nx,endpoint=False)
y = np.linspace(0,Ly,Ny,endpoint=False)
z = np.linspace(0,Lz,Nz)
dz = z[1]-z[0]
X,Y = np.meshgrid(x,y,indexing='ij')

U0=Umax/Lz*z
dU0dz=Umax/Lz*np.ones_like(z)

############################################################################

kx,ky = wavenumbers(Nx,Ny,Lx,Ly)

# initial conditions
q = np.zeros((Nx,Ny,Nz))
DPSI_bottom=np.zeros((Nx,Ny))
DPSI_top=np.zeros((Nx,Ny))

Z = Bu*z/Lz
n = 1/Bu * np.sqrt(((Bu/2 - np.tanh(Bu/2))*((1/np.tanh(Bu/2))-Bu/2)))

for i in range(Nz):
    q[:,:,i]+= 1e-10*np.exp(-(z[i]-Zc)**2/2/alphaz**2)*np.sin(kx[2]*X)
    # q[:,:,i]+= 1e-10*N*(-(1 - Bu/2*(1/np.tanh(Bu/2)))*np.sinh(Z[i])*np.cosh(kx[1]*X) - n*Bu*np.cosh(Z[i])*np.sin(kx[1]*X))
############################################################################

# Fourier analysis
q_hat = transform(q,Nx,Ny)
DPSI_bottom_hat = transform(DPSI_bottom,Nx,Ny)
DPSI_top_hat=transform(DPSI_top,Nx,Ny)
u_hat=np.zeros_like(q_hat)
v_hat=np.zeros_like(q_hat)
w_hat=np.zeros_like(q_hat)
bouyancy_hat = np.zeros_like(q_hat)
psi=np.zeros_like(q_hat)


############################################################################

# Define the banded matrix for the second derivative

D_banded=np.ones((3,Nz))*(f0/N)**2/dz**2
D_banded[1]*=-2

# bottom boundary condition
D_banded[1,0] = -1/dz
D_banded[0,1] = 1/dz

# top boundary condition
D_banded[1,-1] = 1/dz
D_banded[2,-2] = -1/dz

D_banded_zerowavenumber=D_banded.copy()
D_banded_zerowavenumber[0,2]=0
D_banded_zerowavenumber[1,1]=0
D_banded_zerowavenumber[2,0]=1

base=D_banded[1,1:-1].copy()

############################################################################

def linear_term(q_hat,DPSI_bottom_hat,DPSI_top_hat):
    
    B_hat=np.zeros(Nz,dtype=np.complex128)
    psi_hat=np.zeros_like(B_hat)
    for i in range(Nx//2+1):  # We only need to compute the first half of the spectrum
        for j in range(Ny): 
            if i+j>0:       
                D_banded[1, 1:-1] = base - (kx[i]**2 + ky[j]**2)
                print(q_hat.shape)
                B_hat[1:-1] = q_hat[i,j,1:-1]
                B_hat[0] = DPSI_bottom_hat[i,j]
                B_hat[-1] = DPSI_top_hat[i,j]
                psi_hat[:] = solve_banded((1,1), D_banded, B_hat)
                u_hat[i,j]=-1j*ky[j]*psi_hat
                v_hat[i,j]=1j*kx[i]*psi_hat
                psi[i,j]=psi_hat
            else:
                u_hat[i,j]=0
                v_hat[i,j]=0

    nlt_q = U0*inverse_transform((1j*kx*q_hat.T).T,Nx,Ny,Nz,True)
    
    nlt_bottom = U0[0] * inverse_transform2D((1j*kx*DPSI_bottom_hat.T).T,Nx,Ny,True)
    nlt_bottom += inverse_transform2D(v_hat[:,:,0],Nx,Ny,True) * (-dU0dz[0])
        
    nlt_top = U0[-1] * inverse_transform2D((1j*kx*DPSI_top_hat.T).T,Nx,Ny,True)
    nlt_top += inverse_transform2D(v_hat[:,:,-1],Nx,Ny,True) * (-dU0dz[-1])
            
    return transform(nlt_q,Nx,Ny), transform(nlt_bottom,Nx,Ny), transform(nlt_top,Nx,Ny), psi

############################################################################


def nonlinear_term(q_hat,DPSI_bottom_hat,DPSI_top_hat):
    
    B_hat=np.zeros(Nz,dtype=np.complex128)
    psi_hat=np.zeros_like(B_hat)
    for i in range(Nx//2+1):  # We only need to compute the first half of the spectrum
        for j in range(Ny): 
            if i+j>0:       
                D_banded[1, 1:-1] = base - (kx[i]**2 + ky[j]**2)
                B_hat[1:-1] = q_hat[i,j,1:-1]
                B_hat[0] = DPSI_bottom_hat[i,j]
                B_hat[-1] = DPSI_top_hat[i,j]
                psi_hat[:] = solve_banded((1,1), D_banded, B_hat)
                u_hat[i,j]=-1j*ky[j]*psi_hat
                v_hat[i,j]=1j*kx[i]*psi_hat
                w_hat[i,j] = 1j*kx[i]*v_hat[i,j] - 1j*ky[j]*u_hat[i,j]
                psi[i,j]=psi_hat
                
            else:
                u_hat[i,j]=U0*Nx*Ny
                v_hat[i,j]=0
                w_hat[i,j]=0
                
                
    nlt_q = inverse_transform(u_hat,Nx,Ny,Nz,True)*inverse_transform((1j*kx*q_hat.T).T,Nx,Ny,Nz,True)
    nlt_q += inverse_transform(v_hat,Nx,Ny,Nz,True)*inverse_transform(np.moveaxis(1j*ky*np.moveaxis(q_hat,1,-1),-1,1),Nx,Ny,Nz,True)
    
    nlt_bottom = inverse_transform2D(u_hat[:,:,0],Nx,Ny,True) * inverse_transform2D((1j*kx*DPSI_bottom_hat.T).T,Nx,Ny,True)
    nlt_bottom += inverse_transform2D(v_hat[:,:,0],Nx,Ny,True) * (inverse_transform2D(1j*ky*DPSI_bottom_hat,Nx,Ny,True)-dU0dz[0])
        
    nlt_top = inverse_transform2D(u_hat[:,:,-1],Nx,Ny,True) * inverse_transform2D((1j*kx*DPSI_top_hat.T).T,Nx,Ny,True)
    nlt_top += inverse_transform2D(v_hat[:,:,-1],Nx,Ny,True) * (inverse_transform2D(1j*ky*DPSI_top_hat,Nx,Ny,True)-dU0dz[-1])
    
    return transform(nlt_q,Nx,Ny), transform(nlt_bottom,Nx,Ny), transform(nlt_top,Nx,Ny), psi

############################################################################

def Euler(q_hat,DPSI_bottom_hat,DPSI_top_hat):
    NLT_q_hat,NLT_bottom_hat,NLT_top_hat, psi_hat = nonlinear_term(q_hat,DPSI_bottom_hat,DPSI_top_hat)
    q_hat-=dt*NLT_q_hat
    for i in range(Nz):
        q_hat[:,:,i]/=1+dt*nu*(np.repeat(kx[:,np.newaxis]**2,Ny,1)+ky**2)


    # Buoyancy: f0 * d(psi)/dz
    
    for i in range(1, Nz-1):
        bouyancy_hat[:, :, i] = f0 * (psi_hat[:, :, i+1] - psi_hat[:, :, i-1]) / (2 * dz)  # Central difference

    # Boundary conditions for buoyancy
    bouyancy_hat[:, :, 0] = f0 * (psi_hat[:, :, 1] - psi_hat[:, :, 0]) / dz  # One-sided difference at bottom
    bouyancy_hat[:, :, -1] = f0 * (psi_hat[:, :, -1] - psi_hat[:, :, -2]) / dz  # One-sided difference at top

    DPSI_bottom_hat-=dt*NLT_bottom_hat
    DPSI_top_hat-=dt*NLT_top_hat
    return psi_hat
############################################################################
    
for s in range(int(tmax//dt)):
    psi_hat = Euler(q_hat, DPSI_bottom_hat, DPSI_top_hat)

u = inverse_transform(u_hat,Nx,Ny,Nz,True)
w = inverse_transform(w_hat,Nx,Ny,Nz,True)
bouyancy = inverse_transform(bouyancy_hat,Nx,Ny,Nz,True)
q = inverse_transform(q_hat,Nx,Ny,Nz,True)
psi = inverse_transform(psi_hat,Nx,Ny,Nz,True)

plt.subplot(151)
v=inverse_transform(v_hat,Nx,Ny,Nz,True)
plt.contourf(w[:,Ny//2,:].T,levels=18)
plt.xlabel('x')
plt.ylabel('z')
plt.title('w')
plt.colorbar()

plt.subplot(152)
plt.contourf(u[:,Ny//2,:].T,levels=18)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('z')
plt.title('u')

plt.subplot(153)
plt.contourf(v[:,Ny//2,:].T,levels=18)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('z')
plt.title('v')

plt.subplot(154)
plt.contourf(bouyancy[:,Ny//2,:].T,levels=18)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('z')
plt.title('b')

plt.subplot(155)
plt.contourf(psi[:,Ny//2,:].T,levels=18)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('z')
plt.title('$\psi$')
plt.show()
