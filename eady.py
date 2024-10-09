import numpy as np 
import matplotlib.pyplot as plt
from functions import transform, inverse_transform, wavenumbers, inverse_transform2D
from scipy.linalg import solve_banded

############################################################################
# Parameters
############################################################################
Lz = 1e4
Delta = 1
#####################
Lx = 8e6;           Ly = 8e6;              
Nx = 2**4;          Ny = 2**2;          Nz = 50
t = 0;              tmax = 3600*24*0.2;   dt=3600
#####################
#Normalization
Lx /= Lz; Ly /= Lz; Lz /= Lz
dt /= Lz/Delta; tmax /= Lz/Delta; t /= Lz/Delta


Xc = Lx/2;          Yc= Ly/2;          Zc = Lz/2
alphax = Lx/20;       alphay = Ly/20;       alphaz = Lz/20
rho0 = 1
Er = 6371000
Omega = 7.2921e-5
centerlatitude=45 #degrees
N = 1e-2
nu = 0      # Damping term in potential vorticity
nuBC = 0    # Damping term in boundary conditions
Umax = Delta

############################################################################
f0=2*Omega*np.sin(np.deg2rad(centerlatitude))
beta=2*Omega/Er*np.cos(np.deg2rad(centerlatitude))*0    # f plane
Ld = N*Lz/f0
############################################################################

# Grid
x = np.linspace(0,Lx,Nx,endpoint=False)
y = np.linspace(0,Ly,Ny,endpoint=False)
z = np.linspace(0,Lz,Nz)
dz = z[1]-z[0]
X,Y = np.meshgrid(x,y,indexing='ij')

############################################################################

kx,ky = wavenumbers(Nx,Ny,Lx,Ly)

# initial conditions
q = np.zeros((Nx,Ny,Nz))
DPSI_bottom=-Umax/Lz*Y
DPSI_top=-Umax/Lz*Y

for i in range(Nz):
    # q[:,:,i]+= 1e-4*np.exp(-(X-Xc)**2/2/alphax**2-(Y-Yc)**2/2/alphay**2-(z[i]-Zc)**2/2/alphaz**2) # small perturbation
    q[:,:,i]+= 1e-4*np.exp(-(z[i]-Zc)**2/2/alphaz**2)*np.sin(kx[2]*X)
    
############################################################################

# Fourier analysis
q_hat = transform(q)
DPSI_bottom_hat = transform(DPSI_bottom)
DPSI_top_hat=transform(DPSI_top)
u_hat=np.zeros_like(q_hat)
v_hat=np.zeros_like(q_hat)
w_hat=np.zeros_like(q_hat)

############################################################################

# Define the banded matrix for the second derivative

D_banded=np.ones((3,Nz))*(1/Ld**2)/dz**2
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

def nonlinear_term(q_hat,DPSI_bottom_hat,DPSI_top_hat):
    
    B_hat=np.zeros(Nz,dtype=np.complex128)
    psi_hat=np.zeros_like(B_hat)
    for i in range(Nx//2+1):  # We only need to compute the first half of the spectrum
        for j in range(Ny): 
            if i+j>0:       
                D_banded[1, 1:-1] = base + (- kx[i]**2 - ky[j]**2)
                B_hat[1:-1] = q_hat[i,j,1:-1]
                B_hat[0] = DPSI_bottom_hat[i,j]
                B_hat[-1] = DPSI_top_hat[i,j]
                psi_hat[:] = solve_banded((1,1), D_banded, B_hat)
                u_hat[i,j]=-1j*ky[j]*psi_hat
                v_hat[i,j]=1j*kx[i]*psi_hat
                w_hat[i,j] -= (1j*kx[i]*u_hat[i,j]+1j*ky[j]*v_hat[i,j])*dz
            else:
                u_hat[i,j]=0
                v_hat[i,j]=0            
                w_hat[i,j]=0

    nlt_q = inverse_transform(u_hat,True)*inverse_transform((1j*kx*q_hat.T).T,True)
    nlt_q += inverse_transform(v_hat,True)*(beta+inverse_transform(np.moveaxis(1j*ky*np.moveaxis(q_hat,1,-1),-1,1),True))
    
    # for i in range(1, Nz-1):
    #     nlt_q[:,:,i] += inverse_transform2D(w_hat[:,:,i],True)*inverse_transform2D((q_hat[:,:,i]-q_hat[:,:,i-1])/dz,True)
       
       
    nlt_bottom = inverse_transform2D(u_hat[:,:,0],True) * inverse_transform2D((1j*kx*DPSI_bottom_hat.T).T,True)
    nlt_bottom += inverse_transform2D(v_hat[:,:,0],True) * inverse_transform2D(1j*ky*DPSI_bottom_hat,True)
    
    
    nlt_top = inverse_transform2D(u_hat[:,:,-1],True) * inverse_transform2D((1j*kx*DPSI_top_hat.T).T,True)
    nlt_top += inverse_transform2D(v_hat[:,:,-1],True) * inverse_transform2D(1j*ky*DPSI_top_hat,True)
    
    
    
    return transform(nlt_q), transform(nlt_bottom), transform(nlt_top)

############################################################################

def Euler(q_hat,DPSI_bottom_hat,DPSI_top_hat):
    NLT_q_hat,NLT_bottom_hat,NLT_top_hat=nonlinear_term(q_hat,DPSI_bottom_hat,DPSI_top_hat)
    
    q_hat-=dt*NLT_q_hat
    for i in range(Nz):
        q_hat[:,:,i]/=1+dt
        q_hat[:,:,i]/=1+dt*nu*(np.repeat(kx[:,np.newaxis]**2,Ny,1)+ky**2)
    DPSI_bottom_hat-=dt*NLT_bottom_hat
    DPSI_top_hat-=dt*NLT_top_hat
    
############################################################################
    
for _ in range(int(tmax//dt)):
    Euler(q_hat, DPSI_bottom_hat, DPSI_top_hat)

    DPSI_bottom_hat *= np.exp(-nuBC * dt)
    DPSI_top_hat *= np.exp(-nuBC * dt)


plt.subplot(121)
u=inverse_transform(u_hat)
plt.contourf(u[:,Ny//2,:].T,levels=18)
plt.xlabel('x')
plt.ylabel('z')
plt.title('u')
plt.colorbar()
plt.subplot(122)
w=inverse_transform(w_hat)
v=inverse_transform(v_hat)
plt.contourf(v[:,Ny//2,:].T,levels=18)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('z')
plt.title('v')
plt.show()





































