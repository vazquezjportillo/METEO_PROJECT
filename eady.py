import numpy as np 
import matplotlib.pyplot as plt
from functions import transform, inverse_transform, wavenumbers
from scipy.linalg import solve_banded
from line_profiler import LineProfiler


############################################################################
# Parameters
############################################################################

Lx = 1e6;           Ly = Lx;                 Lz = 1e4
Nx = 2**6;          Ny = 2**1;                  Nz = 32
t = 0;              tmax = 3600*24*6;         dt=100

N = 5e-3
nu = 10000      # Damping term in potential vorticity
f0=1e-4
Umax=10

Ld=N*Lz/f0
Bu = Ld/Lx
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
d2U0dz2=np.zeros_like(z)

############################################################################

kx,ky = wavenumbers(Nx,Ny,Lx,Ly)

# initial conditions (perturbations only)
q = np.zeros((Nx,Ny,Nz))
DPSI_bottom=np.zeros((Nx,Ny))
DPSI_top=np.zeros((Nx,Ny))

Z = Bu*(z/Lz - 0.5)
n = 1/Bu * np.sqrt(((Bu/2 - np.tanh(Bu/2))*((1/np.tanh(Bu/2))-Bu/2)))

for i in range(Nz):
    q[:,:,i]+= 1e-4*np.exp(-(z[i]-Zc)**2/2/alphaz**2)*np.sin(kx[1]*X)
    # q[:,:,i]+= 1e-10*N*(-(1 - Bu/2*(1/np.tanh(Bu/2)))*np.sinh(Z[i])*np.cosh(kx[1]*X) - n*Bu*np.cosh(Z[i])*np.sin(kx[1]*X))
############################################################################

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
                B_hat[1:-1] = q_hat[i,j,1:-1]
                B_hat[0] = DPSI_bottom_hat[i,j]
                B_hat[-1] = DPSI_top_hat[i,j]
                psi_hat[:] = solve_banded((1,1), D_banded, B_hat)
                v_hat[i,j]=1j*kx[i]*psi_hat
            else:
                v_hat[i,j]=0

    #All terms are linear now 

    nlt_q = U0*(1j*kx*q_hat.T).T - v_hat*(f0/N)**2*d2U0dz2 
    
    nlt_bottom = U0[0] * (1j*kx*DPSI_bottom_hat.T).T
    nlt_bottom += v_hat[:,:,0] * (-dU0dz[0])
        
    nlt_top = U0[-1] * (1j*kx*DPSI_top_hat.T).T
    nlt_top += v_hat[:,:,-1] * (-dU0dz[-1])
    V=inverse_transform(v_hat)

    return nlt_q, nlt_bottom, nlt_top, np.max(np.abs(V))

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
            else:
                u_hat[i,j]=U0*Nx*Ny
                v_hat[i,j]=0

    V=inverse_transform(v_hat,True)
    nlt_q = inverse_transform(u_hat,True)*inverse_transform((1j*kx*q_hat.T).T,True) 
    nlt_q += V*(inverse_transform(np.moveaxis(1j*ky*np.moveaxis(q_hat,1,-1),-1,1),True) - (f0/N)**2*d2U0dz2)

    nlt_bottom = inverse_transform(u_hat[:,:,0],True) * inverse_transform((1j*kx*DPSI_bottom_hat.T).T,True)
    nlt_bottom += inverse_transform(v_hat[:,:,0],True) * (inverse_transform(1j*ky*DPSI_bottom_hat,True)-dU0dz[0])
        
    nlt_top = inverse_transform(u_hat[:,:,-1],True) * inverse_transform((1j*kx*DPSI_top_hat.T).T,True)
    nlt_top += inverse_transform(v_hat[:,:,-1],True) * (inverse_transform(1j*ky*DPSI_top_hat,True)-dU0dz[-1])
            
    return transform(nlt_q,True), transform(nlt_bottom,True), transform(nlt_top,True),np.max(np.abs(V))

############################################################################

def Euler(q_hat,DPSI_bottom_hat,DPSI_top_hat):
    NLT_q_hat,NLT_bottom_hat,NLT_top_hat,maxV=nonlinear_term(q_hat,DPSI_bottom_hat,DPSI_top_hat)
    
    q_hat-=dt*NLT_q_hat
    for i in range(Nz):
        q_hat[:,:,i]/=1+dt*nu*(np.repeat(kx[:,np.newaxis]**2,Ny,1)+ky**2)
    DPSI_bottom_hat-=dt*NLT_bottom_hat
    DPSI_top_hat-=dt*NLT_top_hat
    return maxV

lp = LineProfiler()
lp_wrapper = lp(Euler)
lp_wrapper(q_hat,DPSI_bottom_hat,DPSI_top_hat)
lp.print_stats()
    
############################################################################
    

# for s in range(int(tmax//dt)):
#     maxV=Euler(q_hat, DPSI_bottom_hat, DPSI_top_hat)
#     plt.plot(s,maxV,'ob')

# plt.figure()
# plt.subplot(121)
# u=inverse_transform(u_hat)
# plt.contourf(u[:,Ny//2,:].T,levels=18)
# plt.xlabel('x')
# plt.ylabel('z')
# plt.title('u')
# plt.colorbar()
# plt.subplot(122)
# v=inverse_transform(v_hat)
# plt.contourf(v[:,Ny//2,:].T,levels=18)
# plt.colorbar()
# plt.xlabel('x')
# plt.ylabel('z')
# plt.title('v')
# plt.show()