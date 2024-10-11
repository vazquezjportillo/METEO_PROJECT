import numpy as np 
import matplotlib.pyplot as plt
from functions import transform, inverse_transform, wavenumbers, inverse_transform2D
from scipy.linalg import solve_banded
############################################################################
# Parameters
############################################################################

Lx = 8e6;           Ly = 8e6;                 Lz = 1e4
Nx = 2**6;          Ny = 2**2;              Nz = 20
t = 0;              tmax = 3600*24*10;         dt=1000

N = 1e-2
nu = 0     # Damping term in potential vorticity
f0=1e-4
Umax=10

nuBC = 0    # Damping term in boundary conditions
Ld=N*Lz/f0
#####################
#Normalization
Xc = Lx/2;          Yc= Ly/2;               Zc = Lz/2
alphax = Lx/20;     alphay = Ly/20;       alphaz = Lz/20

############################################################################

# Grid
x = np.linspace(0,Lx,Nx,endpoint=False)
y = np.linspace(0,Ly,Ny,endpoint=False)
z = np.linspace(0,Lz,Nz)
dz = z[1]-z[0]
X,Y = np.meshgrid(x,y,indexing='ij')

U0=Umax/Lz*z
V0=np.zeros_like(U0)

############################################################################

kx,ky = wavenumbers(Nx,Ny,Lx,Ly)
# initial conditions
q = np.zeros((Nx,Ny,Nz))
DPSI_bottom=np.zeros((Nx,Ny))
DPSI_top=np.zeros((Nx,Ny))

for i in range(Nz):
    # q[:,:,i]+= 1e-4*np.exp(-(X-Xc)**2/2/alphax**2-(Y-Yc)**2/2/alphay**2-(z[i]-Zc)**2/2/alphaz**2) # small perturbation
    q[:,:,i]+= 1e-8*np.exp(-(z[i]-Zc)**2/2/alphaz**2)*np.sin(kx[2]*X)
############################################################################

# Fourier analysis
q_hat = transform(q)
DPSI_bottom_hat = transform(DPSI_bottom)
DPSI_top_hat=transform(DPSI_top)
u_hat=np.zeros_like(q_hat)
v_hat=np.zeros_like(q_hat)

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
                u_hat[i,j]=-1j*ky[j]*psi_hat
                v_hat[i,j]=1j*kx[i]*psi_hat
            else:
                u_hat[i,j]=U0*Nx*Ny
                v_hat[i,j]=V0*Nx*Ny

    nlt_q = U0*inverse_transform((1j*kx*q_hat.T).T)
    
    nlt_bottom = inverse_transform2D(v_hat[:,:,0]) * (-Umax/Lz)
        
    nlt_top = U0[-1] * inverse_transform2D((1j*kx*DPSI_top_hat.T).T)
    nlt_top += inverse_transform2D(v_hat[:,:,-1]) * (-Umax/Lz)
            
    return transform(nlt_q), transform(nlt_bottom), transform(nlt_top)

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
                v_hat[i,j]=V0*Nx*Ny

    nlt_q = inverse_transform(u_hat)*inverse_transform((1j*kx*q_hat.T).T)
    nlt_q += inverse_transform(v_hat)*inverse_transform(np.moveaxis(1j*ky*np.moveaxis(q_hat,1,-1),-1,1))
    
    nlt_bottom = inverse_transform2D(u_hat[:,:,0]) * inverse_transform2D((1j*kx*DPSI_bottom_hat.T).T)
    nlt_bottom += inverse_transform2D(v_hat[:,:,0]) * (inverse_transform2D(1j*ky*DPSI_bottom_hat)-Umax/Lz)
        
    nlt_top = inverse_transform2D(u_hat[:,:,-1]) * inverse_transform2D((1j*kx*DPSI_top_hat.T).T)
    nlt_top += inverse_transform2D(v_hat[:,:,-1]) * (inverse_transform2D(1j*ky*DPSI_top_hat)-U0[-1]/Lz)
            
    return transform(nlt_q), transform(nlt_bottom), transform(nlt_top)

def Euler_explicit(q_hat,DPSI_bottom_hat,DPSI_top_hat):
    NLT_q_hat,NLT_bottom_hat,NLT_top_hat=nonlinear_term(q_hat,DPSI_bottom_hat,DPSI_top_hat)
    
    q_hat-=dt*NLT_q_hat
    for i in range(Nz):
        q_hat[:,:,i]/=1+dt*nu*(np.repeat(kx[:,np.newaxis]**2,Ny,1)+ky**2)
        
        # U0[i] = np.mean(inverse_transform(u_hat)[:, :, i])/Lz*z[i]
    # plt.figure()
    # plt.plot(U0, z, label='U0')
    # plt.xlabel('U0')
    # plt.ylabel('z')
    # plt.title('U0 vs z')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    
    DPSI_bottom_hat-=dt*NLT_bottom_hat
    DPSI_top_hat-=dt*NLT_top_hat
    
############################################################################
def velocity_term(q_hat,DPSI_bottom_hat,DPSI_top_hat):
    
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
                v_hat[i,j]=V0*Nx*Ny
            
    return u_hat, v_hat

def Euler_implicit(q_hat,DPSI_bottom_hat,DPSI_top_hat):
    u_hat, v_hat = velocity_term(q_hat,DPSI_bottom_hat,DPSI_top_hat)
    q_hat = q_hat**-1
    q_hat+=dt*transform(inverse_transform(u_hat)*inverse_transform((1j*kx*(q_hat.T)**-1).T))
    q_hat+=dt*transform(inverse_transform(v_hat)*inverse_transform(np.moveaxis(1j*ky*np.moveaxis((q_hat)**-1,1,-1),-1,1)))
    q_hat = q_hat**-1
    
    DPSI_bottom_hat = DPSI_bottom_hat**-1
    DPSI_bottom_hat+=dt*transform(inverse_transform2D(u_hat[:,:,0])*inverse_transform((1j*kx*(DPSI_bottom_hat.T)**-1).T))
    DPSI_bottom_hat+=dt*transform(inverse_transform2D(v_hat[:,:,0])*(inverse_transform2D(1j*ky*DPSI_bottom_hat**-1)-Umax/Lz))
    DPSI_bottom_hat = DPSI_bottom_hat**-1
    
    DPSI_top_hat = DPSI_top_hat**-1
    DPSI_top_hat+=dt*transform(inverse_transform2D(u_hat[:,:,-1])*inverse_transform((1j*kx*(DPSI_top_hat.T)**-1).T))
    DPSI_top_hat+=dt*transform(inverse_transform2D(v_hat[:,:,-1])*(inverse_transform2D(1j*ky*DPSI_top_hat**-1)-Umax/Lz))
    DPSI_top_hat = DPSI_top_hat**-1
############################################################################
    
for s in range(int(tmax//dt)):
    # print(s)
    Euler_explicit(q_hat, DPSI_bottom_hat, DPSI_top_hat)
    
    # u=inverse_transform(u_hat)
    # v=inverse_transform(v_hat)
    # max_velocity = np.max(np.sqrt(u**2 + v**2))

    # # Calculate dx and dy
    # dx = Lx / Nx
    # dy = Ly / Ny

    # # Calculate the CFL condition
    # cfl_x = max_velocity * dt / dx
    # cfl_y = max_velocity * dt / dy

    # print(f"Max velocity: {max_velocity}")

    # print(f"CFL condition in x direction: {cfl_x}")
    # print(f"CFL condition in y direction: {cfl_y}")


plt.subplot(121)
u=inverse_transform(u_hat)
plt.contourf((u[:,Ny//2,:]-U0).T,levels=18)
plt.xlabel('x')
plt.ylabel('z')
plt.title('u')
plt.colorbar()
plt.subplot(122)
v=inverse_transform(v_hat)
plt.contourf(v[:,Ny//2,:].T,levels=18)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('z')
plt.title('v')
plt.show()

u=inverse_transform(u_hat)
v=inverse_transform(v_hat)
max_velocity = np.max(np.sqrt(u**2 + v**2))

# Calculate dx and dy
dx = Lx / Nx
dy = Ly / Ny

# Calculate the CFL condition
cfl_x = max_velocity * dt / dx
cfl_y = max_velocity * dt / dy

print(f"Max velocity: {max_velocity}")

print(f"CFL condition in x direction: {cfl_x}")
print(f"CFL condition in y direction: {cfl_y}")
