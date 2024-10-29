import numpy as np 
import time
import matplotlib.pyplot as plt
from functions import transform, inverse_transform, wavenumbers
from scipy.linalg import solve_banded


############################################################################
# Parameters
############################################################################

Lx = 8e6;           Ly = Lx;                 	Lz = 1e4
Nx = 2**6;          Ny = 2**4;                  Nz = 62
tmax = 3600*24*30;         	dt=3900
# nu=1e6
N = 0.01
f0=1e-4
Umax=10
latitude = 30

# Generate a grid for the quiver plot
x = np.linspace(0, Nx, Nx)
y = np.linspace(0, Ny, Ny)
X, Y = np.meshgrid(x, y)
z = np.linspace(0,Lz,Nz)


U0 = Umax*(z/Lz)**2
dU0dz = 2*Umax*z/Lz**2
d2U0dz2 = 2*Umax/Lz**2*np.ones_like(z)

# U0=Umax*z/Lz + np.cos(np.pi*z/Lz)
# dU0dz=Umax*np.ones_like(z)/Lz - np.pi*np.sin(np.pi*z/Lz)
# d2U0dz2=np.zeros_like(z) + np.pi**2*Umax*np.cos(np.pi*z/Lz)/Lz**2

# U0=2*Umax*z*np.exp(-np.pi*z/Lz)/Lz  
# dU0dz=2*Umax*np.exp(-np.pi*z/Lz)/Lz  - 2*z*Umax*np.pi/Lz*np.exp(-np.pi*z/Lz)/Lz  
# d2U0dz2=2*Umax*(np.pi/Lz)**2*np.exp(-np.pi*z/Lz)/Lz - 2*2*Umax*np.pi*np.exp(-np.pi*z/Lz)/Lz**2 + 2*Umax*z*(np.pi/Lz)**2*np.exp(-np.pi*z/Lz)/Lz**2

# U0 = Umax*np.tanh(z/Lz)
# dU0dz = Umax/Lz*(1-np.tanh(z/Lz)**2)
# d2U0dz2 = -2*Umax/Lz**2*np.tanh(z/Lz)*(1-np.tanh(z/Lz)**2)

def eady_model(Lx, Ly, Lz, Nx, Ny, Nz, tmax, dt, N, f0, Umax, latitude, perturbation = 'random', model = 'nonlinear', plots = True, custom_profile = False, U0 = False, dU0dz = False, d2U0dz2 = False):


    Er = 6371000
    Omega = 7.2921e-5

    def beta(y):
        return 2*Omega/Er*np.cos(np.deg2rad(y))

    ############################################################################

    # Grid
    x = np.linspace(0,Lx,Nx,endpoint=False)
    y = np.linspace(0,Ly,Ny,endpoint=False)
    z = np.linspace(0,Lz,Nz)
    dz = z[1]-z[0]
    X,Y = np.meshgrid(x,y,indexing='ij')
    beta = beta(latitude)
    
    if custom_profile == False:
        U0 = Umax*(z/Lz)
        dU0dz = Umax/Lz*np.ones_like(z)
        d2U0dz2 = np.zeros_like(z)

    print('CFL:', Umax*dt*np.pi/(x[1]-x[0]))

    ############################################################################

    kx,ky = wavenumbers(Nx,Ny,Lx,Ly)

    # initial conditions (perturbations only)
    q = np.zeros((Nx,Ny,Nz))
    DPSI_bottom=np.zeros((Nx,Ny))
    DPSI_top=np.zeros((Nx,Ny))

    if perturbation == 'random':
        
        q_hat = transform(q*0)
        DPSI_bottom=np.random.randn(Nx,Ny)
        DPSI_top=np.random.randn(Nx,Ny)
        
    else:
        q_hat = transform(q*0)
        # Forcing just one wave
        kxx=kx[2]
        Ld=N*Lz/f0
        mu=Ld*kxx
        c=Umax/2+Umax/mu*np.sqrt((mu/2-1/np.tanh(mu/2))*(mu/2-np.tanh(mu/2))+0j) # eady result

        for i in range(Nz):
            q[:,:,i] += 1e-2*(np.cosh(mu*z[i]/Lz)-Umax*c.real/(mu*np.abs(c)**2)*np.sinh(mu*z[i]/Lz))*np.cos(kxx*X) # this is actually the streamfunction
        DPSI_bottom=(q[:,:,1]-q[:,:,0])/dz
        DPSI_top=(q[:,:,-1]-q[:,:,-2])/dz



    ############################################################################

    # Fourier analysis
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

    ############################################################################

    def linear_term(q_hat,DPSI_bottom_hat,DPSI_top_hat):
        
        B_hat=np.zeros(Nz,dtype=np.complex128)
        psi_hat=np.zeros_like(B_hat)
        for i in range(Nx//2+1):  # We only need to compute the first half of the spectrum
            for j in range(Ny): 
                if i+j>0:     
                    D_banded[1, 1:-1] = -2*(f0/N)**2/dz**2 - (kx[i]**2 + ky[j]**2)
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
                    
                    D_banded[1, 1:-1] = -2*(f0/N)**2/dz**2 - (kx[i]**2 + ky[j]**2)
                    B_hat[1:-1] = q_hat[i,j,1:-1]
                    B_hat[0] = DPSI_bottom_hat[i,j]
                    B_hat[-1] = DPSI_top_hat[i,j]
                    psi_hat[:] = solve_banded((1,1), D_banded, B_hat)
                    u_hat[i,j]=-1j*ky[j]*psi_hat
                    v_hat[i,j]=1j*kx[i]*psi_hat
                else:
                    u_hat[i,j]=0
                    v_hat[i,j]=0

        V=inverse_transform(v_hat,True)
        nlt_q = (U0+inverse_transform(u_hat,True))*inverse_transform((1j*kx*q_hat.T).T,True) 
        nlt_q += V*(inverse_transform(np.moveaxis(1j*ky*np.moveaxis(q_hat,1,-1),-1,1),True) - (f0/N)**2*d2U0dz2 + beta)

        nlt_bottom = (U0[0]+inverse_transform(u_hat[:,:,0],True)) * inverse_transform((1j*kx*DPSI_bottom_hat.T).T,True)
        nlt_bottom += V[:,:,0] * (inverse_transform(1j*ky*DPSI_bottom_hat,True)-dU0dz[0])
            
        nlt_top = (U0[-1]+inverse_transform(u_hat[:,:,-1],True)) * inverse_transform((1j*kx*DPSI_top_hat.T).T,True)
        nlt_top += V[:,:,-1] * (inverse_transform(1j*ky*DPSI_top_hat,True)-dU0dz[-1])
                
        return transform(nlt_q,True), transform(nlt_bottom,True), transform(nlt_top,True), np.max(np.abs(V))

    ################################################################

    if model == 'linear':

        def Euler(q_hat,DPSI_bottom_hat,DPSI_top_hat):
            
            NLT_q_hat, NLT_bottom_hat, NLT_top_hat, maxV = linear_term(q_hat, DPSI_bottom_hat, DPSI_top_hat)
            q_hat -= dt*NLT_q_hat
            # q_hat /= 1+dt*nu*np.repeat((np.repeat(kx[:,np.newaxis]**2,Ny,1)+ky**2)[:,:,np.newaxis],Nz,2)

            DPSI_bottom_hat -= dt*NLT_bottom_hat
            DPSI_top_hat -= dt*NLT_top_hat
            return maxV


        ############################################################################

        def Leapfrog(q_hat,DPSI_bottom_hat,DPSI_top_hat,q_hat_old,DPSI_bottom_hat_old,DPSI_top_hat_old):
            NLT_q_hat,NLT_bottom_hat,NLT_top_hat,maxV=linear_term(q_hat,DPSI_bottom_hat,DPSI_top_hat)
            NLT_q_hat[:]= q_hat_old - 2*dt*NLT_q_hat
            
            NLT_bottom_hat[:]= DPSI_bottom_hat_old - 2*dt*NLT_bottom_hat
            NLT_top_hat[:]= DPSI_top_hat_old - 2*dt*NLT_top_hat

            np.copyto(q_hat_old,q_hat)
            np.copyto(DPSI_bottom_hat_old,DPSI_bottom_hat)
            np.copyto(DPSI_top_hat_old,DPSI_top_hat)

            np.copyto(q_hat,NLT_q_hat)
            np.copyto(DPSI_bottom_hat,NLT_bottom_hat)
            np.copyto(DPSI_top_hat,NLT_top_hat)
            
            return maxV
        
        
    else:

        def Euler(q_hat,DPSI_bottom_hat,DPSI_top_hat):
            
            NLT_q_hat, NLT_bottom_hat, NLT_top_hat, maxV = nonlinear_term(q_hat, DPSI_bottom_hat, DPSI_top_hat)
            q_hat -= dt*NLT_q_hat
            
            DPSI_bottom_hat -= dt*NLT_bottom_hat
            DPSI_top_hat -= dt*NLT_top_hat
            return maxV


        ############################################################################

        def Leapfrog(q_hat,DPSI_bottom_hat,DPSI_top_hat,q_hat_old,DPSI_bottom_hat_old,DPSI_top_hat_old):
            NLT_q_hat,NLT_bottom_hat,NLT_top_hat,maxV=nonlinear_term(q_hat,DPSI_bottom_hat,DPSI_top_hat)
            NLT_q_hat[:]= q_hat_old - 2*dt*NLT_q_hat
            
            NLT_bottom_hat[:]= DPSI_bottom_hat_old - 2*dt*NLT_bottom_hat
            NLT_top_hat[:]= DPSI_top_hat_old - 2*dt*NLT_top_hat

            np.copyto(q_hat_old,q_hat)
            np.copyto(DPSI_bottom_hat_old,DPSI_bottom_hat)
            np.copyto(DPSI_top_hat_old,DPSI_top_hat)

            np.copyto(q_hat,NLT_q_hat)
            np.copyto(DPSI_bottom_hat,NLT_bottom_hat)
            np.copyto(DPSI_top_hat,NLT_top_hat)
            
            return maxV
        
    ############################################################################

    q_hat_old=np.copy(q_hat)
    DPSI_bottom_hat_old=np.copy(DPSI_bottom_hat)
    DPSI_top_hat_old=np.copy(DPSI_top_hat)

    maxV=Euler(q_hat, DPSI_bottom_hat, DPSI_top_hat) 
    
    if plots:
        
        plt.figure()
        plt.subplot(121)
        for s in range(1,int(tmax//dt)):

            maxV=Leapfrog(q_hat, DPSI_bottom_hat, DPSI_top_hat, q_hat_old, DPSI_bottom_hat_old, DPSI_top_hat_old)
            plt.semilogy(s*dt/3600/24,maxV,'ob')
            plt.semilogy(s*dt/3600/24,np.exp(0.26*s*dt/3600/24)/1e2,'or')
            plt.xlabel('time (days)')
            plt.ylabel('max(V) (m/s)')
            plt.title('max(V) vs time')
            plt.legend(['model','theoretical (linear)'])
            
    else:
        
        for s in range(1,int(tmax//dt)):
            maxV=Leapfrog(q_hat, DPSI_bottom_hat, DPSI_top_hat, q_hat_old, DPSI_bottom_hat_old, DPSI_top_hat_old)
        
        
        
    if plots:    
        plt.subplot(122)
        plt.plot(U0, z, label='U0(z)')
        plt.xlabel('U0')
        plt.ylabel('z')
        plt.title('Vertical Profile of U0')
        plt.show()
        
    return inverse_transform(q_hat), inverse_transform(v_hat), inverse_transform(u_hat)

q, v, u = eady_model(Lx, Ly, Lz, Nx, Ny, Nz, tmax, dt, N, f0, Umax, latitude, perturbation = 'random', model = 'nonlinear', plots = True, custom_profile = True, U0 = U0, dU0dz = dU0dz, d2U0dz2 = d2U0dz2)

plt.figure()
plt.subplot(131)
plt.contourf(q[:,Ny//2,:].T,levels=18)
plt.xlabel('x')
plt.ylabel('z')
plt.title('q')
plt.colorbar()
plt.subplot(132)
plt.contourf(v[:,Ny//2,:].T,levels=18)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('z')
plt.title('v')
# Plot u and v with arrows
plt.subplot(133)
plt.quiver(X, Y, u[:, :, -Nz].T, v[:, :, -Nz].T)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Velocity Field (u, v)')
plt.grid(True)
plt.show()

