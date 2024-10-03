import numpy as np 
import matplotlib.pyplot as plt
from functions import transform, inverse_transform, wavenumbers, X_derivative, Y_derivative
from scipy.linalg import solve_banded

############################################################################

# Parameters
Lx = 2500;          Ly = 2500;          Lz = 50
Nx = 2**5;          Ny = 2**5;          Nz = 10
Xc = Lx/2;          Yc = Ly/2;          Zc = Lz/2
alphax = 100;       alphay = 100;       alphaz = 5
rho0 = 1
Er = 6371000
Omega = 7.2921e-5
centerlatitude=45 #degrees
nu = 0.01      # Damping term in potential vorticity
Umax = 3
N2 = 1e-5
t = 0;              tmax = 100;          dt=1

############################################################################

f0=2*Omega*np.sin(np.deg2rad(centerlatitude))
beta=2*Omega/Er*np.cos(np.deg2rad(centerlatitude))

############################################################################

# Grid
x = np.linspace(0,Lx,Nx,endpoint=False)
y = np.linspace(0,Ly,Ny,endpoint=False)
z = np.linspace(0,Lz,Nz)
dz = z[1]-z[0]
X,Y = np.meshgrid(x,y,indexing='ij')

# initial conditions
q = np.zeros((Nx,Ny,Nz))
DPSI_bottom=-Umax/Lz*Y
DPSI_top=-Umax/Lz*Y
for i in range(Nz):
    q[:,:,i]+=1e-4*np.exp(-(X-Xc)**2/2/alphax**2-(Y-Yc)**2/2/alphay**2-(z[i]-Zc)**2/2/alphaz**2) # small perturbation

# Fourier analysis
kx,ky = wavenumbers(Nx,Ny,Lx,Ly)
q_hat = transform(q)
DPSI_bottom_hat=transform(DPSI_bottom)
DPSI_top_hat=transform(DPSI_top)
u_hat=np.zeros_like(q_hat)
v_hat=np.zeros_like(q_hat)

D_banded=np.ones((3,Nz))*(f0**2/N2)/dz**2
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
            else:
                u_hat[i,j]=0
                v_hat[i,j]=0                

    nlt_q = inverse_transform(u_hat)*inverse_transform((1j*kx*q_hat.T).T)
    nlt_q += inverse_transform(v_hat)*(beta+inverse_transform(np.moveaxis(1j*ky*np.moveaxis(q_hat,1,-1),-1,1)))
       
    nlt_bottom = inverse_transform(u_hat[:,:,0]) * inverse_transform((1j*kx*DPSI_bottom_hat.T).T)
    nlt_bottom += inverse_transform(v_hat[:,:,0]) * inverse_transform(1j*ky*DPSI_bottom_hat)
    
    nlt_top = inverse_transform(u_hat[:,:,-1]) * inverse_transform((1j*kx*DPSI_top_hat.T).T)
    nlt_top += inverse_transform(v_hat[:,:,-1]) * inverse_transform(1j*ky*DPSI_top_hat)
    
    return transform(nlt_q), transform(nlt_bottom), transform(nlt_top)

def Euler(q_hat,DPSI_bottom_hat,DPSI_top_hat):
    NLT_q_hat,NLT_bottom_hat,NLT_top_hat=nonlinear_term(q_hat,DPSI_bottom_hat,DPSI_top_hat)
    
    q_hat-=dt*NLT_q_hat
    for i in range(Nz):
        q_hat[:,:,i]/=1+dt
        q_hat[:,:,i]/=1+dt*nu*(np.repeat(kx[:,np.newaxis]**2,Ny,1)+ky**2)
    DPSI_bottom_hat-=dt*NLT_bottom_hat
    DPSI_top_hat-=dt*NLT_top_hat
    
for _ in range(int(tmax//dt)):
    Euler(q_hat,DPSI_bottom_hat,DPSI_top_hat)
    
    
plt.subplot(131)
u=inverse_transform(u_hat)
plt.contourf(u[:,:,Nz//2])
plt.colorbar()
plt.subplot(132)
v=inverse_transform(v_hat)
plt.contourf(v[:,:,Nz//2])
plt.colorbar()
plt.subplot(133)
plt.quiver(X,Y,u[:,:,Nz//2],v[:,:,Nz//2])
plt.show()


exit()































# #RK4 scheme spin up
def rhs(q_hat, KX, KY, nu, dz, Nz):
    """Compute the right-hand side of the differential equation."""
    dq_hat = np.zeros_like(q_hat)
    for i in range(1, Nz - 1):
        dq_hat[:,:,i] = (
            -nonlinear_term(q_hat)[0][:,:,i] + 
            nu*(-KX**2 - KY**2)*q_hat[:,:,i] + 
            nu*(q_hat[:,:,i+1] - 2*q_hat[:,:,i] + q_hat[:,:,i-1])/dz**2
        )  
    return dq_hat

for i in range(1, Nz - 1):  # Skip the first and last point
    k1 = dt * rhs(q00_hat, KX, KY, nu, dz, Nz)
    k2 = dt * rhs(q00_hat + 0.5 * k1, KX, KY, nu, dz, Nz)
    k3 = dt * rhs(q00_hat + 0.5 * k2, KX, KY, nu, dz, Nz)
    k4 = dt * rhs(q00_hat + k3, KX, KY, nu, dz, Nz)
    
    q0_hat[:,:,i] = q00_hat[:,:,i] + (k1[:,:,i] + 2*k2[:,:,i] + 2*k3[:,:,i] + k4[:,:,i]) / 6

    
dpsi_hat_dz_h = (nonlinear_term(q_hat)[3][:,:,-1] - nonlinear_term(q_hat)[3][:,:,-2])/dz
dpsi_hat_dz_0 = (nonlinear_term(q_hat)[3][:,:,1] - nonlinear_term(q_hat)[3][:,:,0])/dz


q0_hat[:,:,0] = 0
q0_hat[:,:,-1] = 0


###############################################################################################3

# I GUESS THAT I CAN USE RHS FOR COMPUTING IT IN A CLEARER WAY (?)

for n in range(1, int(tmax//dt - 1)):
    for i in range(1, Nz - 1):  # Skip the first and last point
        # Update using the Adams-Bashforth formula
        q_hat[:,:,i] = q0_hat[:,:,i] + (dt/2)*(
            3*(
                nonlinear_term(q0_hat)[0][:,:,i] + 
                nu*(-KX**2 -KY**2)*q0_hat[:,:,i] + 
                nu*(q0_hat[:,:,i+1] - 2*q0_hat[0:,:,i] + q0_hat[:,:, i-1]) /dz**2
            ) - (
                nonlinear_term(q00_hat)[0][:,:,i] + 
                nu*(-KX**2 -KY**2)*q00_hat[:,:,i] + 
                nu*(q00_hat[:,:,i+1] - 2*q00_hat[0:,:,i] + q00_hat[:,:, i-1]) /dz**2
            )
        )
    
    q_hat[:,:,0] = 0
    q_hat[:,:,-1] = 0
    q00_hat = q0_hat.copy()
    q0_hat = q_hat.copy()
    
    
#############################################################



    x_index = np.argmin(np.abs(x - Lx/2))
    y_index = np.argmin(np.abs(y - Ly/2))

    v = inverse_transform(nonlinear_term(q00_hat)[1],True)
    u = inverse_transform(nonlinear_term(q00_hat)[2],True)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the YZ slice at X = Lx/2
    YZ_slice = u[x_index, :, :]
    contour1 = ax1.contourf(Y[x_index, :, :], Z[x_index, :, :], YZ_slice, cmap='viridis')
    ax1.set_title(f'YZ Slice at X = $L_x/2$')
    ax1.set_xlabel('Y-axis')
    ax1.set_ylabel('Z-axis')

    # Plot the XZ slice at Y = Ly/2
    XZ_slice = u[:, y_index, :]
    contour2 = ax2.contourf(X[:, y_index, :], Z[:, y_index, :], XZ_slice, cmap='viridis')
    ax2.set_title(f'XZ Slice at Y = $L_y/2$')
    ax2.set_xlabel('X-axis')
    ax2.set_ylabel('Z-axis')

    # Adjust layout to make space for the color bar
    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, wspace=0.4)

    # Add a single color bar for both subplots
    cbar = fig.colorbar(contour1, ax=[ax1, ax2], orientation='horizontal', pad=0.2,shrink = 0.6, label='$u_0$ values')

    plt.show()






























