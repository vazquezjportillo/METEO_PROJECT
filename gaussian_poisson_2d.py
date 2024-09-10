import numpy as np 
import matplotlib.pyplot as plt
from poisson_solver import poisson_fourier
import time

error = []
elapsed_time = []

# Parameters
Lx = 1
Ly = 1
alpha = .005
beta = .005
Xc=0.4
Yc=0.6

for i in range(4, 14):
    Nx = 2**i
    print(Nx)
    Ny = Nx
    x = np.linspace(0,Lx,Nx,endpoint=False)
    y = np.linspace(0,Ly,Ny,endpoint=False)

    X,Y = np.meshgrid(x,y,indexing='ij')
    T = np.zeros((Nx,Ny))
    T_analytical = np.exp(-(X-Xc)**2 / alpha - (Y-Yc)**2 / beta)
    F = (-(2/alpha + 2/beta) + 4/alpha**2*(X-Xc)**2 + 4/beta**2*(Y-Yc)**2)*T_analytical
    T_analytical-=np.mean(T_analytical)

    t1 = time.time()
    T = poisson_fourier(Lx, Ly, Nx, Ny, F)
    t2 = time.time()
    elapsed_time.append(t2-t1)
    error.append( np.linalg.norm(T_analytical - T) )
    
plt.figure()
plt.loglog(np.arange(4,14),error)

plt.figure()
plt.loglog(np.arange(4,14),elapsed_time)
plt.show()
#     fig, axs = plt.subplots(2, 1, figsize=(10, 8))

#     # Plot the analytical solution
#     contour1 = axs[0].contourf(X, Y, T_analytical)
#     fig.colorbar(contour1, ax=axs[0])
#     axs[0].set_title('Analytical Solution')

#     # Plot the computed solution
#     contour2 = axs[1].contourf(X, Y, T)
#     fig.colorbar(contour2, ax=axs[1])
#     axs[1].set_title('Spectral Methods Solution')

#     plt.tight_layout()
#     plt.show()
    
# print(error)
# Nx_values = [2**i for i in range(4, 7)]
# print(Nx_values)

# plt.plot(Nx_values, error, marker='o', linestyle='-', color='b', label='Error Norm')
# plt.xlabel('Nx')
# plt.ylabel('Error Norm')
# plt.title('Error Norm vs Nx')
# plt.grid(True)
# plt.legend()
# plt.show()