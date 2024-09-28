import numpy as np 
import matplotlib.pyplot as plt

# Parameters
Lx = 2;             Ly = 2;             Lz = 100
Nx = 2**5;          Ny = 2**5;          Nz = 100
Xc = Lx/2;          Yc = Ly/2;          Zc = Lz/2
alpha = 0.1;        kappa = 0.1;        gamma = 10
Umax = 3

# Grid
x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
z = np.linspace(0, Lz, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Compute psi0 values
psi0 = -Z*Y*Umax/Lz + np.exp(-(X-Xc)**2/alpha-(Y-Yc)**2/kappa)

# Find the indices for Lx/2 and Ly/2
x_index = np.argmin(np.abs(x - Lx/2))
y_index = np.argmin(np.abs(y - Ly/2))

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot the YZ slice at X = Lx/2
YZ_slice = psi0[x_index, :, :]
contour1 = ax1.contourf(Y[x_index, :, :], Z[x_index, :, :], YZ_slice, cmap='viridis')
ax1.set_title(f'YZ Slice at X = $L_x/2$')
ax1.set_xlabel('Y-axis')
ax1.set_ylabel('Z-axis')

# Plot the XZ slice at Y = Ly/2
XZ_slice = psi0[:, y_index, :]
contour2 = ax2.contourf(X[:, y_index, :], Z[:, y_index, :], XZ_slice, cmap='viridis')
ax2.set_title(f'XZ Slice at Y = $L_y/2$')
ax2.set_xlabel('X-axis')
ax2.set_ylabel('Z-axis')

# Adjust layout to make space for the color bar
plt.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9, wspace=0.4)

# Add a single color bar for both subplots
cbar = fig.colorbar(contour1, ax=[ax1, ax2], orientation='horizontal', pad=0.2, shrink=0.8, label='$\psi_0$ values')

plt.show()


psi0 = -Z*Y*Umax/Lz + np.exp(-(X-Xc)**2/alpha-(Y-Yc)**2/kappa-(Z-Zc)**2/gamma)

v = -2*(X-Xc)/alpha*np.exp(-(X-Xc)**2/alpha-(Y-Yc)**2/kappa-(Z-Zc)**2/gamma)
u = -Z*Umax/Lz - 2*(Y-Yc)/kappa*np.exp(-(X-Xc)**2/alpha-(Y-Yc)**2/kappa-(Z-Zc)**2/gamma)

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