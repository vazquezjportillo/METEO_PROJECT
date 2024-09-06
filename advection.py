import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Nx = 2**5
sigma = 0.05
dt = 0.01
Tfinal = 10
L = 1
c = 1
sigma = 0.05

x = np.linspace(0, L*(1-1/Nx), Nx)

u = np.exp(-((x - 0.5)**2) / (2 * sigma**2))
u_hat = np.fft.rfft(u)
kx = np.fft.rfftfreq(Nx, x.ptp()/(Nx-1)/(2/np.pi))

# Set up the figure and axis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Numerical solution plot
line1, = ax1.plot(x, u)
ax1.set_xlim(0, L)
ax1.set_ylim(-0.1, 1.1)
ax1.set_xlabel('x')
ax1.set_ylabel('u')
ax1.set_title('Numerical Solution')

# Real solution plot
line2, = ax2.plot(x, u)
ax2.set_xlim(0, L)
ax2.set_ylim(-0.1, 1.1)
ax2.set_xlabel('x')
ax2.set_ylabel('u')
ax2.set_title('Real Solution')

# Initialization function
def init():
    line1.set_ydata(np.exp(-((x - 0.5)**2) / (2 * sigma**2)))
    line2.set_ydata(np.exp(-((x - 0.5)**2) / (2 * sigma**2)))
    return line1, line2

# Update function
def update(frame):
    global u_hat
    u_hat -= dt * c * 1j * kx * u_hat
    u_num = np.fft.irfft(u_hat)
    line1.set_ydata(u_num)
    
    # Real solution at time t
    t = frame * dt
    u_real = np.exp(-((x - 0.5 - c * t)**2) / (2 * sigma**2))
    line2.set_ydata(u_real)
    
    return line1, line2

# Create the animation
ani = FuncAnimation(fig, update, frames=int(Tfinal/dt), init_func=init, blit=True)

# Display the animation
plt.tight_layout()
plt.show()