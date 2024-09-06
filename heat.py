import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


Nx = 2**5
sigma = 0.05
kappa = 0.1
dt = 0.001
Tfinal = 10
L = 1


x = np.linspace(0, L*(1-1/Nx), Nx)

T = np.exp(-((x-0.5)**2/(2*sigma**2)))
T_hat = np.fft.rfft(T)
kx = np.fft.rfftfreq(Nx, x.ptp()/(Nx-1)/(2/np.pi))

dt_limit = 2/(kappa*(max(kx)**2))
print(dt_limit)

t = 0
stamps = np.linspace(0, Tfinal, 10)
while t < Tfinal:
    T_hat -= dt*kappa*(T_hat*kx**2)
    if any(np.isclose(t, stamps, atol=dt/2)):
        plt.plot(x, np.fft.irfft(T_hat))  
    t += dt
plt.show()