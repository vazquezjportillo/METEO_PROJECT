![[Armospheric_frontogenesis_numerical_simulation.pdf]]
		- All quantities except Temp and p are y independet
		- Conservation of PV with beta-plane (1.1)
		- Initial conditions:
```
		1e-6*N*(-(1 - Bu/2*(1/np.tanh(Bu/2)))*np.sinh(Z[i])*np.cosh(kx[1]*X) - n*Bu*np.cosh(Z[i])*np.sin(kx[1]*X))
		Z = Bu*z/Lz
		n = 1/Bu * np.sqrt(((Bu/2 - np.tanh(Bu/2))*((1/np.tanh(Bu/2))-Bu/2)))
```
		- Wavelenght of the disturbance: 4000km
		- D = 9km


---
![[biblio/unstability/Eadyproblem.pdf]]


$$

L>\frac{\pi}{2.3994}L_d = 1.309L_d
$$
$$\lambda_{crit} = \frac{2\pi}{2.3994}L_d = 2.62 L_d$$
![[Non_linear_equilibration_2d_eady_waves.pdf]]
	- H = 10 km
	- f = 10e-4
	- Lambda = 10e-3
	- N = 5e-3
	- L = 2000 km

---



![[Frontal_dynamics_near_front_collapse.pdf]]






![[Eady Baroclinic Instability of a Circular Vortex.pdf]]
