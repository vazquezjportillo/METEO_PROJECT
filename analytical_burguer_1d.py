import numpy as np
from scipy.integrate import quad

# Define constants
nu = 1.0  # You can adjust this value
x = 0.5   # Example value of x
t = 0.1   # Example value of t

# Define the integrand for the first integral
def integrand_1(epsilon, ni, nu):
    return np.exp(-1 / (2 * np.pi * nu) * (np.cos(-np.pi * epsilon) + 1)) * np.cos(ni * np.pi * epsilon)

# Define the integrand for the second integral
def integrand_2(epsilon, nu):
    return np.exp(-1 / (2 * np.pi * nu) * (np.cos(-np.pi * epsilon) + 1))

# Sum over i
def summation_part(nu, x, t, max_i=100):
    sum_num = 0
    sum_den = 0

    for i in range(1, max_i + 1):
        ni = i

        # Compute the first integral for numerator and denominator
        integral_num, _ = quad(integrand_1, -1, 1, args=(ni, nu))
        integral_den, _ = quad(integrand_1, -1, 1, args=(ni, nu))

        # Numerator
        sum_num += integral_num * ni * np.pi * np.sin(ni * np.pi * x) * np.exp(-nu * (ni * np.pi)**2 * t)

        # Denominator
        sum_den += integral_den * np.cos(ni * np.pi * x) * np.exp(-nu * (ni * np.pi)**2 * t)

    # Compute the additional term in the denominator
    integral_additional, _ = quad(integrand_2, -1, 1, args=(nu,))
    sum_den += 0.5 * integral_additional

    return sum_num, sum_den

# Calculate the full expression for u(x, t)
def u_x_t(nu, x, t, max_i=100):
    sum_num, sum_den = summation_part(nu, x, t, max_i)
    return 2 * nu * (sum_num / sum_den)

# Example usage
u_value = u_x_t(nu, x, t)
print(f"u(x={x}, t={t}) = {u_value}")