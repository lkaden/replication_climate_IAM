import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import interpolate

class RCKmod:
    def __init__(self, rho, alpha, theta, xi, delta, phi, gamma, beta, eta):
        # Initialize parameters
        self.rho = rho
        self.alpha = alpha
        self.theta = theta
        self.xi = xi
        self.delta = delta
        self.phi = phi
        self.gamma = gamma
        self.beta = beta
        self.eta = eta
        self.kmax = (1 / (self.phi + self.xi + self.delta)) ** (1 / (1 - self.alpha))
        self.kss = np.exp(- self.gamma * self.T**2 / 2)* (self.alpha / (self.rho + self.delta + self.theta * self.gamma))**1/(1-self.alpha)
        self.css = self.kss**(alpha) * (1-beta) * (1-eta) * gamma - (xi + delta + phi)*self.kss
        self.solve()
    
    def output(self, k):
        emissions = self.gamma * k * (1 - self.eta)
        return (k ** self.alpha) * (1 - self.beta * emissions)
    
    def dkdt(self, c, k):
        return self.output(k) - c - (self.phi + self.xi + self.delta) * k
    
    def dcdk(self, c, k):
        dcdt = c / self.rho * (self.alpha * k ** (self.alpha - 1) - self.theta - (self.xi + self.delta) - self.rho * self.phi)
        dkdt = self.dkdt(c, k)
        return dcdt / dkdt
    
    def solve(self, eps=1e-8, npoints=400):
        k_below = np.linspace(self.kss, 1e-10, npoints)
        k_above = np.linspace(self.kss, self.kmax, npoints)
        k = np.concatenate((k_below, k_above)).flatten()
        c_below = odeint(self.dcdk, self.css - eps, k_below)
        c_above = odeint(self.dcdk, self.css + eps, k_above)
        c = np.concatenate((c_below, c_above)).flatten()
        self.cFunc = interpolate.interp1d(k, c, bounds_error=False, fill_value="extrapolate")
    
    def k_dynamics(self, k0, t):
        return odeint(lambda k, t: self.dkdt(self.cFunc(k), k), k0, t)

# Parameters
eta_values = np.linspace(0.1, 0.9, 9)
t = np.linspace(0, 100, 100)
k0 = 20
total_consumption = []

# Model computation and consumption aggregation
for eta in eta_values:
    model = RCKmod(rho=2, alpha=0.3, theta=0.02, xi=0.01, delta=0.08, phi=0.03, gamma=0.1, beta=0.01, eta=eta)
    k = model.k_dynamics(k0, t)
    c = model.cFunc(k).flatten()  # Ensure 'c' is a 1D array
    if c.ndim > 1:
        c = c[:, 0]  # Take the first column if 'c' is 2D
    total_consumption.append(np.trapz(c, t))  # Integrate consumption over time

# Find the optimal eta
optimal_idx = np.argmax(total_consumption)
optimal_eta = eta_values[optimal_idx]
optimal_total_consumption = total_consumption[optimal_idx]

# Output the results
print(f"Optimal eta: {optimal_eta:.3f}")
print(f"Total Consumption for Optimal eta: {optimal_total_consumption:.3f}")
