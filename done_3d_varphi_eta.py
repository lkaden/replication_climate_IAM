import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

# Define the custom color map
colors = ["blue", "purple", "red"]
cmap = LinearSegmentedColormap.from_list("custom_blue_purple_red", colors, N=100)

class RCKmod:
    def __init__(self, rho, alpha, delta, gamma, eta, varphi, phi, chi, zeta, T_initial, theta, g, L0, noise_std=0.013):
        self.rho = rho  # time preference
        self.alpha = alpha
        self.delta = delta  # depreciation rate
        self.gamma = gamma  # damage multiplier
        self.eta = eta  # abatement
        self.chi = chi  # temperature climate response to emissions
        self.zeta = zeta  # MAC slope
        self.varphi = varphi  # allocation of investment
        self.phi = phi  # emission intensity
        self.theta = theta  # savings rate
        self.T_initial = T_initial  # Initial temperature
        self.noise_std = noise_std
        self.g = g  # labor growth rate
        self.L0 = L0  # initial labor

    def labor(self, t):
        return self.L0 * np.log(self.g * t + 1)

    def output(self, k, k_d, cumulative_emissions, t):
        L_t = self.labor(t)
        temp = self.T_initial + self.chi * cumulative_emissions 
        Omega = np.exp(-self.gamma * temp ** 2 / 2)  # damage from increased temperature
        Lambda = np.exp(-self.zeta * self.eta ** 2 / 2)  # abatement cost function
        effective_output = ((k + k_d) ** self.alpha) * (L_t ** (1 - self.alpha)) * Omega * Lambda
        return effective_output, temp, Omega

    def Kappa(self, varphi, k, k_d):
        if varphi < 0:
            r = - varphi
            theta1 = 700
            return -theta1 * r ** 2.1 / (k + k_d)
        return 0

    def subsistence(self, effective_output, dk, dk_d):
        diff = effective_output - dk - dk_d
        return np.exp(-1 / 0.000000001 * (diff))

    def dkdt(self, k_kd_ce, t):
        k, k_d, cumulative_emissions, stranded_assets = k_kd_ce
        effective_output, temp, omega = self.output(k, k_d, cumulative_emissions, t)
        kappa = self.Kappa(self.varphi, k, k_d)

        # Capital dynamics for clean capital
        dk = self.theta * (1 - abs(self.varphi)) * effective_output - (0.1 * (1 - abs(self.varphi)) * effective_output) ** 2 / k - (self.delta + self.g) * k
    
        # Capital dynamics for dirty capital
        dk_d = self.theta * self.varphi * effective_output - (self.delta + self.g) * k_d
    
        if self.varphi < 0:
            # Convert disinvestment in dirty capital to additional clean capital investment
            converted_investment = abs(self.varphi) * k_d * (1 - kappa)
            dk += converted_investment
            stranded_assets = self.varphi * k_d * kappa
            dk_d -= abs(self.varphi) * k_d * (1 - kappa)

        # Ensure k_d does not become negative
        dk_d = max(dk_d, -k_d)
    
        # Calculate emissions
        d_ce = self.phi * k_d * (1 - self.eta)
    
        return [dk, dk_d, d_ce, stranded_assets]

    def solve(self, k0, kd0, t):
        initial_conditions = [k0, kd0, 0, 0]
        result = odeint(self.dkdt, initial_conditions, t)
        return result

# Parameters
rho, alpha, delta, gamma, T_initial, theta, g, L0 = 4, 0.3, 0.04, 0.0077, 1.2, 1, 0.0042, 7.795
phi, chi, zeta = 2, 0.0006, 0.00004
t = np.linspace(0, 200, 1000)
varphi_values = np.linspace(-1, 1, 20)
eta_values = np.linspace(0.1, 0.9, 20)
varphi_grid, eta_grid = np.meshgrid(varphi_values, eta_values)

final_k_d = np.zeros_like(varphi_grid)
final_output = np.zeros_like(varphi_grid)
final_emissions = np.zeros_like(varphi_grid)

for i in range(len(varphi_values)):
    for j in range(len(eta_values)):
        model = RCKmod(rho, alpha, delta, gamma, eta_values[j], varphi_values[i], phi, chi, zeta, T_initial, theta, g, L0)
        result = model.solve(160, 28, t)
        k, k_d, cumulative_emissions, stranded_assets = result.T
        output, emissions, _ = zip(*[model.output(kt, kt_d, ce, time) for kt, kt_d, ce, time in zip(k, k_d, cumulative_emissions, t)])
        final_k_d[j, i] = k_d[-1]
        final_output[j, i] = output[-1]
        final_emissions[j, i] = emissions[-1]

fig = plt.figure(figsize=(18, 6))
ax = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

# Surface plots with custom color map
ax.plot_surface(varphi_grid, eta_grid, final_k_d, cmap=cmap)
ax.set_xlabel('Varphi')
ax.set_ylabel('Eta')
ax.set_zlabel('Final Dirty Capital (K_d) in 2020')
ax.set_title('Dirty Capital')

ax2.plot_surface(varphi_grid, eta_grid, final_output, cmap=cmap)
ax2.set_xlabel('Varphi')
ax2.set_ylabel('Eta')
ax2.set_zlabel('Final Output')
ax2.set_title('Output')

ax3.plot_surface(varphi_grid, eta_grid, final_emissions, cmap=cmap)
ax3.set_xlabel('Varphi')
ax3.set_ylabel('Eta')
ax3.set_zlabel('Final Emissions')
ax3.set_title('Emissions')

plt.tight_layout()
plt.show()
