import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.colors import LinearSegmentedColormap

# Define the new color map
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

    def solve(self, k0, kd0, t, initial_emissions):
        initial_conditions = [k0, kd0, initial_emissions, 0]
        result = odeint(self.dkdt, initial_conditions, t)
        return result

# Constants and model parameters
rho, alpha, delta, gamma, varphi, phi, chi, zeta, T_initial, theta, g, L0 = 4, 0.3, 0.04, 0.0077, 1, 2, 0.0006, 0.00004, 1.2, 1, 0.0042, 7.795
t = np.linspace(0, 100, 1000)
eta_values = np.linspace(0, 1, 11)

# Setup the figure and axes for plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = cmap(np.linspace(0, 1, len(eta_values)))

# Define the starting year
starting_year = 2024
years = starting_year + np.array(t)

initial_emissions = 56

for eta, color in zip(eta_values, colors):
    model = RCKmod(rho, alpha, delta, gamma, eta, varphi, phi, chi, zeta, T_initial, theta, g, L0)
    result = model.solve(160, 28, t, initial_emissions)
    k, k_d, cumulative_emissions, stranded_assets = result.T
    output, temperature, _ = zip(*[model.output(kt, kt_d, ce, time) for kt, kt_d, ce, time in zip(k, k_d, cumulative_emissions, t)])

    # Plot results in the respective subplots
    axes[0, 0].plot(years, output, label=f'Eta={eta:.2f}', color=color, alpha=0.5)
    axes[0, 1].plot(years, temperature, color=color, alpha=0.5)
    axes[1, 0].plot(years, k_d, color=color, alpha=0.5)
    axes[1, 1].plot(years, cumulative_emissions, color=color, alpha=0.5)

# Set labels for each subplot
axes[0, 0].set_ylabel('Output in 2020 US$ trillion')
axes[0, 1].set_ylabel('Temperature in °C')
axes[1, 0].set_ylabel('Dirty Capital in 2020 US$ trillion')
axes[1, 1].set_ylabel('Emissions in Gt')

# Set titles for clarity
axes[0, 0].set_title('Output')
axes[0, 1].set_title('Temperature')
axes[1, 0].set_title('Dirty Capital (K_d)')
axes[1, 1].set_title('Emissions')

# Set x-axis label only for the bottom plots
axes[1, 0].set_xlabel('Year')
axes[1, 1].set_xlabel('Year')

# Add legend
axes[0, 0].legend(loc='upper left', framealpha=1)

plt.tight_layout()
plt.show()
