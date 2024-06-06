import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.colors import LinearSegmentedColormap

# Define the color map
colors = ["blue", "purple", "red"]
cmap = LinearSegmentedColormap.from_list("custom_turquoise_green_yellow", colors, N=100)

class RCKmod:
    def __init__(self, rho, alpha, delta, gamma, eta, varphi, phi, chi, zeta, T_initial, theta, g, L0, noise_std=0.013):
        self.rho = rho  # time preference
        self.alpha = alpha
        self.delta = delta  # depreciation rate
        self.gamma = gamma  # damage multiplier
        self.eta = eta  # abatement
        self.chi = chi  # temperature climate response to emissions
        self.zeta = zeta  # MAC slope
        self.varphi = varphi  # allocation of investment to dirty or clean capital
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
            return -theta1 * r ** 0.5 / (k + k_d)
        return 0

    def subsistence(self, effective_output, dk, dk_d):
        diff = effective_output - dk - dk_d
        return np.exp(-1 / 0.000000001 * (diff))

    def dkdt(self, k_kd_ce, t):
        k, k_d, cumulative_emissions, stranded_assets = k_kd_ce
        effective_output, temp, omega = self.output(k, k_d, cumulative_emissions, t)
        kappa = self.Kappa(self.varphi, k, k_d)

        # Capital dynamics for clean capital
        dk = self.theta * (1 - abs(self.varphi)) * effective_output - ((1 - abs(self.varphi)) * effective_output) ** 2 / (k + k_d) - (self.delta + self.g) * k
    
        # Capital dynamics for dirty capital
        dk_d = self.theta * self.varphi * effective_output - ((1 - abs(self.varphi)) * effective_output) ** 2 / (k + k_d) - (self.delta + self.g) * k_d
    
        if self.varphi < 0:
            # Convert disinvestment in dirty capital to additional clean capital investment
            converted_investment = abs(self.varphi) * k_d * (1 - kappa)
            dk += converted_investment
            dk_d -= abs(self.varphi) * k_d * (1 - kappa)
            stranded_assets = dk_d * kappa  
        # Ensure k_d does not become negative
        dk_d = max(dk_d, -k_d)
    
        # Calculate emissions
        d_ce = self.phi * k_d * (1 - self.eta)
    
        return [dk, dk_d, d_ce, stranded_assets]

    def solve(self, k0, kd0, t):
        initial_conditions = [k0, kd0, 0, 0]
        result = odeint(self.dkdt, initial_conditions, t)
        return result

# Constants and model parameters
rho, alpha, delta, gamma, eta, phi, chi, zeta, T_initial, theta, g, L0 = 4, 0.3, 0.04, 0.0077, 0, 2, 0.0006, 0.00004, 1.2, 1, 0.0042, 7.795
t = np.linspace(0, 100, 1000)  # 100 years
varphi_values = np.linspace(-1, 1, 21)

# To find the highest varphi value where temperature stays below 1.5 degrees
highest_varphi_below_1_5_degrees = None

for varphi in varphi_values:
    model = RCKmod(rho, alpha, delta, gamma, eta, varphi, phi, chi, zeta, T_initial, theta, g, L0)
    result = model.solve(160, 28, t)
    k, k_d, cumulative_emissions, stranded_assets = result.T
    output, temperature, _ = zip(*[model.output(kt, kt_d, ce, time) for kt, kt_d, ce, time in zip(k, k_d, cumulative_emissions, t)])
    
    # Check if the maximum temperature stays below 1.5 degrees
    if max(temperature) < 1.5:
        highest_varphi_below_1_5_degrees = varphi

# Filter varphi values to only include those that keep temperature below 1.5 degrees
valid_varphi_values = [varphi for varphi in varphi_values if varphi <= highest_varphi_below_1_5_degrees]

# Optimize output for valid varphi values
optimal_varphi = None
max_output = -np.inf

for varphi in valid_varphi_values:
    model = RCKmod(rho, alpha, delta, gamma, eta, varphi, phi, chi, zeta, T_initial, theta, g, L0)
    result = model.solve(160, 28, t)
    k, k_d, cumulative_emissions, stranded_assets = result.T
    output, temperature, _ = zip(*[model.output(kt, kt_d, ce, time) for kt, kt_d, ce, time in zip(k, k_d, cumulative_emissions, t)])
    
    total_output = sum(output)
    if total_output > max_output:
        max_output = total_output
        optimal_varphi = varphi

# Print the optimal varphi value
print("Optimal varphi value for maximum output while keeping temperature below 1.5 degrees:", optimal_varphi)

# Setup the figure and axes for plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
fig.delaxes(axes[0, 0])  # Remove the first subplot for the legend

colors = cmap(np.linspace(0, 1, len(varphi_values)))

# Define the starting year
starting_year = 2024
years = starting_year + np.array(t)
years_50 = starting_year + np.array(t[:500])  # First 50 years

# Plotting several varphi values for dirty capital
for varphi, color in zip(varphi_values, colors):
    model = RCKmod(rho, alpha, delta, gamma, eta, varphi, phi, chi, zeta, T_initial, theta, g, L0)
    result = model.solve(160, 28, t)
    k, k_d, cumulative_emissions, stranded_assets = result.T
    output, temperature, _ = zip(*[model.output(kt, kt_d, ce, time) for kt, kt_d, ce, time in zip(k, k_d, cumulative_emissions, t)])

    # Plot results in the respective subplots
    axes[1, 0].plot(years, k_d, color=color, alpha=0.5)

# Plot the optimal varphi for dirty capital
model = RCKmod(rho, alpha, delta, gamma, eta, optimal_varphi, phi, chi, zeta, T_initial, theta, g, L0)
result = model.solve(160, 28, t)
k, k_d, cumulative_emissions, stranded_assets = result.T
output, temperature, _ = zip(*[model.output(kt, kt_d, ce, time) for kt, kt_d, ce, time in zip(k, k_d, cumulative_emissions, t)])

# Plot the optimal varphi in green for dirty capital
axes[1, 0].plot(years, k_d, color='green', linewidth=2)

# Plot the optimal varphi in green for temperature and stranded assets
change_stranded_assets_optimal = np.gradient(stranded_assets, t)
axes[0, 1].plot(years_50, temperature[:500], color='green', linewidth=2)
axes[1, 1].plot(years_50, change_stranded_assets_optimal[:500], color='green', linewidth=2)

# Set labels for each subplot
axes[0, 1].set_ylabel('Temperature in °C')
axes[1, 0].set_ylabel('Dirty Capital in Billion USD/y')
axes[1, 1].set_ylabel('Stranded Assets in Billion USD/y')

# Set titles for clarity
axes[0, 1].set_title('Temperature (Optimal Varphi)')
axes[1, 0].set_title('Dirty Capital (K_d) (various varphi)')
axes[1, 1].set_title('Change in Stranded Assets (Optimal Varphi)')

# Set x-axis label only for the bottom plots
axes[1, 0].set_xlabel('Year')
axes[1, 1].set_xlabel('Year')

# Add a vertical colorbar as a legend on the left side
cax = fig.add_axes([0.1, 0.52, 0.02, 0.4])  # Position: left, bottom, width, height
norm = plt.Normalize(vmin=-1, vmax=1)
cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical')
cb.set_label('Varphi')

plt.show()