"""
UWOC Spectral Channel Model & Blue-Green Analysis

Author: Riant Dadra

Description:
Simulation of underwater optical channel over 400–800 nm wavelength range,
modeling absorption, scattering, and extinction based on ocean parameters.
Compares Blue (450 nm) and Green (520 nm) laser performance with power
efficiency and attenuation analysis.

Features:
- Spectral attenuation modeling (a, b, c coefficients)
- Chlorophyll and particle effects
- Blue vs Green transmitter comparison
- Optical power efficiency analysis
- Continuous spectral plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 1. Define the Wavelength Range

wavelengths = np.linspace(400, 800, 500) # lambda in nm
# print("wavelengths",wavelengths)
# 2. Environmental & System Constants

Cc = 1.2       # Total chlorophyll concentration (mg/m^3) coastal ocean standard value
Cc_0 = 1.0     # Reference chlorophyll concentration (mg/m^3)
Cs = 0.05      # Concentration of small particles
Cl = 0.05      # Concentration of large particles

# Constants for Humic Acid
ah_0 = 18.828  # Specific absorption coefficient of humic acid (m^2/mg)
k_h = 0.01105  # Constant for humic acid (nm^-1)

# 3. Base Empirical Functions (Approximations for pure water)

def pure_water_absorption(lam): # aw(lembda) & lam denotes wavelength
    return 0.01 + 0.5 * np.exp(-((lam - 450) / 50)**2) + 0.05 * np.exp((lam - 600) / 100)

def specific_chlorophyll_absorption(lam): # ac^(lembda)
    return 0.05 * np.exp(-((lam - 440) / 20)**2) + 0.02 * np.exp(-((lam - 675) / 20)**2)

def pure_water_scattering(lam): # bw^(lembda)
    return 0.00288 * (500 / lam)**4.32

# 4. Calculating Continuous Coefficients

aw = pure_water_absorption(wavelengths)
ac_0 = specific_chlorophyll_absorption(wavelengths)
C_h = 0.19334 * Cc * np.exp(0.12343) * (Cc / Cc_0)

a_lambda = aw + ac_0 * (Cc / Cc_0)**0.602 + ah_0 * C_h * np.exp(-k_h * wavelengths)

bw = pure_water_scattering(wavelengths)
bs_0 = 1.151302 * (400 / wavelengths)**1.7
bl_0 = 0.341074 * (400 / wavelengths)**0.3

b_lambda = bw + bs_0 * Cs + bl_0 * Cl
c_lambda = a_lambda + b_lambda

# 5. BLUE vs GREEN TRANSMITTER POWER ANALYSIS

print("--- TRANSMITTER POWER ANALYSIS ---")
V = 5          # Voltage (Volts)
I = 1          # Current (Ampere)
P_electrical = V * I   # Electrical Power (W)

eta_blue = 0.45    # Blue laser efficiency (45%)
eta_green = 0.40   # Green laser efficiency (40%)

PT_blue = eta_blue * P_electrical
PT_green = eta_green * P_electrical

print(f"Electrical Input Power = {P_electrical:.2f} W")
print(f"Blue Transmitted Optical Power = {PT_blue:.2f} W")
print(f"Green Transmitted Optical Power = {PT_green:.2f} W\n")

# 6. DISCRETE BLUE VS GREEN CHANNEL COMPARISON

lambda_blue = 450
lambda_green = 520

idx_blue = np.argmin(np.abs(wavelengths - lambda_blue))
idx_green = np.argmin(np.abs(wavelengths - lambda_green))

a_blue, a_green = a_lambda[idx_blue], a_lambda[idx_green]
b_blue, b_green = b_lambda[idx_blue], b_lambda[idx_green]
c_blue, c_green = c_lambda[idx_blue], c_lambda[idx_green]

print("--- DISCRETE CHANNEL COEFFICIENTS (m^-1) ---")
print(f"Blue  (450 nm) -> a: {a_blue:.4f} | b: {b_blue:.4f} | c: {c_blue:.4f}")
print(f"Green (520 nm) -> a: {a_green:.4f} | b: {b_green:.4f} | c: {c_green:.4f}\n")

# =====================================================================
# 7. PLOTTING FIGURE 1: SPECTRAL CONTINUOUS GRAPHS
# =====================================================================
plt.figure(figsize=(10, 15))

# Plot 1: Absorption
plt.subplot(3, 1, 1)
plt.plot(wavelengths, a_lambda, 'r-', linewidth=2)
plt.title('Absorption Coefficient vs Wavelength', fontsize=14)
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Absorption a($\lambda$) ($m^{-1}$)', fontsize=12)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
plt.grid(True, linestyle='--', alpha=0.7)

# Plot 2: Scattering
plt.subplot(3, 1, 2)
plt.plot(wavelengths, b_lambda, 'b-', linewidth=2)
plt.title('Scattering Coefficient vs Wavelength', fontsize=14)
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Scattering b($\lambda$) ($m^{-1}$)', fontsize=12)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
plt.grid(True, linestyle='--', alpha=0.7)

# Plot 3: Extinction
plt.subplot(3, 1, 3)
plt.plot(wavelengths, c_lambda, 'g-', linewidth=2)
plt.title('Total Extinction Coefficient vs Wavelength', fontsize=14)
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Extinction c($\lambda$) ($m^{-1}$)', fontsize=12)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
plt.axvspan(400, 550, color='yellow', alpha=0.3, label='Optimal Window (400-550nm)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()

# =====================================================================
# 8. PLOTTING FIGURE 2: BLUE VS GREEN BAR CHART
# =====================================================================
plt.figure(figsize=(8, 6))

labels = ['Blue (450nm)', 'Green (520nm)']
absorption_vals = [a_blue, a_green]
scattering_vals = [b_blue, b_green]
attenuation_vals = [c_blue, c_green]

x = np.arange(len(labels))
width = 0.25

plt.bar(x - width, absorption_vals, width, label='Absorption (a)', color='salmon')
plt.bar(x, scattering_vals, width, label='Scattering (b)', color='skyblue')
plt.bar(x + width, attenuation_vals, width, label='Total Attenuation (c)', color='lightgreen')

plt.xticks(x, labels, fontsize=12)
plt.ylabel('Coefficient Value ($m^{-1}$)', fontsize=12)
plt.title('Rigorous Model Comparison: Blue vs Green Transmitters', fontsize=14)
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

