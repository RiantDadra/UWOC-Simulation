"""
UWOC BER Simulation - OOK / QAM

Author: Riant Dadra

Description:
This script simulates BER performance of UWOC systems using
Beer-Lambert attenuation and AWGN channel.

Features:
- OOK / 4-QAM modulation
- BER vs SNR analysis
- Channel attenuation modeling
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# Simulation parameters
N_bits = 10**5
SNR_dB = np.arange(0, 21, 2)

# UWOC parameters
distances = [5, 10, 15]  # meters

# Water types (attenuation coefficient c = a + b)
water_types = {
    "Clear": 0.15,
    "Coastal": 0.30,
    "Turbid": 0.60
}

def generate_bits(N):
    return np.random.randint(0, 2, N)

def ook_mod(bits):
    return bits.astype(float)

def qam4_mod(bits):
    bits = bits.reshape(-1, 2)
    symbols = (2*bits[:,0]-1) + 1j*(2*bits[:,1]-1)
    symbols /= np.sqrt(2)
    return symbols

def uwoc_channel(signal, distance, c):
    h = np.exp(-c * distance)  # Beer-Lambert law
    return signal * h

def add_awgn(signal, snr_db):
    snr_linear = 10**(snr_db/10)
    power = np.mean(np.abs(signal)**2)
    noise_var = power / snr_linear

    noise = np.sqrt(noise_var/2) * (
        np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape)
    )

    return signal + noise

def ook_demod(rx, h):
    threshold = h / 2
    return (rx > threshold).astype(int)

def qam4_demod(rx):
    bits = []
    for r in rx:
        bits.append(1 if r.real > 0 else 0)
        bits.append(1 if r.imag > 0 else 0)
    return np.array(bits)

def compute_ber(tx_bits, rx_bits):
    return np.mean(tx_bits != rx_bits)

#frequency diversity
def spatial_diversity(signal, distance, c, branches=2):
    combined = 0
    for _ in range(branches):
        h = np.exp(-c * distance)
        faded = signal * h
        combined += faded
    return combined / branches

def frequency_diversity(signal, distance, c):
    h1 = np.exp(-c * distance)
    h2 = np.exp(-0.8 * c * distance)
    return (signal*h1 + signal*h2) / 2

ook_results = {}

for water, c in water_types.items():
    ber_dist = []

    for d in distances:
        ber_snr = []

        for snr in SNR_dB:
            bits = generate_bits(N_bits)

            tx = ook_mod(bits)

            # Apply frequency diversity
            ch = frequency_diversity(tx, d, c)

            # Add noise
            rx = add_awgn(ch, snr)

            # Effective channel gain
            h_eff = (np.exp(-c*d) + np.exp(-0.8*c*d)) / 2

            # Detection
            bits_hat = ook_demod(rx.real, h_eff)

            # BER
            ber = compute_ber(bits, bits_hat)

            ber_snr.append(ber)

        ber_dist.append(ber_snr)

    ook_results[water] = ber_dist

# 4 QAM BER Simulation
qam_results = {}

for water, c in water_types.items():
    ber_dist = []

    for d in distances:
        ber_snr = []

        for snr in SNR_dB:
            bits = generate_bits(N_bits)
            bits = bits[:len(bits)//2*2]

            tx = qam4_mod(bits)
            ch = uwoc_channel(tx, d, c)
            rx = add_awgn(ch, snr)

            bits_hat = qam4_demod(rx)
            ber = compute_ber(bits, bits_hat)

            ber_snr.append(ber)

        ber_dist.append(ber_snr)

    qam_results[water] = ber_dist

# OOK results
for water in water_types:
    for i, d in enumerate(distances):
        plt.semilogy(SNR_dB, ook_results[water][i], label=f"{water}, {d}m")

plt.title("OOK BER vs SNR")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.legend()
plt.grid()
plt.show()

# 4 QAM Results
for water in water_types:
    for i, d in enumerate(distances):
        plt.semilogy(SNR_dB, qam_results[water][i], label=f"{water}, {d}m")

plt.title("4-QAM BER vs SNR")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.legend()
plt.grid()
plt.show()

ch = spatial_diversity(tx, d, c, branches=3)

ch = frequency_diversity(tx, d, c)

import matplotlib.pyplot as plt


eps = 1e-12

# ============================================
# 1) OOK vs 4-QAM Comparison at Fixed Distance
# ============================================

fixed_distance = 15
fixed_index = distances.index(fixed_distance)

plt.figure(figsize=(9, 6))

for water in water_types:
    # Replace zeros with eps for log-scale plotting
    ook_curve = [max(ber, eps) for ber in ook_results[water][fixed_index]]
    qam_curve = [max(ber, eps) for ber in qam_results[water][fixed_index]]

    # OOK curve
    plt.semilogy(
        SNR_dB,
        ook_curve,
        marker='o',
        linestyle='--',
        linewidth=2,
        label=f"OOK - {water}"
    )

    # 4-QAM curve
    plt.semilogy(
        SNR_dB,
        qam_curve,
        marker='s',
        linestyle='-',
        linewidth=2,
        label=f"4-QAM - {water}"
    )

plt.title(f"OOK vs 4-QAM Comparison at Fixed Distance = {fixed_distance} m", fontsize=13)
plt.xlabel("SNR (dB)", fontsize=12)
plt.ylabel("Bit Error Rate (BER)", fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend(fontsize=10, loc='best')
plt.tight_layout()

# Save image for PPT / Overleaf
plt.savefig("comparison_plot.png", dpi=300, bbox_inches='tight')
plt.show()


# ============================================
# 2) BER vs Distance at Fixed SNR
# ============================================

fixed_snr = 10
snr_index = list(SNR_dB).index(fixed_snr)

plt.figure(figsize=(9, 6))

for water in water_types:
    # BER values across distance
    ook_ber_dist = [ook_results[water][i][snr_index] for i in range(len(distances))]
    qam_ber_dist = [qam_results[water][i][snr_index] for i in range(len(distances))]

    # Replace zeros with eps for log-scale plotting
    ook_ber_dist = [max(ber, eps) for ber in ook_ber_dist]
    qam_ber_dist = [max(ber, eps) for ber in qam_ber_dist]

    # OOK plot
    plt.semilogy(
        distances,
        ook_ber_dist,
        marker='o',
        linestyle='--',
        linewidth=2,
        label=f"OOK - {water}"
    )

    # 4-QAM plot
    plt.semilogy(
        distances,
        qam_ber_dist,
        marker='s',
        linestyle='-',
        linewidth=2,
        label=f"4-QAM - {water}"
    )

plt.title(f"BER Variation with Transmission Distance at Fixed SNR = {fixed_snr} dB", fontsize=13)
plt.xlabel("Distance (m)", fontsize=12)
plt.ylabel("Bit Error Rate (BER)", fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend(fontsize=10, loc='best')
plt.tight_layout()

# Save image for PPT / Overleaf
plt.savefig("distance_plot.png", dpi=300, bbox_inches='tight')
plt.show()

