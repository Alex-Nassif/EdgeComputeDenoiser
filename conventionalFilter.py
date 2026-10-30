import numpy as np
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt

# 1. Load generated dataset
data = np.load("denoising_dataset.npz")

# Use small subset of data (1000 signals) while testing code
# Each signal has shape (512, 1), so remove the final, single-channel dimension
noisy = data["noisy_signals"][:, :, 0]
clean = data["clean_labels"][:, :, 0]

print("Noisy signals:", noisy.shape)
print("Clean signals:", clean.shape)

# 2. Conventional Filter
fs = 512        # Sampling rate (512 samples/sec = 512 Hz)
fc = 110        # Cutoff frequency (Hz)
order = 4       # Butterworth filter order

sos = butter(
    N=order, 
    Wn=fc,
    btype="lowpass",
    fs=fs,
    output="sos")

# 3. Filter the noisy signals
# Forward-backward zero-phase filter
filtered_signals = sosfiltfilt(sos, noisy, axis=1)

# 4. Compare filtered signals to original clean signals
# Compute the root mean squared error (RMSE)
noisy_RMSE = np.sqrt(np.mean((noisy - clean) ** 2))
filtered_RMSE = np.sqrt(np.mean((filtered_signals - clean) ** 2))

print(f"Noisy input RMSE: {noisy_RMSE:.4f}")
print(f"Filtered output RMSE: {filtered_RMSE:.4f}")

RMSE_decrease = (noisy_RMSE - filtered_RMSE) / (noisy_RMSE) * 100
print(f"RMSE decrease: {RMSE_decrease:.2f}%")

# 5. Plot representative signals
# Find noisy signals with 10%, 50%, and 90% SNRs

signal_power = np.mean(clean**2, axis=1)
noise_power = np.mean((noisy - clean)**2, axis=1)

# Compute signal-to-noise ratios
SNR = 10 * np.log10(signal_power / noise_power)
rep_percentiles = [10, 50, 90]
rep_indices = []

# Find signals with SNRs at the desired percentiles
for percent in rep_percentiles:
    rep_SNR = np.percentile(SNR, percent)
    index = np.argmin(np.abs(SNR - rep_SNR))
    rep_indices.append(index)
    print(
        f"{percent}th percentile: Signal {index}, " 
        f"SNR = {SNR[index]:.2f} dB")

# Plot
t = np.arange(fs) / fs
for i in rep_indices:
    noisy_signal = noisy[i]
    clean_signal = clean[i]
    filtered_signal = filtered_signals[i]

    plt.figure(figsize=(12, 5))
    plt.plot(t, clean_signal, label="Clean Signal", linewidth=2)
    plt.plot(t, noisy_signal, label="Noisy Signal", alpha=0.5, linewidth=1)
    plt.plot(t, filtered_signal, label="Filtered Signal", linewidth=1.5)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(f"Clean, Noisy, and Filtered Signals (index={i})")

    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"signal_{i}_conventional_denoising.png", dpi=300)
    plt.show()