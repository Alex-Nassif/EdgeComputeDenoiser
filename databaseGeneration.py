import numpy as np
import scipy.io

# Genereate 50,000 clean and noisy signals
# Each signal has 512 time samples
def generate_denoising_dataset(num_signals=50000, signal_len=512):
    """
    Vectorized generation of thousands of unique noisy/clean signal pairs.
    """
    t = np.linspace(0, 1, signal_len, endpoint=False).reshape(1, -1)

    # Randomize the frequency, phase, and amplitude of the clean
    # signal's 1st sinusoidal component
    freqs = np.random.uniform(5, 50, size=(num_signals, 1))
    phases = np.random.uniform(0, 2 * np.pi, size=(num_signals, 1))
    amplitudes = np.random.uniform(0.5, 2.0, size=(num_signals, 1))

    # x1(t) = Asin(2pif1t + phi)
    clean_signals = amplitudes * np.sin(2 * np.pi * freqs * t + phases)

    # Randomize the frequency of the 2nd sinusoidal component
    freqs2 = np.random.uniform(10, 100, size=(num_signals, 1))
    clean_signals += (amplitudes / 2) * np.cos(2 * np.pi * freqs2 * t)
    
    signal_power = np.mean(clean_signals ** 2, axis=1, keepdims=True)

    # Randomize SNR to create noisy signal
    target_snr_db = np.random.uniform(2, 15, size=(num_signals, 1))
    target_snr_linear = 10 ** (target_snr_db / 10)
    
    noise_power = signal_power / target_snr_linear
    std_devs = np.sqrt(noise_power)
    
    pure_noise = np.random.normal(0, 1, size=(num_signals, signal_len))
    scaled_noise = pure_noise * std_devs

    # Adds white Gaussian noise
    noisy_signals = clean_signals + scaled_noise

    # Adds a new dimension to tell model that each time step contains
    # only one feature (signal amplitude)
    X = noisy_signals[..., np.newaxis]
    Y = clean_signals[..., np.newaxis]
    
    return X, Y

X_train, Y_train = generate_denoising_dataset(num_signals=50000, signal_len=512)

print(f"X_train Shape (Noisy Inputs): {X_train.shape}")
print(X_train[0, :5, 0]) 
print(f"Y_train Shape (Clean Targets): {Y_train.shape}")
np.savez_compressed('denoising_dataset.npz', noisy_signals=X_train, clean_labels=Y_train)
print("Dataset successfully generated and stored!")