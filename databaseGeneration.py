import numpy as np
import scipy.io

def generate_denoising_dataset(num_signals=50000, signal_len=512):
    """
    Vectorized generation of thousands of unique noisy/clean signal pairs.
    """
    t = np.linspace(0, 1, signal_len, endpoint=False).reshape(1, -1)
    
    freqs = np.random.uniform(5, 50, size=(num_signals, 1))
    phases = np.random.uniform(0, 2 * np.pi, size=(num_signals, 1))
    amplitudes = np.random.uniform(0.5, 2.0, size=(num_signals, 1))
    
    clean_signals = amplitudes * np.sin(2 * np.pi * freqs * t + phases)
    
    freqs2 = np.random.uniform(10, 100, size=(num_signals, 1))
    clean_signals += (amplitudes / 2) * np.cos(2 * np.pi * freqs2 * t)
    
    signal_power = np.mean(clean_signals ** 2, axis=1, keepdims=True)
    
    target_snr_db = np.random.uniform(2, 15, size=(num_signals, 1))
    target_snr_linear = 10 ** (target_snr_db / 10)
    
    noise_power = signal_power / target_snr_linear
    std_devs = np.sqrt(noise_power)
    
    pure_noise = np.random.normal(0, 1, size=(num_signals, signal_len))
    scaled_noise = pure_noise * std_devs
    
    noisy_signals = clean_signals + scaled_noise
    
    X = noisy_signals[..., np.newaxis]
    Y = clean_signals[..., np.newaxis]
    
    return X, Y

X_train, Y_train = generate_denoising_dataset(num_signals=50000, signal_len=512)

print(f"X_train Shape (Noisy Inputs): {X_train.shape}")
print(X_train[0, :5, 0]) 
print(f"Y_train Shape (Clean Targets): {Y_train.shape}")
np.savez_compressed('denoising_dataset.npz', noisy_signals=X_train, clean_labels=Y_train)
print("Dataset successfully generated and stored!")



npz_data = np.load('denoising_dataset.npz')
py_array_noisy = npz_data['noisy_signals']
py_array_clean = npz_data['clean_labels']
scipy.io.savemat('converted_file.mat', {'matlabMatrixNoisy': py_array_noisy, 'matlabMatrixClean': py_array_clean})
