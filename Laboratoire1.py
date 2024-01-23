import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Function to calculate Fourier coefficients for a given signal
def calculate_fourier_coefficients(signal, num_harmonics):
    dt = 1/1000  # Time step
    T = len(signal) * dt  # Total time period
    t = np.arange(0, T, dt)
    
    X = np.zeros(2 * num_harmonics + 1, dtype=complex)
    
    # Coefficient DC
    X[num_harmonics] = np.sum(signal) * dt
    
    # Coefficients de Fourier
    for k in range(1, num_harmonics + 1):
        X[num_harmonics + k] = np.sum(signal * np.exp(-1j * 2 * np.pi * k * t / T)) * dt
        X[num_harmonics - k] = np.conjugate(X[num_harmonics + k])
    
    return X

# Signal (a)
t_a = np.arange(0, 2, 1/1000)
x_a = 0.5 * signal.square(2 * np.pi * t_a, duty=0.5) 

# Signal (b)
t_b = np.arange(0, 2, 1/1000)
x_b = 0.5 * signal.square(2 * np.pi * 1 * t_b, duty=0.25) + 0.5

# Signal (c)
t_c = np.arange(0, 2, 1/1000)
x_c = np.sin(2 * np.pi * 1 * t_c) + 0.5 * np.sin(2 * np.pi * 2 * t_c)

# Number of harmonics
num_harmonics = 20

# Calculate Fourier coefficients for each signal
X_a = calculate_fourier_coefficients(x_a, num_harmonics)
X_b = calculate_fourier_coefficients(x_b, num_harmonics)
X_c = calculate_fourier_coefficients(x_c, num_harmonics)

# Plotting Fourier coefficients
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.stem(range(-num_harmonics, num_harmonics + 1), np.abs(X_a))
plt.title('Fourier Coefficients for Signal (a)')

plt.subplot(3, 1, 2)
plt.stem(range(-num_harmonics, num_harmonics + 1), np.abs(X_b))
plt.title('Fourier Coefficients for Signal (b)')

plt.subplot(3, 1, 3)
plt.stem(range(-num_harmonics, num_harmonics + 1), np.abs(X_c))
plt.title('Fourier Coefficients for Signal (c)')

plt.tight_layout()
plt.show()
