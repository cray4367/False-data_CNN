import numpy as np
import pandas as pd

# Parameters
num_samples = 20000  # Number of rows (samples)
attack_ratio = 0.3  # Fraction of samples with FDI attacks

# Generate normal smart grid data
np.random.seed(42)
voltage = np.random.uniform(220, 240, size=num_samples)  # Voltage (V)
current = np.random.uniform(5, 15, size=num_samples)  # Current (A)
power_factor = np.random.uniform(0.8, 1.0, size=num_samples)  # Power Factor
power = voltage * current * power_factor  # Power (W)
frequency = np.random.uniform(49.5, 50.5, size=num_samples)  # Frequency (Hz)

# Combine features into a matrix
normal_data = np.column_stack((voltage, current, power_factor, power, frequency))

# Inject FDI Attacks
n_attack_samples = int(num_samples * attack_ratio)
attack_indices = np.random.choice(num_samples, n_attack_samples, replace=False)

fdi_data = normal_data.copy()

# Apply attacks
fdi_data[attack_indices, 0] *= np.random.uniform(0.5, 1.5, size=n_attack_samples)  # Voltage attack
fdi_data[attack_indices, 3] *= np.random.uniform(0.5, 2.0, size=n_attack_samples)  # Power fluctuation
fdi_data[attack_indices, 4] += np.random.uniform(-1.5, 1.5, size=n_attack_samples)  # Frequency deviation

# Labels (0 = normal, 1 = FDI attack)
labels = np.zeros(num_samples)
labels[attack_indices] = 1

# Convert to DataFrame
columns = ["Voltage", "Current", "Power Factor", "Power", "Frequency"]
df = pd.DataFrame(fdi_data, columns=columns)
df['Label'] = labels

# Save dataset to CSV
file_name = "dataset.csv"
df.to_csv(file_name, index=False)
print(f"Dataset saved as {file_name}")

# Summary
print("\n📊 *Dataset Summary*")
print(f"Total samples: {num_samples}")
print(f"FDI attack samples: {n_attack_samples}")
print(f"Normal samples: {num_samples - n_attack_samples}")
print(f"Shape: {df.shape}")
print(df.head())
