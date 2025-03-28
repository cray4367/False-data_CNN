import numpy as np
import pandas as pd
import os

def generate_dataset(num_samples=10000, num_features=10, test=False):
    """
    Generates a synthetic dataset with specified characteristics
    - Training data: More clustered patterns
    - Test data: More varied values with outliers
    """
    np.random.seed(42)
    
    # Base features with different distributions
    data = {}
    for i in range(num_features):
        if test:
            # Test data: wider ranges and uniform distribution
            data[f'feature_{i}'] = np.random.uniform(-20, 20, num_samples)
            
            # Add 10% outliers
            if i % 2 == 0:
                data[f'feature_{i}'] *= np.random.choice([1, 5], num_samples, p=[0.9, 0.1])
        else:
            # Training data: normal distribution with some pattern
            base = np.random.normal(i*0.5, 2, num_samples)
            noise = np.random.normal(0, 1, num_samples)
            data[f'feature_{i}'] = base + noise

    df = pd.DataFrame(data)
    
    # Create labels based on non-linear combination of features
    if test:
        # More complex relationship for test data
        label_condition = (
            (df['feature_0'] * df['feature_1']) > np.random.normal(15, 5, num_samples)
        ) & (
            (df['feature_2'] + df['feature_3']**2) < np.random.uniform(50, 200, num_samples)
        )
    else:
        # Training data relationship
        label_condition = (
            (df['feature_0'] + df['feature_1']*2) > np.random.normal(5, 2, num_samples)
        )

    df['label'] = np.where(label_condition, 1, 0)
    
    # Add noise to labels
    flip_mask = np.random.rand(num_samples) < 0.05  # 5% label noise
    df['label'] = np.where(flip_mask, 1 - df['label'], df['label'])
    
    return df

if __name__ == "__main__":
    # Generate and save datasets
    train_df = generate_dataset(num_samples=10000, test=False)
    test_df = generate_dataset(num_samples=2000, test=True)
    
    os.makedirs('data', exist_ok=True)
    train_df.to_csv('data/dataset.csv', index=False)
    test_df.to_csv('data/test_dataset.csv', index=False)
    print("Datasets generated:\n- data/dataset.csv (10,000 samples)\n- data/test_dataset.csv (2,000 samples)")