from Data import load_usps
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
from tqdm import tqdm

# Function to compute iterative silhouette index for a dataset
def compute_iterative_silhouette(X_train, y_train):
    """Compute iterative silhouette index for training data."""
    n_samples = len(X_train)
    iterative_silhouette = np.full(n_samples, np.nan)  # Initialize with NaN

    for i in tqdm(range(1, n_samples)):
        try:
            # Compute silhouette score for data up to and including the current sample
            iterative_silhouette[i] = silhouette_score(X_train[:i + 1], y_train[:i + 1])
        except ValueError:
            # Handle cases where silhouette score cannot be computed
            iterative_silhouette[i] = np.nan

    return iterative_silhouette

# Function to load data and compute iterative silhouette indices for all shuffles
def compute_silhouette_across_shuffles(random_state=42, shuffles=[False, True, "semi"]):
    results = {}

    for shuffle in shuffles:
        # Load the dataset with the specified shuffle
        X_train, y_train, X_test, y_test = load_usps(random_state=random_state, shuffle=shuffle)

        # Compute the iterative silhouette index
        iterative_silhouette = compute_iterative_silhouette(X_train, y_train)

        # Store results in a dictionary
        results[str(shuffle)] = {
            "iterative_silhouette": iterative_silhouette
        }

    return results

if __name__ == "__main__":
    # Define experiment parameters
    random_state = 42
    shuffles = [False, True, "semi"]

    # Compute iterative silhouette indices for all shuffles
    silhouette_results = compute_silhouette_across_shuffles(random_state=random_state, shuffles=shuffles)

    # Save results to CSV files
    for shuffle, data in silhouette_results.items():
        df = pd.DataFrame({
            "Sample Index": np.arange(len(data["iterative_silhouette"])),
            "Iterative Silhouette": data["iterative_silhouette"]
        })
        file_name = f"iterative_silhouette_shuffle_{shuffle}.csv"
        df.to_csv(file_name, index=False)
        print(f"Results saved to {file_name}")
