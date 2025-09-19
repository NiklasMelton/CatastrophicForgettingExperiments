from Data import load_usps
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from tqdm import tqdm

def compute_parallel_metric(X, y, metric_func, name, min_samples=2, n_jobs=-1):
    """Generic parallel computation of a clustering index."""
    n_samples = len(X)

    def compute_at(i):
        try:
            return i, metric_func(X[:i + 1], y[:i + 1])
        except ValueError:
            return i, np.nan

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_at)(i) for i in tqdm(range(min_samples, n_samples), desc=name)
    )

    values = np.full(n_samples, np.nan)
    for i, val in results:
        values[i] = val

    return values

def compute_all_indices(X_train, y_train, n_jobs=-1):
    """Compute all three clustering indices iteratively and in parallel."""
    silhouette = compute_parallel_metric(X_train, y_train, silhouette_score, "Silhouette", min_samples=2, n_jobs=n_jobs)
    db_index   = compute_parallel_metric(X_train, y_train, davies_bouldin_score, "Davies-Bouldin", min_samples=2, n_jobs=n_jobs)
    ch_index   = compute_parallel_metric(X_train, y_train, calinski_harabasz_score, "Calinski-Harabasz", min_samples=2, n_jobs=n_jobs)
    return silhouette, db_index, ch_index

def compute_indices_across_shuffles(random_state=42, shuffles=[False, True, "semi"], n_jobs=-1):
    """Compute all metrics across different shuffles."""
    results = {}

    for shuffle in shuffles:
        X_train, y_train, X_test, y_test = load_usps(random_state=random_state, shuffle=shuffle)

        silhouette, db_index, ch_index = compute_all_indices(X_train, y_train, n_jobs=n_jobs)

        results[str(shuffle)] = {
            "silhouette": silhouette,
            "davies_bouldin": db_index,
            "calinski_harabasz": ch_index
        }

    return results

if __name__ == "__main__":
    random_state = 42
    shuffles = [False, True, "semi"]
    n_jobs = 4  # Adjust this based on your CPU/memory

    results = compute_indices_across_shuffles(random_state=random_state, shuffles=shuffles, n_jobs=n_jobs)

    for shuffle, data in results.items():
        df = pd.DataFrame({
            "Sample Index": np.arange(len(data["silhouette"])),
            "Silhouette": data["silhouette"],
            "Davies-Bouldin": data["davies_bouldin"],
            "Calinski-Harabasz": data["calinski_harabasz"]
        })
        file_name = f"usps_iterative_clustering_indices_shuffle_{shuffle}.csv"
        df.to_csv(file_name, index=False)
        print(f"Results saved to {file_name}")
