from artlib import FuzzyART, HypersphereART, GaussianART, SimpleARTMAP
from Data import generate_overlap_data
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, \
    silhouette_score, davies_bouldin_score
import pandas as pd
from tqdm import tqdm
import numpy as np


# List of models to evaluate with their respective hyperparameters
models = [
    {"model": FuzzyART, "params": {"rho": 0.0, "alpha": 1e-10, "beta": 1.0}},
    {"model": HypersphereART, "params": {"rho": 0.0, "alpha": 1e-10, "r_hat": 0.5 * np.sqrt(2), "beta": 1.0}},
    {"model": GaussianART, "params": {"rho": 0.0, "alpha": 1e-10, "sigma_init": np.array([0.5, 0.5])}},
]

# Function to load and generate data for the experiments
def data_loader(
        n_per_cluster=200,  # Number of data points per cluster
        sizes=[0.1, 0.2, 0.3],  # Sizes of the clusters
        spacings=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5],  # Spacing between clusters
        radii=[0.5],
        random_states=[42],  # Random seeds for reproducibility
        shuffles=[False, True]  # Whether to shuffle the data
):
    # Iterate over all combinations of parameters
    for size in sizes:
        for spacing in spacings:
            for radius in radii:
                for random_state in random_states:
                    for shuffle in shuffles:
                        # Generate synthetic overlapping data
                        X_train, y_train, X_test, y_test = generate_overlap_data(
                            num_points_per_class=n_per_cluster,
                            side_length=size,
                            spacing=spacing,
                            radius=radius,
                            random_state=random_state,
                            shuffle=shuffle
                        )
                        # Compute clustering metrics
                        db_score_train = davies_bouldin_score(X_train, y_train)
                        db_score_test = davies_bouldin_score(X_test, y_test)
                        sil_score_train = silhouette_score(X_train, y_train)
                        sil_score_test = silhouette_score(X_test, y_test)

                        # Store metadata for the current configuration
                        meta = {
                            "n_per_cluster": n_per_cluster,
                            "size": size, "spacing": spacing, "radius": radius,
                            "random_state": random_state, "shuffle": str(shuffle),
                            "db_train": db_score_train, "db_test": db_score_test,
                            "sil_train": sil_score_train, "sil_test": sil_score_test
                        }

                        # Yield the generated data and metadata
                        yield X_train, y_train, X_test, y_test, meta

# Function to load experiments by combining data and models
def experiment_loader(
        n_per_cluster=200,  # Number of data points per cluster
        sizes=[0.1, 0.2, 0.3, 0.4],  # Sizes of the clusters
        spacings=[0.1, 0.2, 0.3, 0.4, 0.5],  # Spacing between clusters
        radii=[0.5],
        random_states=[42],  # Random seeds for reproducibility
        shuffles=[False, True]  # Whether to shuffle the data
):
    # Iterate over all data and combine it with the models
    for X_train, y_train, X_test, y_test, meta in data_loader(
            n_per_cluster=n_per_cluster,
            sizes=sizes,
            spacings=spacings,
            radii=radii,
            random_states=random_states,
            shuffles=shuffles
    ):
        for model in models:
            # Add model information to metadata
            yield model, X_train, y_train, X_test, y_test, meta

# Function to run all experiments
def run_experiments(
        n_per_cluster=200,  # Number of data points per cluster
        sizes=[0.1, 0.2, 0.3, 0.4],  # Sizes of the clusters
        spacings=[0.1, 0.2, 0.3, 0.4, 0.5],  # Spacing between clusters
        radii=[0.5],
        random_states=[42],  # Random seeds for reproducibility
        shuffles=[False, True]  # Whether to shuffle the data
):
    # Total number of experiments to compute progress
    n_experiments = len(sizes) * len(spacings) * len(random_states) * len(shuffles) * len(models)
    results = []  # List to store results

    # Iterate through all experiment configurations
    for model, X_train, y_train, X_test, y_test, meta in tqdm(experiment_loader(
            n_per_cluster=n_per_cluster,
            sizes=sizes,
            spacings=spacings,
            radii=radii,
            random_states=random_states,
            shuffles=shuffles
    ), total=n_experiments):
        # Initialize SimpleARTMAP with the given model
        cls = SimpleARTMAP(model["model"](**model["params"]))
        cls.module_a.d_max_ = np.array([1.0, 1.0])  # Set maximum bounds
        cls.module_a.d_min_ = np.array([0.0, 0.0])  # Set minimum bounds

        # Prepare data for the model
        X_train_local = cls.prepare_data(X_train)
        X_test_local = cls.prepare_data(X_test)

        # Train the model and make predictions
        cls = cls.fit(X_train_local, y_train)
        y_pred = cls.predict(X_test_local)

        # Compute evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")

        # Update metadata with metrics and append to results
        meta.update({"accuracy": acc, "precision": pre, "f1": f1, "recall": rec,
                     "model": str(cls.module_a.__class__.__name__)})
        results.append(dict(meta))

        # Save results to a file in Parquet format
        df = pd.DataFrame(results)
        df.to_parquet("overlap.parquet")

# Entry point of the script
if __name__ == "__main__":
    n_per_cluster = 200  # Number of data points per cluster
    sizes = np.linspace(0.1, 0.5, 6).tolist()
    spacings = np.linspace(0.0, 0.5, 20).tolist()  # Spacing between clusters
    radii = [0.5]
    random_states = [42]  # Random seeds for reproducibility
    shuffles = [False, True]  # Whether to shuffle the data


    run_experiments(n_per_cluster=n_per_cluster, sizes=sizes, spacings=spacings,
                    radii=radii, random_states=random_states, shuffles=shuffles)