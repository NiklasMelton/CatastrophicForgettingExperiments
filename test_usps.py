from artlib import FuzzyART, HypersphereART, GaussianART, SimpleARTMAP, normalize
from Data import load_usps
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, \
    silhouette_score, davies_bouldin_score, confusion_matrix
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.neural_network import MLPClassifier
from CNNClassifier import CNNClassifier
import json


CNN_PARAMS = {
    "input_shape": (16, 16, 1),
    "num_classes": 10,
    "conv_layers": [(32, (3, 3))],
    "pool_size": (2, 2),
    "dense_layers": [128],
    "dropout": 0.5,
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": 32
}



# List of models to evaluate with their respective hyperparameters
models = [
    {"model": CNNClassifier, "params": CNN_PARAMS},
    {"model": FuzzyART, "params": {"rho": 0.0, "alpha": 1e-10, "beta": 1.0}},
    # {"model": HypersphereART, "params": {"rho": 0.0, "alpha": 1e-10, "r_hat": 0.5 *
    #                                                                           np.sqrt(256), "beta": 1.0}},
    # {"model": GaussianART, "params": {"rho": 0.0, "alpha": 1e-10, "sigma_init":
    #     np.array(256*[0.5])}},
    {"model": FuzzyART, "params": {"rho": 1.0, "alpha": 1e-10, "beta": 1.0}},
    # {"model": HypersphereART, "params": {"rho": 1.0, "alpha": 1e-10, "r_hat": 0.5 *
    #                                                                           np.sqrt(
    #                                                                               256),
    #                                      "beta": 1.0}},
    # {"model": GaussianART, "params": {"rho": 1.0, "alpha": 1e-10, "sigma_init":
    #     np.array(256 * [0.5])}},

]

# Function to load and generate data for the experiments
def data_loader(
        random_states=[42],  # Random seeds for reproducibility
        shuffles=[False, True, "semi"]  # Whether to shuffle the data
):
    for random_state in random_states:
        for shuffle in shuffles:
            # load USPS data
            X_train, y_train, X_test, y_test = load_usps(
                random_state=random_state,
                shuffle=shuffle
            )
            # Compute clustering metrics
            # db_score_train = davies_bouldin_score(X_train, y_train)
            # db_score_test = davies_bouldin_score(X_test, y_test)
            sil_score_train = silhouette_score(X_train, y_train)
            sil_score_test = silhouette_score(X_test, y_test)

            # Store metadata for the current configuration
            meta = {
                "random_state": random_state, "shuffle": str(shuffle),
                # "db_train": db_score_train, "db_test": db_score_test,
                "sil_train": sil_score_train, "sil_test": sil_score_test
            }

            # Yield the generated data and metadata
            yield X_train, y_train, X_test, y_test, meta

# Function to load experiments by combining data and models
def experiment_loader(
        random_states=[42],  # Random seeds for reproducibility
        shuffles=[False, True, "semi"]  # Whether to shuffle the data
):
    # Iterate over all data and combine it with the models
    for X_train, y_train, X_test, y_test, meta in data_loader(
            random_states,
            shuffles
    ):
        for model in models:
            # Add model information to metadata
            meta.update(
                {
                    "model": str(model["model"].__name__),
                    "params": str(model["params"]),
                    "rho": model["params"].get("rho", None)
                }
            )
            yield model, X_train, y_train, X_test, y_test, meta


def class_accuracy(y_true, y_pred, c):
    true_mask = y_true == c
    c_acc = sum(y_pred[true_mask] == c) / sum(true_mask)
    return c_acc


def step_train(cls, X_train, y_train, X_test, y_test, **fit_kwargs):
    metrics = []
    target_names = sorted(np.unique(y_test).tolist())

    # Generate batches
    change_indices = np.where(np.diff(np.sort(y_train)) != 0)[0] + 1
    change_indices = np.concatenate(([0], change_indices, [len(y_train)]))
    batch_idxs = list(zip(change_indices[:-1], change_indices[1:]))

    for start_idx, end_idx in batch_idxs:

        # Extract batch data
        X_batch = X_train[start_idx:end_idx]
        y_batch = y_train[start_idx:end_idx]

        cls = cls.partial_fit(X_batch, y_batch, **fit_kwargs)
        y_pred = cls.predict(X_test)

        # Compute evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred, average="weighted", zero_division=np.nan)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=np.nan)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=np.nan)
        cfm = confusion_matrix(y_test, y_pred).reshape((-1,))

        if start_idx > 0:
            sil = silhouette_score(X_train.reshape((-1, 256))[:end_idx], y_train[
                                                                       :end_idx])
        else:
            sil = np.nan

        acc_by_class = {
            str(name): class_accuracy(y_test, y_pred, name)
            for i, name in enumerate(target_names)
        }
        metrics.append(
            {
                "accuracy": acc,
                "precision": pre,
                "f1": f1,
                "recall": rec,
                "accuracy_by_class": acc_by_class,
                "silhouette": sil,
                "batch_size": end_idx-start_idx,
                "cfm": cfm
            }
        )

    accuracy_matrix = np.vstack(
        [
            np.array(
                [
                    m["accuracy_by_class"][str(i)]
                    for i in target_names
                ]
            ).reshape((1, -1))
            for m in metrics
        ]
    )
    cfm_matrix = np.vstack(
        [
            m["cfm"]
            for m in metrics
        ]
    )
    accuracy_array = np.array([m["accuracy"] for m in metrics])
    precision_array = np.array([m["precision"] for m in metrics])
    f1_array = np.array([m["f1"] for m in metrics])
    recall_array = np.array([m["recall"] for m in metrics])
    sil_array = np.array([m["silhouette"] for m in metrics])
    batch_array = np.array([m["batch_size"] for m in metrics])

    meta = {
        "accuracy_by_class_iterative": accuracy_matrix.reshape((-1,)),
        "cfm_iterative": cfm_matrix.reshape((-1,)),
        "accuracy_iterative": accuracy_array,
        "precision_iterative": precision_array,
        "f1_iterative": f1_array,
        "recall_iterative": recall_array,
        "silhouette_iterative": sil_array,
        "target_names": target_names,
        "batch_size": batch_array
    }
    return cls, meta


# Function to run all experiments
def run_experiments(
        random_states=[42],  # Random seeds for reproducibility
        shuffles=[False, True, "semi"]  # Whether to shuffle the data
):
    # Total number of experiments to compute progress
    n_experiments = len(random_states) * len(shuffles) * len(models)
    results = []  # List to store results

    # Iterate through all experiment configurations
    for model, X_train, y_train, X_test, y_test, meta in tqdm(experiment_loader(
            random_states,
            shuffles
    ), total=n_experiments):
        if "ART" in meta["model"]:
            # Initialize SimpleARTMAP with the given model
            cls = SimpleARTMAP(model["model"](**model["params"]))
            cls.module_a.d_max_ = np.array(256*[1.0])  # Set maximum bounds
            cls.module_a.d_min_ = np.array(256*[0.0])  # Set minimum bounds

            # Prepare data for the model
            X_train_local = cls.prepare_data(X_train)
            X_test_local = cls.prepare_data(X_test)

            # Train the model and make predictions
            cls, step_meta = step_train(cls, X_train_local, y_train,
                                        X_test_local, y_test, match_tracking="MT+",
                                        epsilon=1e-6)
        else:
            cls = model["model"](**model["params"])
            X_train_local, _, _ = normalize(X_train, np.array(256 * [1.0]),
                                            np.array(256 * [
                                                0.0]))
            X_test_local, _, _ = normalize(X_test, np.array(256 * [1.0]),
                                           np.array(256 * [
                                               0.0]))
            target_names = sorted(np.unique(y_test).tolist())

            if "CNN" in meta["model"]:
                print(meta["model"])
                X_train_local = X_train_local.reshape((-1, 16, 16, 1))
                X_test_local = X_test_local.reshape((-1, 16, 16, 1))

                cls, step_meta = step_train(cls, X_train_local, y_train, X_test_local,
                                            y_test, classes=target_names)
            else:
                cls, step_meta = step_train(cls, X_train_local, y_train, X_test_local,
                                            y_test, classes=target_names)

        y_pred = cls.predict(X_test_local)

        meta.update(step_meta)

        # Compute evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")

        # Update metadata with metrics and append to results
        meta.update({"accuracy": acc, "precision": pre, "f1": f1, "recall": rec})
        results.append(dict(meta))

        # Save results to a file in Parquet format
        df = pd.DataFrame(results)
        df.to_parquet("usps.parquet")

# Entry point of the script
if __name__ == "__main__":
    run_experiments()