from artlib import FuzzyART, HypersphereART, GaussianART, SimpleARTMAP, normalize
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, \
    silhouette_score, davies_bouldin_score, confusion_matrix
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import numpy as np
from MLPClassifier import CustomMLPClassifier as MLPClassifier
import json
import umap

MLP_PARAMS = {
    "hidden_layer_sizes": (10,),  # Increased capacity
    "activation": 'relu',
    "solver": 'sgd',
    "alpha": 0.0001,  # Reduced regularization
    "batch_size": 1,
    "learning_rate": 'constant',
    "learning_rate_init": 0.01,  # Increased learning rate
    "max_iter": 500,  # Allow more iterations
    "shuffle": False,
    "random_state": 42,
    "tol": 1e-7,
    "momentum": 0.0,
    "nesterovs_momentum": False,
    "n_iter_no_change": 200,
}




# List of models to evaluate with their respective hyperparameters
models = [
    {"model": MLPClassifier, "params": MLP_PARAMS},
    # {"model": CNNClassifier, "params": CNN_PARAMS},
    {"model": FuzzyART, "params": {"rho": 0.0, "alpha": 1e-10, "beta": 1.0}},
    # {"model": HypersphereART, "params": {"rho": 0.0, "alpha": 1e-10, "r_hat": 0.5 *
    #                                                                           np.sqrt(4), "beta": 1.0}},
    # {"model": GaussianART, "params": {"rho": 0.0, "alpha": 1e-10, "sigma_init":
    #     np.array(4*[0.5])}},
    {"model": FuzzyART, "params": {"rho": 1.0, "alpha": 1e-10, "beta": 1.0}},
    # {"model": HypersphereART, "params": {"rho": 1.0, "alpha": 1e-10, "r_hat": 0.5 * np.sqrt(4), "beta": 1.0}},
    # {"model": GaussianART, "params": {"rho": 1.0, "alpha": 1e-10, "sigma_init":np.array(4 * [0.5])}},

]

# Function to load and generate data for the experiments
def data_loader(
        random_states=[42],  # Random seeds for reproducibility
        shuffles=[False, True, "semi"],  # Whether to shuffle the data,
        use_umap=False,
):
    for random_state in random_states:
        for shuffle in shuffles:

            # Perform the train-test split
            iris = datasets.load_iris()
            reducer = umap.UMAP(n_components=4)
            if use_umap:
                X_umap = reducer.fit_transform(iris.data, y=iris.target)
            else:
                X_umap = iris.data
            X_train, X_test, y_train, y_test = train_test_split(
                X_umap, iris.target, test_size=0.3, random_state=random_state,
                stratify=iris.target
            )
            sorted_indices = np.argsort(y_train)
            X_train = X_train[sorted_indices]
            y_train = y_train[sorted_indices]

            sorted_indices = np.argsort(y_test)
            X_test = X_test[sorted_indices]
            y_test = y_test[sorted_indices]
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
        shuffles=[False, True, "semi"],  # Whether to shuffle the data
        use_umap=False,
):
    # Iterate over all data and combine it with the models
    for X_train, y_train, X_test, y_test, meta in data_loader(
            random_states,
            shuffles,
            use_umap
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

    for batch_num, (start_idx, end_idx) in enumerate(batch_idxs):
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
            sil = silhouette_score(X_train.reshape((-1, 4))[:end_idx], y_train[:end_idx])
        else:
            sil = np.nan

        acc_by_class = {
            str(name): class_accuracy(y_test, y_pred, name)
            for name in target_names
        }

        # Collect weights and biases for MLP or ART models
        weights, biases, shapes = [], [], []

        if cls.__class__.__name__ == "CustomMLPClassifier":
            # Store MLP layer-wise weights and biases as flat arrays
            for i in range(len(cls.coefs_)):
                weights.append(cls.coefs_[i].flatten().tolist())
                shapes.append(cls.coefs_[i].shape)
                biases.append(cls.intercepts_[i].tolist())

        else:
            # Store ART class-specific weights as flat arrays
            for i in range(len(cls.module_a.W)):
                weights.append(cls.module_a.W[i].flatten().tolist())
                shapes.append(cls.module_a.W[i].shape)

        metrics.append(
            {
                "accuracy": acc,
                "precision": pre,
                "f1": f1,
                "recall": rec,
                "accuracy_by_class": acc_by_class,
                "silhouette": sil,
                "batch_size": end_idx - start_idx,
                "cfm": cfm,
                "weights": weights,
                "biases": biases if biases else None,
                "shapes": shapes,
                "model_type": "MLP" if cls.__class__.__name__ == "CustomMLPClassifier" else "ART"
            }
        )

    # Collect per-batch metrics
    accuracy_matrix = np.vstack([np.array([m["accuracy_by_class"][str(i)] for i in target_names]) for m in metrics])
    cfm_matrix = np.vstack([m["cfm"] for m in metrics])
    accuracy_array = np.array([m["accuracy"] for m in metrics])
    precision_array = np.array([m["precision"] for m in metrics])
    f1_array = np.array([m["f1"] for m in metrics])
    recall_array = np.array([m["recall"] for m in metrics])
    sil_array = np.array([m["silhouette"] for m in metrics])
    batch_array = np.array([m["batch_size"] for m in metrics])

    meta = {
        "accuracy_by_class_iterative": accuracy_matrix.flatten(),
        "cfm_iterative": cfm_matrix.flatten(),
        "accuracy_iterative": accuracy_array,
        "precision_iterative": precision_array,
        "f1_iterative": f1_array,
        "recall_iterative": recall_array,
        "silhouette_iterative": sil_array,
        "target_names": target_names,
        "batch_size": batch_array,
        "weights_across_batches": [m["weights"] for m in metrics],
        "biases_across_batches": [m["biases"] for m in metrics] if any(m["biases"] for m in metrics) else None,
        "shapes_across_batches": [m["shapes"] for m in metrics],
        "model_type": metrics[0]["model_type"]
    }
    return cls, meta



# Function to run all experiments
def run_experiments(
        random_states=[42],  # Random seeds for reproducibility
        shuffles=[False, True, "semi"],  # Whether to shuffle the data
        use_umap=False
):
    # Total number of experiments to compute progress
    n_experiments = len(random_states) * len(shuffles) * len(models)
    results = []  # List to store results

    # Iterate through all experiment configurations
    # for model, X_train, y_train, X_test, y_test, meta in tqdm(experiment_loader(
    #         random_states,
    #         shuffles,
    #         use_umap
    # ), total=n_experiments):
    for model, X_train, y_train, X_test, y_test, meta in experiment_loader(
            random_states,
            shuffles,
            use_umap
    ):
        DMAX = np.max(np.concatenate([X_train, X_test], axis=0), axis=0)
        DMIN = np.min(np.concatenate([X_train, X_test], axis=0), axis=0)
        if "ART" in meta["model"]:
            # Initialize SimpleARTMAP with the given model
            cls = SimpleARTMAP(model["model"](**model["params"]))
            cls.module_a.d_max_ = DMAX  # Set maximum bounds
            cls.module_a.d_min_ = DMIN  # Set minimum bounds

            # Prepare data for the model
            X_train_local = cls.prepare_data(X_train)
            X_test_local = cls.prepare_data(X_test)

            # Train the model and make predictions
            cls, step_meta = step_train(cls, X_train_local, y_train,
                                        X_test_local, y_test, match_tracking="MT+",
                                        epsilon=1e-6)
        else:
            cls = model["model"](**model["params"])
            X_train_local, _, _ = normalize(X_train, DMAX, DMIN)
            X_test_local, _, _ = normalize(X_test, DMAX, DMIN)
            target_names = sorted(np.unique(y_test).tolist())

            cls, step_meta = step_train(cls, X_train_local, y_train, X_test_local,
                                            y_test, classes=target_names)

        y_pred = cls.predict(X_test_local)

        meta.update(step_meta)

        # Compute evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred, average="weighted",  zero_division=np.nan)
        f1 = f1_score(y_test, y_pred, average="weighted",  zero_division=np.nan)
        rec = recall_score(y_test, y_pred, average="weighted",  zero_division=np.nan)

        # Update metadata with metrics and append to results
        meta.update({"accuracy": acc, "precision": pre, "f1": f1, "recall": rec})
        results.append(dict(meta))

        # Save results to a file in Parquet format
        df = pd.DataFrame(results)
        if use_umap:
            df.to_parquet("iris_umap.parquet")
        else:
            df.to_parquet("iris.parquet")

# Entry point of the script
if __name__ == "__main__":
    run_experiments(shuffles=[False])
    run_experiments(shuffles=[False], use_umap=True)
