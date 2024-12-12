from artlib import FuzzyART, HypersphereART, GaussianART, SimpleARTMAP
from Data import generate_overlap_data
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, \
    silhouette_score, davies_bouldin_score
import pandas as pd
from tqdm import tqdm
import numpy as np


models = [
    {"model": FuzzyART, "params":{"rho":0.0, "alpha": 1e-10, "beta": 1.0}},
    {"model": HypersphereART, "params":{"rho":0.0, "alpha": 1e-10, "r_hat":
        0.5*np.sqrt(2), "beta": 1.0}},
    {"model": GaussianART, "params":{"rho":0.0, "alpha": 1e-10, "sigma_init":
        np.array([0.5, 0.5])}
     },
]

def data_loader(
        n_per_cluster = 200,
        sizes = [0.1, 0.2, 0.3, 0.4],
        spacings = [0.1, 0.2, 0.3, 0.4, 0.5],
        random_states=[42],
        shuffles=[False, True]
):
    for size in sizes:
        for spacing in spacings:
            for random_state in random_states:
                for shuffle in shuffles:
                    X_train, y_train, X_test, y_test = generate_overlap_data(
                        n_per_cluster,
                        side_length=size,
                        spacing=spacing,
                        random_state=random_state,
                        shuffle=shuffle
                    )
                    db_score_train = davies_bouldin_score(X_train, y_train)
                    db_score_test = davies_bouldin_score(X_test, y_test)
                    sil_score_train = silhouette_score(X_train, y_train)
                    sil_score_test = silhouette_score(X_test, y_test)
                    meta = {
                        "size": size, "spacing": spacing,
                        "random_state": random_state, "shufle":shuffle,
                        "db_train": db_score_train, "db_test": db_score_test,
                        "sil_train": sil_score_train, "sil_test": sil_score_test
                    }

                    yield X_train, y_train, X_test, y_test, meta

def experiment_loader(
        n_per_cluster = 200,
        sizes = [0.1, 0.2, 0.3, 0.4],
        spacings = [0.1, 0.2, 0.3, 0.4, 0.5],
        random_states=[42],
        shuffles=[False, True]
):
    for X_train, y_train, X_test, y_test, meta in data_loader(
            n_per_cluster,
            sizes,
            spacings,
            random_states,git
            shuffles
    ):
        for model in models:
            meta.update({"model": str(model.__class__)})
            yield model, X_train, y_train, X_test, y_test, meta


def run_experiments(
        n_per_cluster = 200,
        sizes = [0.1, 0.2, 0.3, 0.4],
        spacings = [0.1, 0.2, 0.3, 0.4, 0.5],
        random_states=[42],
        shuffles=[False, True]
):
    n_experiments = len(sizes)*len(spacings)*len(random_states)*len(shuffles)*len(models)
    results = []
    for model, X_train, y_train, X_test, y_test, meta in tqdm(experiment_loader(
            n_per_cluster,
            sizes,
            spacings,
            random_states,
            shuffles
    ), total=n_experiments):
        cls = SimpleARTMAP(model["model"](**model["params"]))
        cls.module_a.d_max_ = np.array([1.0, 1.0])
        cls.module_a.d_min_ = np.array([0.0, 0.0])

        X_train_local = cls.prepare_data(X_train)
        X_test_local = cls.prepare_data(X_test)
        cls = cls.fit(X_train_local, y_train)
        y_pred = cls.predict(X_test_local)

        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")

        meta.update({"acc": acc, "pre": pre, "f1":f1, "rec": rec})
        results.append(dict(meta))

        df = pd.DataFrame(results)
        df.to_parquet("overlap.parquet")


if __name__ =="__main__":
    run_experiments()