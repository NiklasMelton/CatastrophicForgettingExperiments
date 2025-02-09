import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_parquet("overlap.parquet")
data.sort_values(by="spacing", inplace=True)



n_groups = len(list(data.groupby(["size", "radius"])))

for model, model_group in data.groupby("model"):
    fig, ax = plt.subplots(n_groups, 2)
    for j, ((size, radius), size_group) in enumerate(model_group.groupby(["size", "radius"])):
        randomized = size_group[size_group["shuffle"]=="True"]
        ordered = size_group[size_group["shuffle"]=="False"]

        for i, df in enumerate([randomized, ordered]):
            ax[j, i].plot(df["spacing"], df["accuracy"], label="accuracy")
            ax[j, i].plot(df["spacing"], df["precision"], label="precision")
            ax[j, i].plot(df["spacing"], df["recall"], label="recall")
            ax[j, i].plot(df["spacing"], df["f1"], label="F1")
            ax[j, i].plot(df["spacing"], df["sil_test"], label="Silhouette")
            ax[j, i].legend()
            if i == 0:
                ax[j, i].set_title(f"Randomize Results, size={size}")
            else:
                ax[j, i].set_title(f"Ordered Results, size={size}")
    fig.suptitle(model)
plt.show()