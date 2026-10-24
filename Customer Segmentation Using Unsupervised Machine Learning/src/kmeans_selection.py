import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

INPUT_FILE = "data/processed/clustering_features.csv"
OUTPUT_FILE = "outputs/tables/kmeans_selection.csv"
FIGURE_FILE = "outputs/figures/kmeans_selection.png"

X = pd.read_csv(INPUT_FILE)

results = []

for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(X)

    results.append({
        "K": k,
        "Silhouette": silhouette_score(X, labels),
        "Davies_Bouldin": davies_bouldin_score(X, labels),
        "Calinski_Harabasz": calinski_harabasz_score(X, labels)
    })

results_df = pd.DataFrame(results)

results_df.to_csv(OUTPUT_FILE, index=False)

print("\n--- K-MEANS CLUSTER SELECTION ---")
print(results_df.round(4).to_string(index=False))

print("\n--- BEST SILHOUETTE ---")
print(
    results_df.loc[
        results_df["Silhouette"].idxmax()
    ]
)

print("\n--- BEST DAVIES-BOULDIN ---")
print(
    results_df.loc[
        results_df["Davies_Bouldin"].idxmin()
    ]
)

print("\n--- BEST CALINSKI-HARABASZ ---")
print(
    results_df.loc[
        results_df["Calinski_Harabasz"].idxmax()
    ]
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(results_df["K"], results_df["Silhouette"], marker="o")
axes[0].set_title("Silhouette Score")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Score")

axes[1].plot(results_df["K"], results_df["Davies_Bouldin"], marker="o")
axes[1].set_title("Davies-Bouldin Index")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Score")

axes[2].plot(
    results_df["K"],
    results_df["Calinski_Harabasz"],
    marker="o"
)
axes[2].set_title("Calinski-Harabasz Index")
axes[2].set_xlabel("Number of Clusters (K)")
axes[2].set_ylabel("Score")

plt.tight_layout()
plt.savefig(FIGURE_FILE, dpi=300)
plt.show()