import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

INPUT_FILE = "data/processed/clustering_features.csv"
OUTPUT_FILE = "outputs/tables/agglomerative_selection.csv"

X = pd.read_csv(INPUT_FILE)

results = []

for k in range(2, 11):

    model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"
    )

    labels = model.fit_predict(X)

    results.append({
        "K": k,
        "Silhouette": silhouette_score(X, labels),
        "Davies_Bouldin": davies_bouldin_score(X, labels),
        "Calinski_Harabasz": calinski_harabasz_score(X, labels)
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n--- AGGLOMERATIVE CLUSTERING SELECTION ---")
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