import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

INPUT_FILE = "data/processed/clustering_features.csv"
OUTPUT_FILE = "outputs/tables/dbscan_evaluation.csv"

X = pd.read_csv(INPUT_FILE)

eps_values = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

results = []

for eps in eps_values:

    model = DBSCAN(
        eps=eps,
        min_samples=5
    )

    labels = model.fit_predict(X)

    # -1 represents DBSCAN noise
    noise_count = (labels == -1).sum()
    noise_percentage = noise_count / len(labels) * 100

    # Evaluate only the clustered observations
    clustered_mask = labels != -1
    clustered_X = X[clustered_mask]
    clustered_labels = labels[clustered_mask]

    n_clusters = len(set(clustered_labels))

    if n_clusters >= 2:
        silhouette = silhouette_score(
            clustered_X,
            clustered_labels
        )

        davies_bouldin = davies_bouldin_score(
            clustered_X,
            clustered_labels
        )

        calinski_harabasz = calinski_harabasz_score(
            clustered_X,
            clustered_labels
        )
    else:
        silhouette = None
        davies_bouldin = None
        calinski_harabasz = None

    results.append({
        "eps": eps,
        "clusters": n_clusters,
        "noise_count": noise_count,
        "noise_percentage": noise_percentage,
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n--- DBSCAN PARAMETER EVALUATION ---")
print(results_df.round(4).to_string(index=False))