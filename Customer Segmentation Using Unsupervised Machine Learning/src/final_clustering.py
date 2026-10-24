import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

FEATURE_FILE = "data/processed/clustering_features.csv"
CUSTOMER_FILE = "data/processed/customer_features.csv"

X = pd.read_csv(FEATURE_FILE)
customers = pd.read_csv(CUSTOMER_FILE)

models = {
    "KMeans": KMeans(
        n_clusters=2,
        random_state=42,
        n_init=20
    ),
    "Agglomerative": AgglomerativeClustering(
        n_clusters=2,
        linkage="ward"
    ),
    "DBSCAN": DBSCAN(
        eps=1.1,
        min_samples=5
    )
}

results = []
cluster_profiles = {}

for name, model in models.items():

    labels = model.fit_predict(X)

    # For DBSCAN, noise is labelled -1.
    # Internal clustering metrics are calculated
    # only on observations assigned to clusters.
    valid_mask = labels != -1

    X_valid = X.loc[valid_mask]
    labels_valid = labels[valid_mask]

    n_clusters = len(set(labels_valid))

    if n_clusters >= 2:

        silhouette = silhouette_score(
            X_valid,
            labels_valid
        )

        davies_bouldin = davies_bouldin_score(
            X_valid,
            labels_valid
        )

        calinski_harabasz = calinski_harabasz_score(
            X_valid,
            labels_valid
        )

    else:
        silhouette = None
        davies_bouldin = None
        calinski_harabasz = None

    results.append({
        "Model": name,
        "Clusters": n_clusters,
        "Noise": int((labels == -1).sum()),
        "Silhouette": silhouette,
        "Davies_Bouldin": davies_bouldin,
        "Calinski_Harabasz": calinski_harabasz
    })

    # Attach labels to original customer features
    model_customers = customers.copy()
    model_customers["Cluster"] = labels

    profile = (
        model_customers[model_customers["Cluster"] != -1]
        .groupby("Cluster")
        .agg(
            Customers=("CustomerID", "count"),
            Recency=("Recency", "mean"),
            Frequency=("Frequency", "mean"),
            Monetary=("Monetary", "mean"),
            TotalQuantity=("TotalQuantity", "mean"),
            ProductDiversity=("ProductDiversity", "mean"),
            AverageOrderValue=("AverageOrderValue", "mean")
        )
        .reset_index()
    )

    cluster_profiles[name] = profile

    model_customers.to_csv(
        f"outputs/tables/{name.lower()}_customer_clusters.csv",
        index=False
    )

results_df = pd.DataFrame(results)

results_df.to_csv(
    "outputs/tables/final_model_comparison.csv",
    index=False
)

print("\n--- FINAL MODEL COMPARISON ---")
print(results_df.round(4).to_string(index=False))

for name, profile in cluster_profiles.items():

    print(f"\n--- {name.upper()} CUSTOMER PROFILES ---")
    print(profile.round(2).to_string(index=False))