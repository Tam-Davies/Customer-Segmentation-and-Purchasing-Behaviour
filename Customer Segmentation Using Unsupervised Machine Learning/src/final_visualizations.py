import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path


FEATURE_FILE = "data/processed/clustering_features.csv"
CLUSTER_FILE = "outputs/tables/kmeans_customer_clusters.csv"
OUTPUT_DIR = Path("outputs/figures")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load final K-Means results
features = pd.read_csv(FEATURE_FILE)
customers = pd.read_csv(CLUSTER_FILE)


# --------------------------------------------------
# 1. PCA plot of final K-Means clusters
# --------------------------------------------------

pca = PCA(n_components=2)
X_pca = pca.fit_transform(features)

plt.figure(figsize=(9, 6))

for cluster in sorted(customers["Cluster"].unique()):

    mask = customers["Cluster"] == cluster

    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        s=18,
        alpha=0.55,
        label=f"Cluster {cluster}"
    )

plt.xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)"
)

plt.ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)"
)

plt.title("Final K-Means Customer Segmentation")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "final_kmeans_pca_clusters.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 2. Cluster size distribution
# --------------------------------------------------

profiles = (
    customers
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

plt.figure(figsize=(8, 5))

plt.bar(
    profiles["Cluster"].astype(str),
    profiles["Customers"]
)

plt.xlabel("K-Means Cluster")
plt.ylabel("Number of Customers")
plt.title("Customer Distribution Across Final K-Means Clusters")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "kmeans_cluster_sizes.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 3. Customer profile comparison
# --------------------------------------------------

profile_metrics = [
    "Recency",
    "Frequency",
    "Monetary",
    "TotalQuantity",
    "ProductDiversity",
    "AverageOrderValue"
]

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 9)
)

axes = axes.flatten()

for ax, metric in zip(axes, profile_metrics):

    ax.bar(
        profiles["Cluster"].astype(str),
        profiles[metric]
    )

    ax.set_title(metric)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Mean value")

plt.suptitle(
    "K-Means Customer Profile Comparison",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "kmeans_customer_profile_comparison.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 4. Standardised customer profile comparison
# --------------------------------------------------

profile_values = profiles[profile_metrics]

profile_z = (
    profile_values - profile_values.mean()
) / profile_values.std()

plt.figure(figsize=(10, 4.5))

plt.imshow(
    profile_z.values,
    aspect="auto"
)

plt.xticks(
    range(len(profile_metrics)),
    profile_metrics,
    rotation=30,
    ha="right"
)

plt.yticks(
    range(len(profiles)),
    [f"Cluster {c}" for c in profiles["Cluster"]]
)

plt.colorbar(label="Standardized mean")

plt.title(
    "Relative Customer Profile by K-Means Cluster"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "kmeans_profile_heatmap.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# Display actual results
# --------------------------------------------------

print("\n--- FINAL K-MEANS CUSTOMER PROFILES ---")
print(
    profiles.round(2).to_string(index=False)
)

print("\n--- FIGURES CREATED ---")

for file in [
    "final_kmeans_pca_clusters.png",
    "kmeans_cluster_sizes.png",
    "kmeans_customer_profile_comparison.png",
    "kmeans_profile_heatmap.png"
]:
    print(OUTPUT_DIR / file)