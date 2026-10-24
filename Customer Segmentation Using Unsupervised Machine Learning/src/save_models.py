import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN


# --------------------------------------------------
# Paths
# --------------------------------------------------

CUSTOMER_FILE = "data/processed/customer_features.csv"
MODEL_DIR = Path("outputs/models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load customer-level features
# --------------------------------------------------

df = pd.read_csv(CUSTOMER_FILE)

feature_columns = [
    "Recency",
    "Frequency",
    "Monetary",
    "TotalQuantity",
    "ProductDiversity",
    "AverageOrderValue"
]

X = df[feature_columns].copy()


# --------------------------------------------------
# Apply the same preprocessing used for clustering
# --------------------------------------------------

for column in feature_columns:
    X[column] = np.log1p(X[column])


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# Save scaler
joblib.dump(
    scaler,
    MODEL_DIR / "scaler.joblib"
)


# --------------------------------------------------
# PCA
# --------------------------------------------------

pca = PCA(n_components=2)

pca.fit(X_scaled)

joblib.dump(
    pca,
    MODEL_DIR / "pca_2d.joblib"
)


# --------------------------------------------------
# K-Means
# --------------------------------------------------

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=20
)

kmeans.fit(X_scaled)

joblib.dump(
    kmeans,
    MODEL_DIR / "kmeans_final.joblib"
)


# --------------------------------------------------
# Agglomerative Clustering
# --------------------------------------------------

agglomerative = AgglomerativeClustering(
    n_clusters=2,
    linkage="ward"
)

agglomerative.fit(X_scaled)

joblib.dump(
    agglomerative,
    MODEL_DIR / "agglomerative_final.joblib"
)


# --------------------------------------------------
# DBSCAN
# --------------------------------------------------

dbscan = DBSCAN(
    eps=1.1,
    min_samples=5
)

dbscan.fit(X_scaled)

joblib.dump(
    dbscan,
    MODEL_DIR / "dbscan_final.joblib"
)


# --------------------------------------------------
# Confirmation
# --------------------------------------------------

print("\n--- MODEL ARTIFACTS SAVED ---")

for file in sorted(MODEL_DIR.iterdir()):
    print(file)

print("\nK-Means clusters:", kmeans.n_clusters)
print("K-Means inertia:", round(kmeans.inertia_, 2))

print(
    "DBSCAN noise points:",
    int((dbscan.labels_ == -1).sum())
)