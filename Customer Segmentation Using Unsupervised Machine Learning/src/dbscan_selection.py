import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

INPUT_FILE = "data/processed/clustering_features.csv"
FIGURE_FILE = "outputs/figures/dbscan_k_distance.png"

X = pd.read_csv(INPUT_FILE)

# DBSCAN needs a neighbourhood-distance estimate.
# Use 5 neighbours because the feature space contains six variables.
neighbors = NearestNeighbors(n_neighbors=5)
neighbors.fit(X)

distances, _ = neighbors.kneighbors(X)

# Distance to the 5th nearest neighbour
k_distances = np.sort(distances[:, -1])

plt.figure(figsize=(9, 6))
plt.plot(k_distances)

plt.xlabel("Customers sorted by distance")
plt.ylabel("Distance to 5th nearest neighbour")
plt.title("DBSCAN k-Distance Plot")
plt.tight_layout()

plt.savefig(FIGURE_FILE, dpi=300)
plt.show()

print("\n--- K-DISTANCE SUMMARY ---")
print("Minimum:", round(k_distances.min(), 4))
print("25th percentile:", round(np.percentile(k_distances, 25), 4))
print("Median:", round(np.median(k_distances), 4))
print("75th percentile:", round(np.percentile(k_distances, 75), 4))
print("90th percentile:", round(np.percentile(k_distances, 90), 4))
print("95th percentile:", round(np.percentile(k_distances, 95), 4))
print("Maximum:", round(k_distances.max(), 4))