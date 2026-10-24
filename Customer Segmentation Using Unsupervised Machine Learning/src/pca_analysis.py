import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

INPUT_FILE = "data/processed/clustering_features.csv"
OUTPUT_FILE = "outputs/figures/pca_customer_space.png"

df = pd.read_csv(INPUT_FILE)

pca = PCA(n_components=2)
components = pca.fit_transform(df)

pca_df = pd.DataFrame(
    components,
    columns=["PC1", "PC2"]
)

print("\n--- PCA EXPLAINED VARIANCE ---")
print("PC1:", round(pca.explained_variance_ratio_[0], 4))
print("PC2:", round(pca.explained_variance_ratio_[1], 4))

print(
    "Total explained variance:",
    round(pca.explained_variance_ratio_.sum(), 4)
)

plt.figure(figsize=(9, 6))
plt.scatter(
    pca_df["PC1"],
    pca_df["PC2"],
    alpha=0.5,
    s=15
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customer Behaviour in Principal Component Space")
plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=300)
plt.show()