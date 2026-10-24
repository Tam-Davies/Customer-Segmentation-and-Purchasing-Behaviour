import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

INPUT_FILE = "data/processed/customer_features.csv"
OUTPUT_FILE = "data/processed/clustering_features.csv"

df = pd.read_csv(INPUT_FILE)

# Customer ID is an identifier, not a behavioural feature
feature_columns = [
    "Recency",
    "Frequency",
    "Monetary",
    "TotalQuantity",
    "ProductDiversity",
    "AverageOrderValue"
]

X = df[feature_columns].copy()

# Log transformation reduces the influence of extreme values
for column in feature_columns:
    X[column] = np.log1p(X[column])

# Standardise all features to the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scaled_df = pd.DataFrame(
    X_scaled,
    columns=feature_columns
)

scaled_df.to_csv(OUTPUT_FILE, index=False)

print("\n--- CLUSTERING FEATURES ---")
print("Shape:", scaled_df.shape)

print("\n--- FEATURE MEANS ---")
print(scaled_df.mean().round(4))

print("\n--- FEATURE STANDARD DEVIATIONS ---")
print(scaled_df.std().round(4))

print("\n--- FIRST 5 ROWS ---")
print(scaled_df.head())