import pandas as pd

INPUT_FILE = "data/processed/clean_retail.csv"
OUTPUT_FILE = "data/processed/customer_features.csv"

df = pd.read_csv(INPUT_FILE, parse_dates=["InvoiceDate"])

# Reference date: one day after the final transaction
reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

customer_features = (
    df.groupby("CustomerID")
    .agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalAmount", "sum"),
        TotalQuantity=("Quantity", "sum"),
        ProductDiversity=("StockCode", "nunique")
    )
    .reset_index()
)

# Average spending per order
order_values = (
    df.groupby(["CustomerID", "InvoiceNo"])["TotalAmount"]
    .sum()
    .groupby("CustomerID")
    .mean()
    .rename("AverageOrderValue")
)

customer_features = customer_features.merge(
    order_values,
    on="CustomerID",
    how="left"
)

customer_features.to_csv(OUTPUT_FILE, index=False)

print("\n--- CUSTOMER FEATURE DATASET ---")
print("Shape:", customer_features.shape)

print("\n--- FEATURES ---")
print(customer_features.columns.tolist())

print("\n--- FIRST 5 CUSTOMERS ---")
print(customer_features.head())

print("\n--- SUMMARY ---")
print(customer_features.describe())