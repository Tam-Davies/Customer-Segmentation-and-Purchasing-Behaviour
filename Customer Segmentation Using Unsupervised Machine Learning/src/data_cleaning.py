import pandas as pd

INPUT_FILE = "data/raw/retail.xlsx"
OUTPUT_FILE = "data/processed/clean_retail.csv"

df = pd.read_excel(INPUT_FILE)

print("Original shape:", df.shape)

# Remove exact duplicate transactions
df = df.drop_duplicates()

# Keep transactions linked to a customer
df = df.dropna(subset=["CustomerID"])

# Remove cancelled invoices
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

# Keep valid purchase quantities and prices
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]

# Calculate transaction value
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

# Save cleaned transaction-level data
df.to_csv(OUTPUT_FILE, index=False)

print("Cleaned shape:", df.shape)
print("Unique customers:", df["CustomerID"].nunique())
print("Date range:", df["InvoiceDate"].min(), "to", df["InvoiceDate"].max())
print("Total sales value:", round(df["TotalAmount"].sum(), 2))
print("\nMissing values after cleaning:")
print(df.isnull().sum())