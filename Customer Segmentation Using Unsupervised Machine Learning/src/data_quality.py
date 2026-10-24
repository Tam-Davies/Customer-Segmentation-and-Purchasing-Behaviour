import pandas as pd

file_path = "data/raw/retail.xlsx"

df = pd.read_excel(file_path)

print("\n--- MISSING CUSTOMER IDs ---")
print(df["CustomerID"].isna().sum())

print("\n--- CANCELLATION INVOICES ---")
print(df["InvoiceNo"].astype(str).str.startswith("C").sum())

print("\n--- NEGATIVE QUANTITIES ---")
print((df["Quantity"] < 0).sum())

print("\n--- ZERO OR NEGATIVE UNIT PRICES ---")
print((df["UnitPrice"] <= 0).sum())

print("\n--- MISSING DESCRIPTIONS ---")
print(df["Description"].isna().sum())

print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

print("\n--- COUNTRIES ---")
print(df["Country"].value_counts())