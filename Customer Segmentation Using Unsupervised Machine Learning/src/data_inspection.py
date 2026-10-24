import pandas as pd

file_path = "data/raw/retail.xlsx"

df = pd.read_excel(file_path)

print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

print("\n--- BASIC NUMERICAL SUMMARY ---")
print(df.describe())

print("\n--- UNIQUE COUNTS ---")
for column in df.columns:
    print(f"{column}: {df[column].nunique()} unique values")