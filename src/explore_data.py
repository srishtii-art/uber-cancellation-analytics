import pandas as pd

# Load the Uber dataset
df = pd.read_csv("data/ncr_ride_bookings.csv")

# Show first 5 rows
print("\n--- FIRST 5 ROWS ---")
print(df.head())

# Show number of rows and columns
print("\n--- DATASET SIZE ---")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Show column names
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

# Show information about each column
print("\n--- DATA INFORMATION ---")
print(df.info())

# Count missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())