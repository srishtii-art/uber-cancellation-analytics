import pandas as pd
import sqlite3
import os

# Load cleaned Uber data
df = pd.read_csv("data/processed/uber_cleaned.csv")

# Create database folder
os.makedirs("data/database", exist_ok=True)

# Database location
database_path = "data/database/uber.db"

# Connect to SQLite database
connection = sqlite3.connect(database_path)

# Store the dataset as a SQL table
df.to_sql(
    "rides",
    connection,
    if_exists="replace",
    index=False
)

connection.close()

print("Database created successfully!")
print("Location:", database_path)
print("Table created: rides")
print("Rows:", len(df))