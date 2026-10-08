import pandas as pd
import os

# -----------------------------
# 1. LOAD RAW DATA
# -----------------------------

input_file = "data/ncr_ride_bookings.csv"

df = pd.read_csv(input_file)

print("Original dataset shape:", df.shape)


# -----------------------------
# 2. CLEAN COLUMN NAMES
# -----------------------------

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(r"[^\w]", "", regex=True)
)

print("\nCleaned column names:")
print(df.columns.tolist())


# -----------------------------
# 3. REMOVE DUPLICATES
# -----------------------------

duplicates = df.duplicated().sum()

print("\nDuplicate rows:", duplicates)

df = df.drop_duplicates()


# -----------------------------
# 4. CHECK MISSING VALUES
# -----------------------------

print("\nMissing values:")
print(df.isnull().sum())


# -----------------------------
# 5. CONVERT NUMERIC COLUMNS
# -----------------------------

numeric_columns = [
    "booking_value",
    "ride_distance",
    "driver_ratings",
    "customer_rating",
    "avg_vtat",
    "avg_ctat"
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")


# -----------------------------
# 6. CREATE CANCELLATION FLAG
# -----------------------------

if "booking_status" in df.columns:

    df["is_cancelled"] = (
        df["booking_status"]
        .astype(str)
        .str.lower()
        .str.contains("cancel")
        .astype(int)
    )

    print("\nCancellation distribution:")
    print(df["is_cancelled"].value_counts())


# -----------------------------
# 7. CREATE OUTPUT FOLDER
# -----------------------------

os.makedirs("data/processed", exist_ok=True)


# -----------------------------
# 8. SAVE CLEANED DATA
# -----------------------------

output_file = "data/processed/uber_cleaned.csv"

df.to_csv(output_file, index=False)

print("\nCleaned dataset saved to:")
print(output_file)

print("\nFinal dataset shape:", df.shape)