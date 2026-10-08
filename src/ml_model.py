import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# -----------------------------------------
# 1. LOAD DATA
# -----------------------------------------

df = pd.read_csv("data/processed/uber_cleaned.csv")

print("Dataset loaded!")
print("Rows:", len(df))


# -----------------------------------------
# 2. CREATE TARGET
# -----------------------------------------

# is_cancelled is our target:
# 1 = cancelled
# 0 = not cancelled

y = df["is_cancelled"]


# -----------------------------------------
# 3. SELECT PREDICTIVE FEATURES
# -----------------------------------------

features = [
    "vehicle_type",
    "pickup_location",
    "drop_location",
    "avg_vtat",
    "avg_ctat",
    "booking_value",
    "ride_distance",
    "driver_ratings",
    "customer_rating",
    "payment_method"
]

X = df[features]


# -----------------------------------------
# 4. IDENTIFY COLUMN TYPES
# -----------------------------------------

categorical_features = [
    "vehicle_type",
    "pickup_location",
    "drop_location",
    "payment_method"
]

numeric_features = [
    "avg_vtat",
    "avg_ctat",
    "booking_value",
    "ride_distance",
    "driver_ratings",
    "customer_rating"
]


# -----------------------------------------
# 5. HANDLE MISSING VALUES
# -----------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# -----------------------------------------
# 6. PREPROCESS DATA
# -----------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# -----------------------------------------
# 7. CREATE RANDOM FOREST MODEL
# -----------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)


# -----------------------------------------
# 8. CREATE COMPLETE PIPELINE
# -----------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# -----------------------------------------
# 9. SPLIT DATA
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# -----------------------------------------
# 10. TRAIN MODEL
# -----------------------------------------

print("\nTraining Random Forest model...")

pipeline.fit(X_train, y_train)

print("Model training completed!")


# -----------------------------------------
# 11. MAKE PREDICTIONS
# -----------------------------------------

y_pred = pipeline.predict(X_test)


# -----------------------------------------
# 12. EVALUATE MODEL
# -----------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("MODEL PERFORMANCE")
print("===================================")

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# -----------------------------------------
# 13. FEATURE IMPORTANCE
# -----------------------------------------

# Get the trained Random Forest model
trained_model = pipeline.named_steps["model"]

# Get feature names after preprocessing
feature_names = pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

# Get importance scores
importance = trained_model.feature_importances_

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
})

# Sort from most important to least important
feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
)

print("\n===================================")
print("TOP 20 FEATURE IMPORTANCES")
print("===================================")

print(
    feature_importance.head(20).to_string(index=False)
)
# -----------------------------------------
# 14. FEATURE IMPORTANCE GRAPH
# -----------------------------------------

import matplotlib.pyplot as plt

top_features = feature_importance.head(10).sort_values(
    "importance"
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.title("Top 10 Features Used for Cancellation Prediction")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    "visualizations/feature_importance.png"
)

plt.close()

print("\nFeature importance graph saved!")
print("visualizations/feature_importance.png")