import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------------------
# 1. LOAD CLEANED DATA
# -----------------------------------------

df = pd.read_csv("data/processed/uber_cleaned.csv")

# Create visualization folder if it doesn't exist
os.makedirs("visualizations", exist_ok=True)


# -----------------------------------------
# 2. BOOKING STATUS DISTRIBUTION
# -----------------------------------------

status_counts = df["booking_status"].value_counts()

plt.figure(figsize=(10, 6))
status_counts.plot(kind="bar")

plt.title("Uber Booking Status Distribution")
plt.xlabel("Booking Status")
plt.ylabel("Number of Bookings")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("visualizations/booking_status_distribution.png")
plt.close()


# -----------------------------------------
# 3. DRIVER VS CUSTOMER CANCELLATIONS
# -----------------------------------------

cancelled = df[
    df["booking_status"].isin(
        ["Cancelled by Driver", "Cancelled by Customer"]
    )
]

cancellation_type = cancelled["booking_status"].value_counts()

plt.figure(figsize=(8, 6))
cancellation_type.plot(kind="bar")

plt.title("Driver vs Customer Cancellations")
plt.xlabel("Cancellation Type")
plt.ylabel("Number of Cancellations")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig("visualizations/driver_vs_customer_cancellations.png")
plt.close()


# -----------------------------------------
# 4. DRIVER CANCELLATION REASONS
# -----------------------------------------

driver_cancelled = df[
    df["booking_status"] == "Cancelled by Driver"
]

driver_reasons = (
    driver_cancelled["driver_cancellation_reason"]
    .value_counts()
)

plt.figure(figsize=(10, 6))
driver_reasons.plot(kind="bar")

plt.title("Reasons for Driver Cancellations")
plt.xlabel("Driver Cancellation Reason")
plt.ylabel("Number of Cancellations")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig("visualizations/driver_cancellation_reasons.png")
plt.close()


# -----------------------------------------
# 5. CANCELLATION RATE BY VTAT RANGE
# -----------------------------------------

def vtat_category(value):
    if pd.isna(value):
        return "Unknown"
    elif value < 5:
        return "<5 min"
    elif value < 10:
        return "5-10 min"
    elif value < 15:
        return "10-15 min"
    else:
        return "15+ min"


df["vtat_range"] = df["avg_vtat"].apply(vtat_category)

vtat_analysis = (
    df.groupby("vtat_range")
    .agg(
        total_bookings=("booking_status", "count"),
        cancelled_bookings=("is_cancelled", "sum")
    )
)

vtat_analysis["cancellation_rate"] = (
    vtat_analysis["cancelled_bookings"]
    / vtat_analysis["total_bookings"]
    * 100
)

# Put ranges in logical order
order = ["<5 min", "5-10 min", "10-15 min", "15+ min"]

vtat_analysis = vtat_analysis.reindex(order)

plt.figure(figsize=(9, 6))
vtat_analysis["cancellation_rate"].plot(kind="bar")

plt.title("Cancellation Rate by Vehicle Arrival Time")
plt.xlabel("Average Vehicle Arrival Time")
plt.ylabel("Cancellation Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("visualizations/cancellation_rate_by_vtat.png")
plt.close()


# -----------------------------------------
# 6. CANCELLATION RATE BY VEHICLE TYPE
# -----------------------------------------

vehicle_analysis = (
    df.groupby("vehicle_type")
    .agg(
        total_bookings=("booking_status", "count"),
        cancelled_bookings=("is_cancelled", "sum")
    )
)

vehicle_analysis["cancellation_rate"] = (
    vehicle_analysis["cancelled_bookings"]
    / vehicle_analysis["total_bookings"]
    * 100
)

vehicle_analysis = vehicle_analysis.sort_values(
    "cancellation_rate",
    ascending=False
)

plt.figure(figsize=(10, 6))
vehicle_analysis["cancellation_rate"].plot(kind="bar")

plt.title("Cancellation Rate by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("Cancellation Rate (%)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig("visualizations/cancellation_rate_by_vehicle.png")
plt.close()


print("===================================")
print("ALL GRAPHS CREATED SUCCESSFULLY!")
print("===================================")

print("\nGraphs saved inside:")
print("visualizations/")

print("\nFiles created:")
print("- booking_status_distribution.png")
print("- driver_vs_customer_cancellations.png")
print("- driver_cancellation_reasons.png")
print("- cancellation_rate_by_vtat.png")
print("- cancellation_rate_by_vehicle.png")
# -----------------------------------------
# 7. CANCELLATION RATE BY PICKUP LOCATION
# -----------------------------------------

location_analysis = (
    df.groupby("pickup_location")
    .agg(
        total_bookings=("booking_status", "count"),
        cancelled_bookings=("is_cancelled", "sum")
    )
)

location_analysis["cancellation_rate"] = (
    location_analysis["cancelled_bookings"]
    / location_analysis["total_bookings"]
    * 100
)

# Remove locations with very few bookings
location_analysis = location_analysis[
    location_analysis["total_bookings"] >= 100
]

# Select top 10 locations by cancellation rate
top_locations = location_analysis.sort_values(
    "cancellation_rate",
    ascending=False
).head(10)

plt.figure(figsize=(12, 6))

top_locations["cancellation_rate"].sort_values().plot(
    kind="barh"
)

plt.title("Top Pickup Locations by Cancellation Rate")
plt.xlabel("Cancellation Rate (%)")
plt.ylabel("Pickup Location")
plt.tight_layout()

plt.savefig(
    "visualizations/location_cancellation_rate.png"
)

plt.close()


print("\nLocation analysis completed!")
print(top_locations)