import pandas as pd

# -----------------------------------------
# LOAD DATA
# -----------------------------------------

df = pd.read_csv("data/processed/uber_cleaned.csv")


# -----------------------------------------
# 1. OVERALL BUSINESS IMPACT
# -----------------------------------------

total_bookings = len(df)

cancelled_bookings = df["is_cancelled"].sum()

cancellation_rate = (
    cancelled_bookings / total_bookings
) * 100

print("\n===================================")
print("OVERALL BUSINESS METRICS")
print("===================================")

print("Total bookings:", total_bookings)
print("Cancelled bookings:", cancelled_bookings)
print(
    "Cancellation rate:",
    round(cancellation_rate, 2),
    "%"
)


# -----------------------------------------
# 2. DRIVER VS CUSTOMER CANCELLATIONS
# -----------------------------------------

driver_cancelled = (
    df["booking_status"]
    == "Cancelled by Driver"
).sum()

customer_cancelled = (
    df["booking_status"]
    == "Cancelled by Customer"
).sum()

print("\n===================================")
print("CANCELLATION BREAKDOWN")
print("===================================")

print(
    "Driver cancellations:",
    driver_cancelled
)

print(
    "Customer cancellations:",
    customer_cancelled
)

print(
    "Driver share of cancellations:",
    round(
        driver_cancelled /
        cancelled_bookings * 100,
        2
    ),
    "%"
)


# -----------------------------------------
# 3. POTENTIAL RECOVERABLE BOOKINGS
# -----------------------------------------

print("\n===================================")
print("SCENARIO ANALYSIS")
print("===================================")

for reduction in [5, 10, 15, 20]:

    recovered = round(
        cancelled_bookings * reduction / 100
    )

    print(
        f"If cancellations fall by {reduction}%, "
        f"approximately {recovered} bookings "
        f"could be recovered."
    )


# -----------------------------------------
# 4. VTAT ANALYSIS
# -----------------------------------------

completed = df[
    df["booking_status"] == "Completed"
]

customer_cancelled_df = df[
    df["booking_status"]
    == "Cancelled by Customer"
]

completed_vtat = completed["avg_vtat"].mean()

customer_cancel_vtat = (
    customer_cancelled_df["avg_vtat"].mean()
)

print("\n===================================")
print("WAIT-TIME ANALYSIS")
print("===================================")

print(
    "Average VTAT for completed rides:",
    round(completed_vtat, 2),
    "minutes"
)

print(
    "Average VTAT for customer-cancelled rides:",
    round(customer_cancel_vtat, 2),
    "minutes"
)

print(
    "Difference:",
    round(
        customer_cancel_vtat -
        completed_vtat,
        2
    ),
    "minutes"
)


# -----------------------------------------
# 5. BUSINESS RECOMMENDATIONS
# -----------------------------------------

print("\n===================================")
print("BUSINESS RECOMMENDATIONS")
print("===================================")

print("""
1. Reduce customer waiting time:
   Focus on reducing high-VTAT bookings
   because long expected arrival times are
   strongly associated with cancellations.

2. Prioritize high-risk bookings:
   Use the ML model to identify bookings
   with high cancellation risk.

3. Improve driver-side operations:
   Driver cancellations represent the majority
   of cancellations, so operational interventions
   should target driver-side issues.

4. Monitor payment and trip characteristics:
   Ride distance, booking value, payment method,
   and trip-time variables were important
   predictive features.

5. Avoid over-focusing on vehicle type:
   Vehicle categories showed very similar
   cancellation rates in this dataset.
""")