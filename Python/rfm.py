import pandas as pd
import matplotlib.pyplot as plt

# STEP 1: LOAD DATA
df = pd.read_csv("Dataset/cleaned_data.csv", low_memory=False)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print("Dataset loaded successfully!")
print(df.shape)
print(df["InvoiceDate"].min())
print(df["InvoiceDate"].max())


# STEP 2: RECENCY
reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

recency = (
    df.groupby("CustomerID")["InvoiceDate"]
    .max()
    .apply(lambda x: (reference_date - x).days)
)

print("\n" + "=" * 50)
print("RECENCY ANALYSIS")
print("=" * 50)
print(recency.head(10))


# STEP 3: FREQUENCY
print("\nStarting Frequency calculation...")

frequency = df.groupby("CustomerID")["InvoiceNo"].nunique()

print("\n" + "=" * 50)
print("FREQUENCY ANALYSIS")
print("=" * 50)
print(frequency.head(10))

print("\nFrequency calculation completed!")

# STEP 4: MONETARY

monetary = (
    df.groupby("CustomerID")["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
)

print("\n" + "=" * 50)
print("MONETARY ANALYSIS")
print("=" * 50)

print(monetary.head(10))

# STEP 5: CREATE RFM TABLE

rfm = pd.DataFrame({
    "Recency": recency,
    "Frequency": frequency,
    "Monetary": monetary
})

print("\n" + "=" * 50)
print("RFM TABLE")
print("=" * 50)

print(rfm.head(10))

# STEP 6: RFM SCORING

rfm["R_Score"] = pd.qcut(
    rfm["Recency"],
    5,
    labels=[5, 4, 3, 2, 1]
)

rfm["F_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
)

rfm["M_Score"] = pd.qcut(
    rfm["Monetary"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
)

print("\n" + "=" * 50)
print("RFM SCORES")
print("=" * 50)

print(
    rfm[
        [
            "Recency",
            "Frequency",
            "Monetary",
            "R_Score",
            "F_Score",
            "M_Score"
        ]
    ].head(10)
)

# STEP 7: OVERALL RFM SCORE

rfm["RFM_Score"] = (
    rfm["R_Score"].astype(int)
    + rfm["F_Score"].astype(int)
    + rfm["M_Score"].astype(int)
)

print("\n" + "=" * 50)
print("OVERALL RFM SCORE")
print("=" * 50)

print(
    rfm[
        [
            "Recency",
            "Frequency",
            "Monetary",
            "R_Score",
            "F_Score",
            "M_Score",
            "RFM_Score"
        ]
    ].head(10)
)

# STEP 7: OVERALL RFM SCORE

rfm["RFM_Score"] = (
    rfm["R_Score"].astype(int)
    + rfm["F_Score"].astype(int)
    + rfm["M_Score"].astype(int)
)

print("\n" + "=" * 50)
print("OVERALL RFM SCORE")
print("=" * 50)

print(
    rfm[
        [
            "Recency",
            "Frequency",
            "Monetary",
            "R_Score",
            "F_Score",
            "M_Score",
            "RFM_Score"
        ]
    ].head(10)
)

# STEP 8: CUSTOMER SEGMENTS

def segment_customer(score):
    if score >= 13:
        return "Champions"
    elif score >= 10:
        return "Loyal Customers"
    elif score >= 7:
        return "Potential Loyalists"
    elif score >= 5:
        return "At Risk"
    else:
        return "Lost Customers"


rfm["Segment"] = rfm["RFM_Score"].apply(segment_customer)

print("\n" + "=" * 50)
print("CUSTOMER SEGMENTS")
print("=" * 50)

print(rfm["Segment"].value_counts())

rfm["Segment"].value_counts().plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("RFM Customer Segments")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# STEP 9: REVENUE BY RFM SEGMENT

segment_revenue = (
    rfm.groupby("Segment")["Monetary"]
    .sum()
    .sort_values(ascending=False)
)

print("\n" + "=" * 50)
print("REVENUE BY RFM SEGMENT")
print("=" * 50)

print(segment_revenue)

segment_revenue.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Revenue by RFM Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Total Revenue")

plt.tight_layout()
plt.show()

# STEP 10: SAVE RFM DATASET

rfm.to_csv(
    "Dataset/rfm_customer_segments.csv"
)

print("\n" + "=" * 50)
print("RFM ANALYSIS COMPLETED")
print("=" * 50)

print("RFM dataset saved successfully!")
print("File: Dataset/rfm_customer_segments.csv")