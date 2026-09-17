import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv(
    "Dataset/cleaned_data.csv",
    low_memory=False
)

print("Dataset loaded successfully!")
print(df.shape)

# Step 3: Customer purchase frequency

customer_orders = (
    df.groupby("CustomerID")["InvoiceNo"]
    .nunique()
    .sort_values(ascending=False)
)

print("\nTop 10 Customers by Number of Orders:")
print(customer_orders.head(10))

# Step 4: Repeat vs One-Time Customers

customer_order_counts = (
    df.groupby("CustomerID")["InvoiceNo"]
    .nunique()
)

one_time_customers = (customer_order_counts == 1).sum()
repeat_customers = (customer_order_counts > 1).sum()

total_customers = customer_order_counts.count()

print("\n" + "=" * 50)
print("CUSTOMER LOYALTY ANALYSIS")
print("=" * 50)

print(f"Total Customers: {total_customers:,}")
print(f"One-Time Customers: {one_time_customers:,}")
print(f"Repeat Customers: {repeat_customers:,}")

print(
    f"Repeat Customer Rate: "
    f"{(repeat_customers / total_customers) * 100:.2f}%"
)

# Step 5: Customer Spending

customer_spending = (
    df.groupby("CustomerID")["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
)

print("\n" + "=" * 50)
print("TOP CUSTOMERS BY SPENDING")
print("=" * 50)

print(customer_spending.head(10))

customer_spending.head(10).plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Top 10 Customers by Total Spending")
plt.xlabel("Customer ID")
plt.ylabel("Total Spending")

plt.tight_layout()
plt.show()

# Step 6: Average Spending per Customer

average_customer_spending = customer_spending.mean()

print("\n" + "=" * 50)
print("CUSTOMER SPENDING ANALYSIS")
print("=" * 50)

print(f"Average Customer Spending: £{average_customer_spending:,.2f}")

# Step 7: Customer Segmentation

def classify_customer(spending):
    if spending >= 1000:
        return "High Value"
    elif spending >= 300:
        return "Medium Value"
    else:
        return "Low Value"


customer_segment = customer_spending.apply(classify_customer)

print("\n" + "=" * 50)
print("CUSTOMER SEGMENTS")
print("=" * 50)

print(customer_segment.value_counts())

customer_segment.value_counts().plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Customer Segmentation")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# Step 9: Average Order Value by Customer

customer_revenue = df.groupby("CustomerID")["TotalAmount"].sum()

customer_orders = df.groupby("CustomerID")["InvoiceNo"].nunique()

customer_aov = (
    customer_revenue / customer_orders
).sort_values(ascending=False)

print("\n" + "=" * 50)
print("TOP 10 CUSTOMERS BY AVERAGE ORDER VALUE")
print("=" * 50)

print(customer_aov.head(10))

customer_aov.head(10).plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Top 10 Customers by Average Order Value")
plt.xlabel("Customer ID")
plt.ylabel("Average Order Value")

plt.tight_layout()
plt.show()

# Step 10: Save Customer Analysis

customer_analysis = pd.DataFrame({
    "Total_Spending": customer_spending,
    "Total_Orders": customer_orders,
    "Average_Order_Value": customer_aov,
    "Customer_Segment": customer_segment
})

customer_analysis.to_csv(
    "Dataset/customer_analysis.csv"
)

print("\nCustomer analysis saved successfully!")
print("File: Dataset/customer_analysis.csv")