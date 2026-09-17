import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Dataset/cleaned_data.csv", low_memory=False)

# Step 2: Top 10 countries
print(df["Country"].value_counts().head(10))

# Step 3: Create chart
df["Country"].value_counts().head(10).plot(kind="bar")

plt.title("Top 10 Countries")
plt.xlabel("Country")
plt.ylabel("Orders")

plt.show()

# Step 4: Top 10 Products
print("\nTop 10 Products:")
print(df["Description"].value_counts().head(10))

df["Description"].value_counts().head(10).plot(kind="bar")

plt.title("Top 10 Most Purchased Products")
plt.xlabel("Product")
plt.ylabel("Number of Purchases")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()

# Step 5: Revenue by Country

country_revenue = df.groupby("Country")["TotalAmount"].sum()
country_revenue = country_revenue.sort_values(ascending=False)

print("\nTop 10 Countries by Revenue:")
print(country_revenue.head(10))

country_revenue.head(10).plot(kind="bar")

plt.title("Top 10 Countries by Revenue")
plt.xlabel("Country")
plt.ylabel("Total Revenue")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 6: Business KPIs

total_revenue = df["TotalAmount"].sum()
total_quantity = df["Quantity"].sum()
total_orders = df["InvoiceNo"].nunique()
total_customers = df["CustomerID"].nunique()
average_order_value = total_revenue / total_orders

print("\n" + "=" * 40)
print("BUSINESS KPIs")
print("=" * 40)

print(f"Total Revenue: £{total_revenue:,.2f}")
print(f"Total Quantity Sold: {total_quantity:,}")
print(f"Total Orders: {total_orders:,}")
print(f"Total Customers: {total_customers:,}")
print(f"Average Order Value: £{average_order_value:,.2f}")

# Step 7: Monthly Sales Trend

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

monthly_sales = df.groupby(
    df["InvoiceDate"].dt.to_period("M")
)["TotalAmount"].sum()

print("\nMonthly Sales:")
print(monthly_sales)

monthly_sales.plot(kind="line", figsize=(10, 5))

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Step 8: Top Customers by Revenue

customer_revenue = (
    df.groupby("CustomerID")["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTop 10 Customers by Revenue:")
print(customer_revenue.head(10))

customer_revenue.head(10).plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Top 10 Customers by Revenue")
plt.xlabel("Customer ID")
plt.ylabel("Total Revenue")

plt.tight_layout()
plt.show()

# Step 9: Top Products by Revenue

product_revenue = (
    df.groupby("Description")["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTop 10 Products by Revenue:")
print(product_revenue.head(10))

product_revenue.head(10).plot(
    kind="bar",
    figsize=(12, 5)
)

plt.title("Top 10 Products by Revenue")
plt.xlabel("Product")
plt.ylabel("Revenue")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()

# Step 10: Correlation Analysis

correlation = df[
    ["Quantity", "UnitPrice", "CustomerID", "TotalAmount"]
].corr()

print("\nCorrelation Matrix:")
print(correlation)

import seaborn as sns

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Numerical Variables")
plt.tight_layout()
plt.show()

# Step 11: Final Business Insights

print("\n" + "=" * 60)
print("FINAL BUSINESS INSIGHTS")
print("=" * 60)

# Top country by number of orders
top_country = df["Country"].value_counts().idxmax()
top_country_orders = df["Country"].value_counts().max()

# Top product by number of purchases
top_product = df["Description"].value_counts().idxmax()
top_product_orders = df["Description"].value_counts().max()

# Highest revenue country
top_revenue_country = country_revenue.idxmax()
top_revenue = country_revenue.max()

# Highest revenue product
top_revenue_product = product_revenue.idxmax()
top_product_revenue = product_revenue.max()

print(f"1. Country with most orders: {top_country}")
print(f"   Number of orders: {top_country_orders:,}")

print(f"\n2. Most purchased product: {top_product}")
print(f"   Number of purchases: {top_product_orders:,}")

print(f"\n3. Country generating the most revenue: {top_revenue_country}")
print(f"   Revenue: £{top_revenue:,.2f}")

print(f"\n4. Product generating the most revenue: {top_revenue_product}")
print(f"   Revenue: £{top_product_revenue:,.2f}")

print(f"\n5. Total revenue: £{total_revenue:,.2f}")
print(f"6. Total orders: {total_orders:,}")
print(f"7. Total customers: {total_customers:,}")
print(f"8. Average order value: £{average_order_value:,.2f}")
    