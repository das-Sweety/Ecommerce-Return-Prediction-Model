## EDA Preprocessing
##
## Cleans the raw order dataset, handles missing values, creates the target variable,
## engineers time, customer, product, and category features, and saves a cleaned dataset
## ready for analysis and modeling

import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("order_dataset.csv")

# Basic Cleaning
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.drop_duplicates()

numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Target Variable 
df["Is_Returned"] = (
    (df["Final Quantity"] < 0) |
    (df["Refunded Item Count"] < 0) |
    (df["Refunds"] < 0)
).astype(int)

df = df.sort_values("Date").reset_index(drop=True)

# Feature Engineering
df["Revenue_Per_Item"] = df["Total Revenue"] / (df["Purchased Item Count"] + 1e-5)
df["Discount_Rate"] = df["Price Reductions"] / (df["Total Revenue"] + 1e-5)
df["Tax_Rate"] = df["Sales Tax"] / (df["Final Revenue"] + 1e-5)
df["Has_Discount"] = (df["Price Reductions"] > 0).astype(int)
df["Net_Quantity"] = df["Purchased Item Count"] - df["Refunded Item Count"]

# Time Features
df["Order_Year"] = df["Date"].dt.year
df["Order_Month"] = df["Date"].dt.month
df["Order_DayOfWeek"] = df["Date"].dt.dayofweek
df["Order_Quarter"] = df["Date"].dt.quarter
df["Is_Weekend"] = (df["Order_DayOfWeek"] >= 5).astype(int)

# Customer
df["Customer_Order_Count"] = df.groupby("Buyer ID")["Transaction ID"].transform("count")

# Product
df["Product_Popularity"] = df["Item Code"].map(df["Item Code"].value_counts())


# Time-Aware Historical Features 
# CUSTOMER 
df["Customer_Return_Rate"] = (
    df.groupby("Buyer ID")["Is_Returned"]
      .apply(lambda x: x.shift().expanding().mean())
      .reset_index(level=0, drop=True)
)

# PRODUCT 
df["Product_Return_Rate"] = (
    df.groupby("Item Code")["Is_Returned"]
      .apply(lambda x: x.shift().expanding().mean())
      .reset_index(level=0, drop=True)
)

# CATEGORY 
df["Category_Return_Rate"] = (
    df.groupby("Category")["Is_Returned"]
      .apply(lambda x: x.shift().expanding().mean())
      .reset_index(level=0, drop=True)
)

# Replace NaN with 0
df[["Customer_Return_Rate", "Product_Return_Rate", "Category_Return_Rate"]] = \
    df[["Customer_Return_Rate", "Product_Return_Rate", "Category_Return_Rate"]].fillna(0)

# Save cleaned dataset
df.to_csv("cleaned_order_dataset.csv", index=False)
print("Saved cleaned dataset: cleaned_order_dataset.csv")
