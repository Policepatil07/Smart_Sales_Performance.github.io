import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("sales_data.csv")
df.head()

df.drop_duplicates(inplace=True)
df["Date"] = pd.to_datetime(df["Date"])
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df.dropna(inplace=True)

df["Revenue"] = df["Price"] * df["Quantity"]
df["Month"] = df["Date"].dt.month
df["Quarter"] = df["Date"].dt.quarter
df["Year"] = df["Date"].dt.year

df.describe()
df.groupby("Product")["Revenue"].sum().sort_values(ascending=False)
df.groupby("Region")["Revenue"].sum()

df.groupby("Month")["Revenue"].sum().plot(kind="bar")
plt.title("Monthly Revenue")
plt.show()

monthly = df.groupby("Month")["Revenue"].sum()

daily = df.groupby("Date")["Revenue"].sum()
daily.plot()

df["DayIndex"] = (df["Date"] - df["Date"].min()).dt.days
X = df[["DayIndex"]]
y = df["Revenue"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

st.title("Sales Dashboard")
st.line_chart(df.groupby("Date")["Revenue"].sum())
st.bar_chart(df.groupby("Product")["Revenue"].sum())


#Bar Chart → Revenue by Region
region_sales = df.groupby("Region")["Revenue"].sum().sort_values()

region_sales.plot(kind="bar", color="skyblue")
plt.title("Revenue by Region")
plt.xlabel("Region")
plt.ylabel("Revenue")
plt.xticks(rotation=0)
plt.show()

#Line Chart → Monthly Sales Trend
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.to_period("M")

monthly_sales = df.groupby("Month")["Revenue"].sum()

monthly_sales.plot(kind="line", marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.grid()
plt.show()

#Heatmap → Product vs Region Sales

pivot = df.pivot_table(
    index="Product",
    columns="Region",
    values="Revenue",
    aggfunc="sum",
    fill_value=0
)

plt.figure(figsize=(8,5))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Product vs Region Sales Heatmap")
plt.show()

#Pie Chart → Category Contribution
category_sales = df.groupby("Category")["Revenue"].sum()

plt.figure(figsize=(6,6))
plt.pie(category_sales, labels=category_sales.index, autopct="%1.1f%%")
plt.title("Category Contribution to Sales")
plt.show()

#Forecast Plot → Actual vs Predicted Sales

plt.figure(figsize=(10,5))

plt.plot(y_test.values, label="Actual Sales", marker="o")
plt.plot(predictions, label="Predicted Sales", marker="x")

plt.title("Actual vs Predicted Sales")
plt.xlabel("Time")
plt.ylabel("Revenue")
plt.legend()
plt.show()
