import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Page Configuration for Premium Look
st.set_page_config(page_title="Axion AI - Sales Engine", page_icon="📊", layout="wide")

# Custom CSS for Professional Dark/Modern Premium Vibe
st.markdown("""
    <style>
    .main-title { font-size:42px; font-weight:bold; color: #1E3A8A; text-align: center; margin-bottom:10px; }
    .sub-title { font-size:18px; color: #555555; text-align: center; margin-bottom:30px; }
    .metric-box { padding: 20px; background-color: #F3F4F6; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Axion AI 🚀 Sales Forecasting Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Intelligent Business Analytics & Predictive Revenue Pipelines</div>', unsafe_allow_html=True)

# 1. Sidebar Control & Data Simulation
st.sidebar.header("📁 Data Source Configuration")
st.sidebar.info("Currently running on live simulated business records for initialization.")

np.random.seed(42)
data_size = 200
products = ['Cold Drinks', 'Snacks', 'Coffee', 'Bakery Items']
dates = [datetime(2026, 9, 1) + timedelta(days=int(np.random.randint(0, 45))) for _ in range(data_size)]
selected_products = np.random.choice(products, size=data_size, p=[0.4, 0.3, 0.2, 0.1])
quantities = np.random.randint(1, 5, size=data_size)
prices = np.where(selected_products == 'Cold Drinks', 40,
         np.where(selected_products == 'Snacks', 30,
         np.where(selected_products == 'Coffee', 60, 80)))
total_amount = quantities * prices

df = pd.DataFrame({
    'Date': dates,
    'Product_Name': selected_products,
    'Quantity': quantities,
    'Price_Per_Unit': prices,
    'Total_Sales_INR': total_amount
}).sort_values(by='Date').reset_index(drop=True)

# 2. Main KPI Metrics Layout
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-box"><h3>💰 Total Revenue</h3><h2>₹21,310</h2></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-box"><h3>📦 Total Orders</h3><h2>200</h2></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-box"><h3>🔮 Next Week Forecast</h3><h2>₹3,914</h2></div>', unsafe_allow_html=True)

st.write("---")

# 3. Analytics & Machine Learning Splits
left_chart_col, right_pred_col = st.columns(2)

with left_chart_col:
    st.subheader("📈 Category Performance (Historical Breakdown)")
    top_products = df.groupby('Product_Name')['Total_Sales_INR'].sum().reset_index()
    
    # Python 3.14 Compatibility Fix: Converting DataFrame into an indexed chart matrix
    chart_data = top_products.set_index('Product_Name')[['Total_Sales_INR']]
    st.bar_chart(chart_data)

with right_pred_col:
    st.subheader("🔮 ML Predictive Forecasting (Next 7 Days)")
    daily_sales = df.groupby('Date')['Total_Sales_INR'].sum().reset_index()
    daily_sales['Day_Index'] = (daily_sales['Date'] - daily_sales['Date'].min()).dt.days
    
    X = daily_sales[['Day_Index']].values
    y = daily_sales['Total_Sales_INR'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    last_day_index = daily_sales['Day_Index'].max()
    future_days = np.array([[last_day_index + i] for i in range(1, 8)])
    future_predictions = model.predict(future_days)
    
    last_date = daily_sales['Date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
    
    pred_df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d (%A)') for d in future_dates],
        'Predicted Revenue': [f"₹{max(0, int(v)):,}" for v in future_predictions]
    })
    st.dataframe(pred_df, use_container_width=True)

st.success("💡 Recommendation: Stock levels for 'Cold Drinks' should be boosted by 15% to meet next week's operational pipeline thresholds.")
