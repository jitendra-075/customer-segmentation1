import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide"
)

st.title("👥 Customer Segmentation Dashboard")
st.write("Enter customer details to predict their customer segment.")

# Load trained model and scaler
kmeans = joblib.load("customer_segmentation_kmeans.pkl")
scaler = joblib.load("customer_segmentation_scaler.pkl")

features = [
    "Income", "Recency", "MntWines", "MntFruits",
    "MntMeatProducts", "MntFishProducts", "MntSweetProducts",
    "MntGoldProds", "NumWebPurchases", "NumCatalogPurchases",
    "NumStorePurchases", "NumWebVisitsMonth", "Total_Spending",
    "Total_Purchases", "Total_Campaign_Acceptance"
]

st.subheader("Enter Customer Details")

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        income = st.number_input("Annual Income", min_value=0, value=50000)
        recency = st.number_input("Recency (days)", min_value=0, value=20)
        wines = st.number_input("Wine Spending", min_value=0, value=300)
        fruits = st.number_input("Fruit Spending", min_value=0, value=30)
        meat = st.number_input("Meat Spending", min_value=0, value=200)

    with col2:
        fish = st.number_input("Fish Spending", min_value=0, value=40)
        sweets = st.number_input("Sweet Spending", min_value=0, value=25)
        gold = st.number_input("Gold Products Spending", min_value=0, value=50)
        web_purchases = st.number_input("Web Purchases", min_value=0, value=4)
        catalog_purchases = st.number_input("Catalog Purchases", min_value=0, value=3)

    with col3:
        store_purchases = st.number_input("Store Purchases", min_value=0, value=6)
        web_visits = st.number_input("Website Visits per Month", min_value=0, value=5)
        campaign_acceptance = st.number_input(
            "Campaigns Accepted (0–5)", min_value=0, max_value=5, value=2
        )

    submitted = st.form_submit_button("Predict Customer Cluster")

if submitted:
    total_spending = wines + fruits + meat + fish + sweets + gold
    total_purchases = web_purchases + catalog_purchases + store_purchases

    customer = pd.DataFrame([{
        "Income": income,
        "Recency": recency,
        "MntWines": wines,
        "MntFruits": fruits,
        "MntMeatProducts": meat,
        "MntFishProducts": fish,
        "MntSweetProducts": sweets,
        "MntGoldProds": gold,
        "NumWebPurchases": web_purchases,
        "NumCatalogPurchases": catalog_purchases,
        "NumStorePurchases": store_purchases,
        "NumWebVisitsMonth": web_visits,
        "Total_Spending": total_spending,
        "Total_Purchases": total_purchases,
        "Total_Campaign_Acceptance": campaign_acceptance
    }})[features]

    scaled_customer = scaler.transform(customer)
    cluster = int(kmeans.predict(scaled_customer)[0])

    st.success(f"Predicted Customer Cluster: {cluster}")
    st.metric
