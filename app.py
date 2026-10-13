
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Intelligence | Segmentation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- PROFESSIONAL THEME ----------------
st.markdown("""
<style>
    .stApp {
        background-color: #F5F7FB;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    [data-testid="stSidebar"] {
        background-color: #111827;
    }
    [data-testid="stSidebar"] * {
        color: #F9FAFB;
    }
    .hero {
        background: linear-gradient(120deg, #102A43, #145374);
        padding: 28px 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 22px;
    }
    .hero h1 {
        color: white;
        font-size: 30px;
        margin-bottom: 8px;
    }
    .hero p {
        color: #D9EAF5;
        margin-bottom: 0;
    }
    .metric-card {
        background: white;
        border: 1px solid #E3E8EF;
        border-radius: 13px;
        padding: 19px;
        min-height: 112px;
    }
    .metric-label {
        color: #64748B;
        font-size: 13px;
    }
    .metric-value {
        color: #102A43;
        font-size: 25px;
        font-weight: 700;
        margin-top: 7px;
    }
    .section-title {
        color: #102A43;
        font-size: 21px;
        font-weight: 700;
        margin-top: 18px;
        margin-bottom: 12px;
    }
    .result-card {
        background: white;
        border: 1px solid #D7E7E9;
        border-left: 5px solid #0D9488;
        border-radius: 12px;
        padding: 23px;
    }
    .small-note {
        color: #64748B;
        font-size: 12px;
    }
    div.stButton > button,
    div.stFormSubmitButton > button {
        background-color: #0D9488;
        color: white;
        border-radius: 8px;
        border: none;
        min-height: 43px;
        font-weight: 600;
    }
    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background-color: #0F766E;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD SAVED MODEL ----------------
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "customer_segmentation_kmeans.pkl")
    scaler_obj = joblib.load(BASE_DIR / "customer_segmentation_scaler.pkl")
    return model, scaler_obj

try:
    kmeans, scaler = load_artifacts()
except Exception as e:
    st.error("Could not load the saved model or scaler.")
    st.caption(
        "Check that both .pkl files are in the same repository folder as app.py."
    )
    st.stop()

# Keep the same feature names and order used during training.
features = [
    "Income",
    "Recency",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "Total_Spending",
    "Total_Purchases",
    "Total_Campaign_Acceptance"
]

if getattr(scaler, "n_features_in_", len(features)) != len(features):
    st.error("The scaler feature count does not match the dashboard.")
    st.stop()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 📊 Customer IQ")
    st.caption("CUSTOMER ANALYTICS PLATFORM")
    st.divider()

    st.markdown("### Navigation")
    page = st.radio(
        "Choose a page",
        ["Dashboard", "About the Model"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("### Model Information")
    st.write("**Algorithm:** K-Means")
    st.write(f"**Clusters:** {kmeans.n_clusters}")
    st.write(f"**Input features:** {len(features)}")

    st.divider()
    st.caption("Customer Segmentation Project")

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>Customer Intelligence Dashboard</h1>
    <p>Understand customer purchasing patterns and predict customer segments
    using machine learning.</p>
</div>
""", unsafe_allow_html=True)

if page == "Dashboard":

    # ---------------- KPI CARDS ----------------
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">CLUSTERING ALGORITHM</div>
            <div class="metric-value">K-Means</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">CUSTOMER SEGMENTS</div>
            <div class="metric-value">{kmeans.n_clusters}</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">MODEL INPUT FEATURES</div>
            <div class="metric-value">{len(features)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Customer Profile & Prediction</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter a customer's details below to predict their assigned cluster."
    )

    # ---------------- INPUT FORM ----------------
    with st.form("customer_profile"):
        st.markdown("#### 01 · Customer Information")
        c1, c2, c3 = st.columns(3)

        with c1:
            income = st.number_input(
                "Annual Income", min_value=0, value=50000, step=1000
            )
            recency = st.number_input(
                "Days Since Last Purchase", min_value=0, value=20
            )

        with c2:
            web_visits = st.number_input(
                "Website Visits / Month", min_value=0, value=5
            )
            campaign_acceptance = st.number_input(
                "Campaigns Accepted", min_value=0, max_value=5, value=2
            )

        with c3:
            web_purchases = st.number_input(
                "Web Purchases", min_value=0, value=4
            )
            catalog_purchases = st.number_input(
                "Catalog Purchases", min_value=0, value=3
            )

        st.divider()
        st.markdown("#### 02 · Spending by Product Category")

        s1, s2, s3 = st.columns(3)

        with s1:
            wines = st.number_input("Wine Spending", min_value=0, value=300)
            fruits = st.number_input("Fruit Spending", min_value=0, value=30)

        with s2:
            meat = st.number_input("Meat Spending", min_value=0, value=200)
            fish = st.number_input("Fish Spending", min_value=0, value=40)

        with s3:
            sweets = st.number_input("Sweet Spending", min_value=0, value=25)
            gold = st.number_input("Gold Product Spending", min_value=0, value=50)

        st.divider()
        st.markdown("#### 03 · Purchase Activity")

        p1, p2 = st.columns(2)

        with p1:
            store_purchases = st.number_input(
                "Store Purchases", min_value=0, value=6
            )

        with p2:
            st.caption(
                "Total purchases are calculated automatically from web, "
                "catalog and store purchases."
            )

        submitted = st.form_submit_button(
            "Predict Customer Segment", use_container_width=True
        )

    # ---------------- PREDICTION ----------------
    if submitted:
        total_spending = wines + fruits + meat + fish + sweets + gold
        total_purchases = (
            web_purchases + catalog_purchases + store_purchases
        )

        
        customer = pd.DataFrame({
            "Income": [income],
            "Recency": [recency],
            "MntWines": [wines],
            "MntFruits": [fruits],
            "MntMeatProducts": [meat],
            "MntFishProducts": [fish],
            "MntSweetProducts": [sweets],
            "MntGoldProds": [gold],
            "NumWebPurchases": [web_purchases],
            "NumCatalogPurchases": [catalog_purchases],
            "NumStorePurchases": [store_purchases],
            "NumWebVisitsMonth": [web_visits],
            "Total_Spending": [total_spending],
            "Total_Purchases": [total_purchases],
            "Total_Campaign_Acceptance": [campaign_acceptance]
        })

        customer = customer[features]


        try:
            scaled_customer = scaler.transform(customer)
            cluster = int(kmeans.predict(scaled_customer)[0])

            st.divider()
            st.markdown(
                '<div class="section-title">Prediction Results</div>',
                unsafe_allow_html=True
            )

            r1, r2 = st.columns([1, 1])

            with r1:
                st.markdown(f"""
                <div class="result-card">
                    <div class="metric-label">PREDICTED CUSTOMER SEGMENT</div>
                    <div class="metric-value">Cluster {cluster}</div>
                    <p class="small-note">
                        Assigned by your trained K-Means model.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            with r2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">TOTAL SPENDING</div>
                    <div class="metric-value">{total_spending:,}</div>
                    <p class="small-note">Sum across six product categories</p>
                </div>
                """, unsafe_allow_html=True)

            q1, q2 = st.columns(2)

            with q1:
                st.metric("Total Purchases", f"{total_purchases:,}")

            with q2:
                st.metric("Campaigns Accepted", campaign_acceptance)

            st.markdown(
                '<div class="section-title">Spending Profile</div>',
                unsafe_allow_html=True
            )

            spending_data = pd.DataFrame({
                "Product Category": [
                    "Wines", "Fruits", "Meat",
                    "Fish", "Sweets", "Gold Products"
                ],
                "Spending": [wines, fruits, meat, fish, sweets, gold]
            })

            st.bar_chart(
                spending_data.set_index("Product Category"),
                y="Spending",
                color="#0D9488"
            )

            channel_data = pd.DataFrame({
                "Purchase Channel": ["Web", "Catalog", "Store"],
                "Purchases": [
                    web_purchases, catalog_purchases, store_purchases
                ]
            })

            st.markdown("#### Purchase Channels")
            st.bar_chart(
                channel_data.set_index("Purchase Channel"),
                y="Purchases",
                color="#3B82F6"
            )

            with st.expander("View customer data sent to the model"):
                st.dataframe(customer, use_container_width=True)

            st.caption(
                "Cluster numbers are model-generated groups, not predefined "
                "customer labels. Interpret each cluster using the cluster "
                "profiles from your training analysis."
            )

        except Exception as e:
            st.error("Prediction failed. Please check the model and scaler.")
            st.exception(e)

    else:
        st.info(
            "Enter the customer details and click **Predict Customer Segment** "
            "to view the prediction and charts."
        )

else:
    # ---------------- MODEL INFORMATION PAGE ----------------
    st.markdown("## About the Model")
    st.write(
        "This project uses K-Means, an unsupervised machine-learning "
        "algorithm, to group customers based on their income, spending, "
        "purchase activity, website visits and campaign acceptance."
    )

    st.markdown("### Model Configuration")

    info = pd.DataFrame({
        "Property": [
            "Algorithm",
            "Number of clusters",
            "Number of input features",
            "Scaling"
        ],
        "Value": [
            "K-Means",
            str(kmeans.n_clusters),
            str(len(features)),
            "Saved training scaler"
        ]
    })

    st.dataframe(info, hide_index=True, use_container_width=True)

    st.markdown("### Features Used")

    st.dataframe(
        pd.DataFrame({"Model Features": features}),
        hide_index=True,
        use_container_width=True
    )

    st.info(
        "Cluster IDs do not automatically indicate high-value or low-value "
        "customers. Use your training notebook's cluster profiles to assign "
        "business interpretations."
    )

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "Customer Intelligence Dashboard · Built with Python, "
    "Streamlit, Pandas and Scikit-learn"
)

