import altair as alt
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ design
INK, TEAL, MUTED, LINE, SURFACE = "#1B2A41", "#0F766E", "#5B6B7F", "#D9E1EA", "#F3F6F9"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="st-"], .stMarkdown, label, input {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3, .hero h1, .segment-name { font-family: 'Manrope', sans-serif; letter-spacing: -0.01em; }

#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 1200px; }

.hero {
    background: #1B2A41; color: #FFFFFF;
    padding: 1.8rem 2rem; border-radius: 10px; margin-bottom: 1.5rem;
}
.hero h1 { color: #FFFFFF; margin: 0 0 .35rem 0; font-size: 1.9rem; font-weight: 800; }
.hero p  { color: #C7D2E0; margin: 0; font-size: 1rem; }

.section-title { font-family: 'Manrope', sans-serif; font-weight: 700; font-size: 1.05rem; color: #1B2A41; margin-bottom: .1rem; }
.section-hint  { color: #5B6B7F; font-size: .85rem; margin-bottom: .6rem; }

div[data-testid="stForm"] { border: none; padding: 0; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-color: #D9E1EA; border-radius: 10px; }

.stFormSubmitButton button {
    background: #0F766E; color: #FFFFFF; border: none; border-radius: 8px;
    padding: .7rem 1.6rem; font-weight: 600; width: 100%;
}
.stFormSubmitButton button:hover { background: #0B5E58; color: #FFFFFF; }
.stFormSubmitButton button:focus-visible { outline: 3px solid #99D5CF; outline-offset: 2px; }

.segment-card { background: #F3F6F9; border: 1px solid #D9E1EA; border-left: 6px solid #0F766E;
                border-radius: 10px; padding: 1.3rem 1.5rem; }
.segment-label { color: #5B6B7F; font-size: .9rem; margin-bottom: .2rem; }
.segment-name  { color: #1B2A41; font-size: 2rem; font-weight: 800; margin: 0; }
.segment-desc  { color: #3B4A5E; margin-top: .4rem; }

div[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #D9E1EA; border-radius: 10px; padding: .9rem 1.1rem; }
div[data-testid="stMetricLabel"] p { color: #5B6B7F; }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ model
FEATURES = [
    "Income", "Recency", "MntWines", "MntFruits",
    "MntMeatProducts", "MntFishProducts", "MntSweetProducts",
    "MntGoldProds", "NumWebPurchases", "NumCatalogPurchases",
    "NumStorePurchases", "NumWebVisitsMonth", "Total_Spending",
    "Total_Purchases", "Total_Campaign_Acceptance",
]

# Give each cluster number a meaningful name once you've profiled your clusters.
# Example: 0: ("Loyal high spenders", "Frequent buyers with high basket value.")
CLUSTER_PROFILES: dict[int, tuple[str, str]] = {}


@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    model = joblib.load("customer_segmentation_kmeans.pkl")
    scaler = joblib.load("customer_segmentation_scaler.pkl")
    return model, scaler


try:
    kmeans, scaler = load_artifacts()
except FileNotFoundError as err:
    st.error(f"Model file not found: {err.filename}. Place the .pkl files next to this script.")
    st.stop()

# ------------------------------------------------------------------ presets
PRESETS = {
    "Typical customer": dict(income=50000, recency=20, wines=300, fruits=30, meat=200, fish=40,
                             sweets=25, gold=50, web=4, catalog=3, store=6, visits=5, campaigns=2),
    "High spender": dict(income=85000, recency=10, wines=900, fruits=80, meat=600, fish=120,
                         sweets=90, gold=150, web=6, catalog=7, store=9, visits=3, campaigns=3),
    "Low engagement": dict(income=25000, recency=75, wines=20, fruits=5, meat=25, fish=5,
                           sweets=5, gold=10, web=1, catalog=0, store=2, visits=7, campaigns=0),
}


def apply_preset():
    for key, value in PRESETS[st.session_state["preset"]].items():
        st.session_state[key] = value


for key, value in PRESETS["Typical customer"].items():
    st.session_state.setdefault(key, value)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### Quick start")
    st.selectbox("Load an example customer", list(PRESETS), key="preset", on_change=apply_preset)
    st.caption("Pick a profile to fill the form, then adjust any value.")
    st.divider()
    st.markdown("### About the model")
    st.write(f"**Algorithm:** K-Means  \n**Segments:** {kmeans.n_clusters}  \n**Input features:** {len(FEATURES)}")
    st.caption("Total spending and total purchases are calculated for you from the values entered.")

# ------------------------------------------------------------------ header
st.markdown(
    """
<div class="hero">
  <h1>Customer Segmentation</h1>
  <p>Enter a customer's spending and activity to find out which segment they belong to.</p>
</div>
""",
    unsafe_allow_html=True,
)


def section(title: str, hint: str):
    st.markdown(f'<div class="section-title">{title}</div><div class="section-hint">{hint}</div>',
                unsafe_allow_html=True)


# ------------------------------------------------------------------ form
with st.form("customer_form"):
    c1, c2, c3 = st.columns(3)

    with c1, st.container(border=True):
        section("Profile & engagement", "Who they are and how recently they engaged.")
        st.number_input("Annual income", min_value=0, step=1000, key="income")
        st.number_input("Days since last purchase", min_value=0, key="recency")
        st.number_input("Website visits per month", min_value=0, key="visits")
        st.number_input("Campaigns accepted", min_value=0, max_value=5, key="campaigns",
                        help="Number of marketing campaigns accepted, from 0 to 5.")

    with c2, st.container(border=True):
        section("Spending by category", "Amount spent in each product category.")
        st.number_input("Wine", min_value=0, key="wines")
        st.number_input("Meat", min_value=0, key="meat")
        st.number_input("Fish", min_value=0, key="fish")
        st.number_input("Fruit", min_value=0, key="fruits")
        st.number_input("Sweets", min_value=0, key="sweets")
        st.number_input("Gold products", min_value=0, key="gold")

    with c3, st.container(border=True):
        section("Purchases by channel", "Number of purchases made through each channel.")
        st.number_input("Web", min_value=0, key="web")
        st.number_input("Catalog", min_value=0, key="catalog")
        st.number_input("Store", min_value=0, key="store")

    submitted = st.form_submit_button("Predict segment")

# ------------------------------------------------------------------ predict
if submitted:
    s = st.session_state
    spending = {
        "Wine": s.wines, "Meat": s.meat, "Fish": s.fish,
        "Fruit": s.fruits, "Sweets": s.sweets, "Gold": s.gold,
    }
    total_spending = sum(spending.values())
    total_purchases = s.web + s.catalog + s.store

    customer = pd.DataFrame([{
        "Income": s.income,
        "Recency": s.recency,
        "MntWines": s.wines,
        "MntFruits": s.fruits,
        "MntMeatProducts": s.meat,
        "MntFishProducts": s.fish,
        "MntSweetProducts": s.sweets,
        "MntGoldProds": s.gold,
        "NumWebPurchases": s.web,
        "NumCatalogPurchases": s.catalog,
        "NumStorePurchases": s.store,
        "NumWebVisitsMonth": s.visits,
        "Total_Spending": total_spending,
        "Total_Purchases": total_purchases,
        "Total_Campaign_Acceptance": s.campaigns,
    }])[FEATURES]

    cluster = int(kmeans.predict(scaler.transform(customer))[0])
    st.session_state["result"] = dict(
        cluster=cluster, spending=spending,
        total_spending=total_spending, total_purchases=total_purchases,
    )

# ------------------------------------------------------------------ results
result = st.session_state.get("result")

if result:
    st.markdown("---")
    name, desc = CLUSTER_PROFILES.get(
        result["cluster"],
        (f"Segment {result['cluster']}", "Add a name and description for this cluster in CLUSTER_PROFILES."),
    )

    left, right = st.columns([1, 1.4], gap="large")

    with left:
        st.markdown(
            f"""
<div class="segment-card">
  <div class="segment-label">Predicted segment</div>
  <p class="segment-name">{name}</p>
  <div class="segment-desc">{desc}</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.write("")
        m1, m2 = st.columns(2)
        m1.metric("Total spending", f"{result['total_spending']:,}")
        m2.metric("Total purchases", f"{result['total_purchases']:,}")
        avg = result["total_spending"] / result["total_purchases"] if result["total_purchases"] else 0
        st.metric("Average spend per purchase", f"{avg:,.1f}")

    with right:
        section("Where the money goes", "Spending by product category.")
        chart_df = pd.DataFrame(
            {"Category": list(result["spending"]), "Amount": list(result["spending"].values())}
        )
        chart = (
            alt.Chart(chart_df)
            .mark_bar(color=TEAL, cornerRadiusEnd=4)
            .encode(
                x=alt.X("Amount:Q", title=None),
                y=alt.Y("Category:N", sort="-x", title=None),
                tooltip=["Category", "Amount"],
            )
            .properties(height=260)
            .configure_axis(labelColor=MUTED, gridColor=LINE, domain=False)
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(chart, use_container_width=True)
else:
    st.info("Fill in the customer details above and select **Predict segment** to see the result.")
