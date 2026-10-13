import altair as alt
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================ design tokens
NAVY, ACCENT, AMBER = "#14213D", "#2F5BEA", "#E08A1E"
MUTED, LINE, BG = "#5B6B7F", "#DDE3EB", "#F4F6F9"
PALETTE = ["#2F5BEA", "#0F9D8A", "#E08A1E", "#C2410C", "#7C3AED", "#64748B"]

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
.stApp, .stApp p, .stApp label, .stApp input, .stApp li, .stApp td, .stApp th { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3, h4, .kpi-value, .hero h1, .side-title { font-family: 'Manrope', sans-serif; }
/* keep Streamlit's own icon font intact */
[data-testid="stIconMaterial"], .material-symbols-rounded { font-family: 'Material Symbols Rounded' !important; }

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.stApp { background: #F4F6F9; }
.block-container { padding-top: 1.2rem; max-width: 1280px; }

/* sidebar */
section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #DDE3EB; }
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
.side-title { color: #14213D; font-weight: 700; font-size: .95rem; margin: .9rem 0 .35rem 0;
              padding-bottom: .3rem; border-bottom: 2px solid #DDE3EB; }

/* readable text on light backgrounds, even if the browser/Streamlit theme is dark */
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] label p,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #14213D !important; }
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #5B6B7F !important; }
section[data-testid="stSidebar"] .side-title { color: #14213D !important; }
button[data-baseweb="tab"], button[data-baseweb="tab"] p { color: #14213D; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p,
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p { color: #14213D; }
.stApp [data-testid="stMarkdownContainer"] li, .stApp [data-testid="stMarkdownContainer"] strong { color: #14213D; }

/* header */
.hero { background: #14213D; border-radius: 12px; padding: 1.4rem 1.8rem; margin-bottom: 1.1rem; }
.hero h1 { color: #FFFFFF; font-size: 1.6rem; font-weight: 800; margin: 0; }
.hero p { color: #B8C4D6; margin: .3rem 0 0 0; font-size: .95rem; }

/* KPI cards */
.kpi { background: #FFFFFF; border: 1px solid #DDE3EB; border-radius: 12px; padding: 1rem 1.2rem; height: 100%; }
.kpi-label { color: #5B6B7F; font-size: .82rem; font-weight: 500; }
.kpi-value { color: #14213D; font-size: 1.7rem; font-weight: 800; line-height: 1.25; }
.kpi-sub { color: #5B6B7F; font-size: .8rem; }
.kpi.primary { border-top: 5px solid var(--seg, #2F5BEA); }

/* panels */
.panel-title { font-family: 'Manrope', sans-serif; font-weight: 700; color: #14213D; font-size: 1.05rem; margin: 0; }
.panel-hint { color: #5B6B7F; font-size: .85rem; margin: .15rem 0 .6rem 0; }
div[data-testid="stVerticalBlockBorderWrapper"] { background: #FFFFFF; border-color: #DDE3EB; border-radius: 12px; }

.insight { border-left: 4px solid #2F5BEA; background: #EEF2FD; border-radius: 6px; padding: .7rem 1rem; margin-bottom: .5rem; color: #14213D; }

/* tabs */
button[data-baseweb="tab"] { font-weight: 600; }
button[data-baseweb="tab"][aria-selected="true"], button[data-baseweb="tab"][aria-selected="true"] p { color: #2F5BEA; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================ model
FEATURES = [
    "Income", "Recency", "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds", "NumWebPurchases",
    "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth",
    "Total_Spending", "Total_Purchases", "Total_Campaign_Acceptance",
]

# Optional: give clusters your own names, e.g. {0: "Loyal big spenders"}.
# Leave empty to use automatic names based on spending level.
CLUSTER_NAMES: dict[int, str] = {}


@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    model = joblib.load("customer_segmentation_kmeans.pkl")
    scaler = joblib.load("customer_segmentation_scaler.pkl")
    # Average customer of every segment, converted back to real units.
    centers = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=FEATURES)
    return model, scaler, centers


try:
    kmeans, scaler, centers = load_artifacts()
except FileNotFoundError as err:
    st.error(f"Model file not found: {err.filename}. Put the .pkl files in the same folder as app.py.")
    st.stop()

N = len(centers)
RANK = centers["Total_Spending"].rank(ascending=False).astype(int)  # 1 = biggest spenders


def seg_name(c: int) -> str:
    if c in CLUSTER_NAMES:
        return CLUSTER_NAMES[c]
    if N == 1:
        return "All customers"
    if RANK[c] == 1:
        return "High spenders"
    if RANK[c] == N:
        return "Low spenders"
    return "Mid-range spenders" if N == 3 else f"Mid-range spenders (rank {RANK[c]} of {N})"


def seg_color(c: int) -> str:
    return PALETTE[c % len(PALETTE)]


def seg_summary(c: int) -> str:
    r = centers.loc[c]
    return (f"A typical customer here earns about {r.Income:,.0f}, spends about {r.Total_Spending:,.0f} "
            f"across {r.Total_Purchases:.0f} purchases, and last bought {r.Recency:.0f} days ago.")


# ============================================================ sidebar inputs
PRESETS = {
    "Typical customer": dict(income=50000, recency=20, visits=5, campaigns=2, wines=300, meat=200, fish=40,
                             fruits=30, sweets=25, gold=50, web=4, catalog=3, store=6),
    "Big spender": dict(income=85000, recency=10, visits=3, campaigns=3, wines=900, meat=600, fish=120,
                        fruits=80, sweets=90, gold=150, web=6, catalog=7, store=9),
    "Inactive customer": dict(income=25000, recency=75, visits=7, campaigns=0, wines=20, meat=25, fish=5,
                              fruits=5, sweets=5, gold=10, web=1, catalog=0, store=2),
}


def apply_preset():
    for k, v in PRESETS[st.session_state["preset"]].items():
        st.session_state[k] = v


for k, v in PRESETS["Typical customer"].items():
    st.session_state.setdefault(k, v)

with st.sidebar:
    st.markdown("#### Customer details")
    st.caption("Change any value and the dashboard updates instantly.")
    st.selectbox("Start from an example", list(PRESETS), key="preset", on_change=apply_preset)

    st.markdown('<div class="side-title">Profile and activity</div>', unsafe_allow_html=True)
    st.number_input("Annual income", 0, step=1000, key="income")
    st.number_input("Days since last purchase", 0, key="recency",
                    help="Lower means the customer bought more recently.")
    st.number_input("Website visits per month", 0, key="visits")
    st.slider("Marketing campaigns accepted", 0, 5, key="campaigns",
              help="How many of the 5 past campaigns this customer responded to.")

    st.markdown('<div class="side-title">Spending by category</div>', unsafe_allow_html=True)
    st.number_input("Wine", 0, key="wines")
    st.number_input("Meat", 0, key="meat")
    st.number_input("Fish", 0, key="fish")
    st.number_input("Fruit", 0, key="fruits")
    st.number_input("Sweets", 0, key="sweets")
    st.number_input("Gold products", 0, key="gold")

    st.markdown('<div class="side-title">Purchases by channel</div>', unsafe_allow_html=True)
    st.number_input("Web", 0, key="web")
    st.number_input("Catalog", 0, key="catalog")
    st.number_input("Store", 0, key="store")

# ============================================================ prediction (live)
s = st.session_state
spending = {"Wine": s.wines, "Meat": s.meat, "Fish": s.fish,
            "Fruit": s.fruits, "Sweets": s.sweets, "Gold": s.gold}
total_spending = sum(spending.values())
total_purchases = s.web + s.catalog + s.store
avg_basket = total_spending / total_purchases if total_purchases else 0

row = pd.DataFrame([{
    "Income": s.income, "Recency": s.recency,
    "MntWines": s.wines, "MntFruits": s.fruits, "MntMeatProducts": s.meat,
    "MntFishProducts": s.fish, "MntSweetProducts": s.sweets, "MntGoldProds": s.gold,
    "NumWebPurchases": s.web, "NumCatalogPurchases": s.catalog, "NumStorePurchases": s.store,
    "NumWebVisitsMonth": s.visits, "Total_Spending": total_spending,
    "Total_Purchases": total_purchases, "Total_Campaign_Acceptance": s.campaigns,
}])[FEATURES]

cluster = int(kmeans.predict(scaler.transform(row))[0])
color = seg_color(cluster)

# ============================================================ helpers
def kpi(label, value, sub="", primary=False):
    style = f' style="--seg:{color}"' if primary else ""
    cls = "kpi primary" if primary else "kpi"
    return (f'<div class="{cls}"{style}><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>')


def panel_head(title, hint):
    st.markdown(f'<p class="panel-title">{title}</p><p class="panel-hint">{hint}</p>', unsafe_allow_html=True)


# ============================================================ header
st.markdown(
    '<div class="hero"><h1>Customer Segmentation Dashboard</h1>'
    '<p>See which customer group a person belongs to, why, and what to do about it.</p></div>',
    unsafe_allow_html=True,
)

tab_result, tab_segments, tab_help = st.tabs(["Customer result", "All segments", "How to read this"])

# ------------------------------------------------------------ tab 1
with tab_result:
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("Predicted segment", seg_name(cluster), f"Segment {cluster} of {N}", primary=True),
                unsafe_allow_html=True)
    k2.markdown(kpi("Total spending", f"{total_spending:,}", "All categories combined"), unsafe_allow_html=True)
    k3.markdown(kpi("Total purchases", f"{total_purchases:,}", "Web + catalog + store"), unsafe_allow_html=True)
    k4.markdown(kpi("Average per purchase", f"{avg_basket:,.0f}", "Spending ÷ purchases"), unsafe_allow_html=True)
    st.write("")

    # --- plain-language insights
    insights = [seg_summary(cluster)]
    top_cat = max(spending, key=spending.get)
    if total_spending:
        insights.append(f"{top_cat} is this customer's biggest category, at "
                        f"{spending[top_cat] / total_spending:.0%} of all spending.")
    if s.recency > 60:
        insights.append(f"They have not bought for {s.recency} days. A win-back offer could help.")
    if RANK[cluster] == 1 and N > 1:
        insights.append("Action: reward loyalty with early access, premium bundles or exclusive offers.")
    elif RANK[cluster] == N and N > 1:
        insights.append("Action: encourage a first larger order with starter bundles or limited-time discounts.")
    else:
        insights.append("Action: suggest products from categories they buy least to grow their basket.")

    with st.container(border=True):
        panel_head("What this means", "A plain-language summary of the prediction.")
        for text in insights:
            st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)

    # --- comparison + spending mix
    left, right = st.columns([1.5, 1], gap="medium")

    with left, st.container(border=True):
        panel_head("This customer compared with the segment average",
                   "Bars to the right mean higher than the typical customer in this segment; left means lower.")
        metrics = [
            ("Income", "Annual income"), ("Total_Spending", "Total spending"),
            ("Total_Purchases", "Total purchases"), ("Recency", "Days since last purchase"),
            ("NumWebVisitsMonth", "Web visits per month"), ("Total_Campaign_Acceptance", "Campaigns accepted"),
        ]
        rows = []
        for col, label in metrics:
            mine, avg = float(row[col].iloc[0]), float(centers.loc[cluster, col])
            diff = (mine / avg - 1) * 100 if avg else 0.0
            rows.append({"Measure": label, "This customer": round(mine, 1),
                         "Segment average": round(avg, 1), "Difference %": round(diff, 1),
                         "Shown": max(min(diff, 200), -100)})
        cmp_df = pd.DataFrame(rows)

        bars = (
            alt.Chart(cmp_df)
            .mark_bar(cornerRadius=3, size=18)
            .encode(
                x=alt.X("Shown:Q", title="% difference from segment average",
                        scale=alt.Scale(domain=[-100, 200])),
                y=alt.Y("Measure:N", sort=None, title=None),
                color=alt.condition(alt.datum.Shown >= 0, alt.value(ACCENT), alt.value(AMBER)),
                tooltip=["Measure", "This customer", "Segment average", "Difference %"],
            )
        )
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=NAVY).encode(x="x:Q")
        st.altair_chart((bars + zero).properties(height=260)
                        .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=LINE, domain=False)
                        .configure_view(strokeWidth=0), use_container_width=True)
        st.dataframe(
            cmp_df.drop(columns="Shown"), hide_index=True, use_container_width=True,
            column_config={"Difference %": st.column_config.NumberColumn(format="%+.1f%%")},
        )

    with right, st.container(border=True):
        panel_head("Spending mix", "Share of total spending in each category.")
        if total_spending:
            mix = pd.DataFrame({"Category": list(spending), "Amount": list(spending.values())})
            mix["Share"] = mix["Amount"] / mix["Amount"].sum()
            donut = (
                alt.Chart(mix)
                .mark_arc(innerRadius=70, outerRadius=120)
                .encode(
                    theta="Amount:Q",
                    color=alt.Color("Category:N", scale=alt.Scale(range=PALETTE),
                                    legend=alt.Legend(orient="bottom", columns=3, title=None)),
                    tooltip=["Category", "Amount", alt.Tooltip("Share:Q", format=".0%")],
                )
                .properties(height=330)
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(donut, use_container_width=True)
        else:
            st.info("Enter some spending amounts to see the mix.")

# ------------------------------------------------------------ tab 2
with tab_segments:
    with st.container(border=True):
        panel_head("The average customer in every segment",
                   "The segment predicted for the current customer is highlighted.")
        seg_df = pd.DataFrame({
            "Segment": [f"Segment {c}" for c in range(N)],
            "Name": [seg_name(c) for c in range(N)],
            "Avg income": centers["Income"].round(0),
            "Avg total spending": centers["Total_Spending"].round(0),
            "Avg purchases": centers["Total_Purchases"].round(1),
            "Avg days since last purchase": centers["Recency"].round(0),
            "Avg web visits / month": centers["NumWebVisitsMonth"].round(1),
            "Current customer": ["Yes" if c == cluster else "" for c in range(N)],
        })
        st.dataframe(seg_df, hide_index=True, use_container_width=True)

        chart = (
            alt.Chart(seg_df)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("Avg total spending:Q", title="Average total spending"),
                y=alt.Y("Name:N", sort="-x", title=None),
                color=alt.condition(alt.datum["Current customer"] == "Yes",
                                    alt.value(ACCENT), alt.value("#B8C2D0")),
                tooltip=["Segment", "Name", "Avg total spending", "Avg income"],
            )
            .properties(height=max(120, 48 * N))
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=LINE, domain=False)
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(chart, use_container_width=True)

    for c in range(N):
        with st.expander(f"Segment {c}: {seg_name(c)}"):
            st.write(seg_summary(c))

# ------------------------------------------------------------ tab 3
with tab_help:
    with st.container(border=True):
        panel_head("How to use this dashboard", "Three steps, no technical knowledge needed.")
        st.markdown(
            """
1. **Enter customer details** in the panel on the left, or pick an example to start from.
2. **Read the result.** The dashboard updates by itself. The top card names the customer's segment.
3. **Check "What this means"** for a short explanation and a suggested next step.

**What is a segment?** Customers with similar income, spending and buying habits are grouped together.
Each segment has an average customer, and the prediction shows which group the customer is closest to.

**Segment names** are chosen automatically from how much the group spends. To use your own names,
edit `CLUSTER_NAMES` near the top of `app.py`.
"""
        )
