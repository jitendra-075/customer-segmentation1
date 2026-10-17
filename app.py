"""
Customer Segmentation Studio
----------------------------
A Streamlit dashboard built on your K-Means model (4 segments, 15 features).

Run:   streamlit run app.py
Files: keep these next to app.py
       - customer_segmentation_kmeans.pkl
       - customer_segmentation_scaler.pkl
       - marketing_campaign*.xlsx   (or upload it from the sidebar)
"""
import io
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Customer Segmentation Studio",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
SPEND = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
         "MntSweetProducts", "MntGoldProds"]
PURCH = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
CAMP = [f"AcceptedCmp{i}" for i in range(1, 6)]
DEFAULT_FEATURES = ["Income", "Recency", *SPEND, *PURCH, "NumWebVisitsMonth",
                    "Total_Spending", "Total_Purchases", "Total_Campaign_Acceptance"]
INPUT_FEATURES = ["Income", "Recency", *SPEND, *PURCH, "NumWebVisitsMonth",
                  "Total_Campaign_Acceptance"]

NICE = {
    "Income": "Income", "Recency": "Recency (days)",
    "MntWines": "Wine", "MntFruits": "Fruits", "MntMeatProducts": "Meat",
    "MntFishProducts": "Fish", "MntSweetProducts": "Sweets", "MntGoldProds": "Gold products",
    "NumWebPurchases": "Web purchases", "NumCatalogPurchases": "Catalog purchases",
    "NumStorePurchases": "Store purchases", "NumWebVisitsMonth": "Web visits / month",
    "Total_Spending": "Total spending", "Total_Purchases": "Total purchases",
    "Total_Campaign_Acceptance": "Campaigns accepted",
    "NumDealsPurchases": "Deal purchases", "Total_Children": "Children at home",
    "Age": "Age", "Response": "Last-campaign response",
}

# Segments are named by their rank on average spending (highest -> lowest)
SEG_ORDER = ["Premium Champions", "Affluent Regulars", "Deal-Seeking Families", "Budget Browsers"]
SEG_COLOR = {
    "Premium Champions": "#F5A623",
    "Affluent Regulars": "#3B82F6",
    "Deal-Seeking Families": "#10B981",
    "Budget Browsers": "#F43F5E",
}
SEG_ICON = {"Premium Champions": "👑", "Affluent Regulars": "💎",
            "Deal-Seeking Families": "🏷️", "Budget Browsers": "🛒"}
SEG_STORY = {
    "Premium Champions": "Highest income and spending, heavy on wine and meat, and by far the most "
                         "responsive to past campaigns. Few children at home.",
    "Affluent Regulars": "High income and strong spending, with a taste for meat and wine, but they "
                         "rarely accept campaigns and visit the website less.",
    "Deal-Seeking Families": "Mid income, children at home, the most deal-driven purchases. Spend is "
                             "moderate and price-sensitive.",
    "Budget Browsers": "Lowest income and spending, but the most website visits. Lots of browsing, "
                       "little buying. Largest segment.",
}
SEG_PLAYBOOK = {
    "Premium Champions": [
        "VIP / loyalty tier with early access and exclusive bundles",
        "Premium wine and meat curations; catalog and in-store experiences",
        "Use them as referral advocates - they respond to campaigns",
    ],
    "Affluent Regulars": [
        "Personalised cross-sell instead of mass campaigns (they rarely respond)",
        "Test premium offers modelled on what converts Champions",
        "Win-back messaging when recency drifts past ~60 days",
    ],
    "Deal-Seeking Families": [
        "Family bundles, multi-buy discounts and deal-of-the-week emails",
        "Points or cash-back programme to lift basket size",
        "Keep discounts targeted to protect margin",
    ],
    "Budget Browsers": [
        "Low-cost web nudges: retargeting, cart reminders, free-shipping thresholds",
        "Entry-price products to convert browsing into first purchases",
        "Avoid heavy discounting spend - low value per customer",
    ],
}


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px;}
.hero {padding: 1.4rem 1.7rem; border-radius: 18px; color: #fff; margin-bottom: 1.1rem;
       background: linear-gradient(120deg,#4f46e5 0%,#7c3aed 55%,#db2777 100%);}
.hero .t {font-size: 1.85rem; font-weight: 750; letter-spacing:-.02em; line-height:1.2;}
.hero .s {opacity:.92; margin-top:.3rem; font-size:.98rem;}
.seg-card {border:1px solid rgba(128,128,128,.28); border-left:6px solid var(--c);
           border-radius:14px; padding:.9rem 1.05rem; height:100%;}
.seg-card .n {font-weight:700; font-size:1.05rem;}
.seg-card .d {font-size:.84rem; opacity:.8; margin:.25rem 0 .6rem;}
.seg-card table {width:100%; font-size:.86rem; border-collapse:collapse;}
.seg-card td {padding:.12rem 0;}
.seg-card td:last-child {text-align:right; font-weight:650;}
.result {border-radius:16px; padding:1.1rem 1.3rem; color:#111; background:var(--c);}
.result .k {font-size:.8rem; text-transform:uppercase; letter-spacing:.08em; opacity:.75;}
.result .v {font-size:1.7rem; font-weight:800;}
div[data-testid="stTabs"] button {font-weight:600;}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Data + model loading
# ----------------------------------------------------------------------------
def _find(pattern: str):
    hits = sorted(BASE.glob(pattern))
    return hits[0] if hits else None


@st.cache_resource(show_spinner=False)
def load_models():
    kp, sp = _find("customer_segmentation_kmeans*.pkl"), _find("customer_segmentation_scaler*.pkl")
    if kp is None or sp is None:
        return None, None
    return joblib.load(kp), joblib.load(sp)


@st.cache_data(show_spinner=False)
def read_table(name: str, content: bytes) -> pd.DataFrame:
    if name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    return pd.read_excel(io.BytesIO(content))


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Same feature engineering as the notebook."""
    df = df.copy()
    df["Total_Spending"] = df[SPEND].sum(axis=1)
    df["Total_Purchases"] = df[PURCH].sum(axis=1)
    if all(c in df.columns for c in CAMP):
        df["Total_Campaign_Acceptance"] = df[CAMP].sum(axis=1)
    return df


def name_segments(df: pd.DataFrame, k: int) -> dict:
    order = df.groupby("Cluster")["Total_Spending"].mean().sort_values(ascending=False).index.tolist()
    if k == len(SEG_ORDER):
        return {c: SEG_ORDER[i] for i, c in enumerate(order)}
    return {c: f"Segment {c}" for c in order}


@st.cache_data(show_spinner="Scoring customers...")
def prepare(raw: pd.DataFrame):
    km, sc = load_models()
    feats = list(getattr(sc, "feature_names_in_", DEFAULT_FEATURES))
    df = derive(raw)
    med = df[feats].median()
    Xs = sc.transform(df[feats].fillna(med))
    df["Cluster"] = km.predict(Xs)
    names = name_segments(df, km.n_clusters)
    df["Segment"] = df["Cluster"].map(names)
    df["Age"] = 2014 - df["Year_Birth"]
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(Xs)
    df["PC1"], df["PC2"] = pcs[:, 0], pcs[:, 1]
    return df, names, med, feats, Xs, pca.explained_variance_ratio_


@st.cache_data(show_spinner="Running K sweep...")
def k_sweep(_Xs, n_rows: int):
    rows = []
    for k in range(2, 11):
        m = KMeans(n_clusters=k, random_state=42, n_init=10).fit(_Xs)
        rows.append((k, m.inertia_, silhouette_score(_Xs, m.labels_)))
    return pd.DataFrame(rows, columns=["k", "Inertia", "Silhouette"])


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def money(x, cur="$"):
    if pd.isna(x):
        return "-"
    if abs(x) >= 1e6:
        return f"{cur}{x/1e6:,.2f}M"
    return f"{cur}{x:,.0f}"


def style(fig, h=380, legend=True):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=45, b=10),
                      legend_title_text="", showlegend=legend,
                      title=dict(font=dict(size=15)))
    return fig


def plot(fig, key=None):
    st.plotly_chart(fig, width="stretch", key=key)


def norm_for_radar(series_df: pd.DataFrame, ref: pd.DataFrame, cols):
    lo, hi = ref[cols].quantile(0.05), ref[cols].quantile(0.95)
    return ((series_df[cols] - lo) / (hi - lo)).clip(0, 1)


RADAR_COLS = ["Income", "Total_Spending", "Total_Purchases",
              "Total_Campaign_Acceptance", "NumWebVisitsMonth", "Recency"]


def radar(fig, values, label, color, fill=True, dash=None):
    cats = [NICE[c] for c in RADAR_COLS]
    fig.add_trace(go.Scatterpolar(
        r=list(values) + [values[0]], theta=cats + [cats[0]], name=label,
        line=dict(color=color, width=2.5, dash=dash),
        fill="toself" if fill else None,
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},.15)" if fill else None,
    ))


# ----------------------------------------------------------------------------
# Boot: models + data
# ----------------------------------------------------------------------------
km, scaler = load_models()
if km is None:
    st.error("Model files not found. Put `customer_segmentation_kmeans.pkl` and "
             "`customer_segmentation_scaler.pkl` in the same folder as app.py.")
    st.stop()

with st.sidebar:
    st.markdown("### 🧭 Segmentation Studio")
    CUR = st.text_input("Currency symbol", "$", max_chars=3)
    uploaded = st.file_uploader("Use a different dataset (optional)", type=["xlsx", "csv"],
                                help="Must have the same columns as marketing_campaign.xlsx")

if uploaded is not None:
    raw = read_table(uploaded.name, uploaded.getvalue())
else:
    data_path = _find("marketing_campaign*.xlsx") or _find("marketing_campaign*.csv")
    if data_path is None:
        st.warning("No dataset found next to app.py. Upload `marketing_campaign.xlsx` in the sidebar.")
        st.stop()
    raw = read_table(data_path.name, data_path.read_bytes())

needed = {"Year_Birth", "Education", "Marital_Status", "Income", "Kidhome", "Teenhome", "Dt_Customer",
          "Recency", *SPEND, *PURCH, "NumWebVisitsMonth", "NumDealsPurchases", *CAMP, "Response", "ID"}
missing_cols = sorted(needed - set(raw.columns))
if missing_cols:
    st.error(f"Dataset is missing columns: {', '.join(missing_cols)}")
    st.stop()

raw["Dt_Customer"] = pd.to_datetime(raw["Dt_Customer"], errors="coerce")
df, SEG_NAMES, MEDIANS, FEATURES, XS, EVR = prepare(raw)
segs_all = [s for s in SEG_ORDER if s in df["Segment"].unique()] or sorted(df["Segment"].unique())
if len(SEG_COLOR) < len(segs_all) or not set(segs_all) <= set(SEG_COLOR):
    palette = px.colors.qualitative.Set2
    SEG_COLOR = {s: palette[i % len(palette)] for i, s in enumerate(segs_all)}

# ----------------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    st.markdown("**Filters**")
    sel_segs = st.multiselect("Segment", segs_all, default=segs_all)
    sel_edu = st.multiselect("Education", sorted(df["Education"].unique()),
                             default=sorted(df["Education"].unique()))
    sel_mar = st.multiselect("Marital status", sorted(df["Marital_Status"].unique()),
                             default=sorted(df["Marital_Status"].unique()))
    inc_lo, inc_hi = int(df["Income"].min()), int(df["Income"].max())
    sel_inc = st.slider("Income range", inc_lo, inc_hi, (inc_lo, inc_hi), step=1000)
    age_lo, age_hi = int(df["Age"].min()), int(df["Age"].max())
    sel_age = st.slider("Age range", age_lo, age_hi, (age_lo, age_hi))
    kids_opt = st.radio("Children at home", ["All", "No children", "Has children"], horizontal=True)
    st.markdown("---")
    st.caption("Model: K-Means, k=4, trained on 15 scaled features. Segment names are assigned by "
               "average spending rank.")

income_ok = df["Income"].between(*sel_inc) | (df["Income"].isna() & (sel_inc == (inc_lo, inc_hi)))
mask = (
    df["Segment"].isin(sel_segs)
    & df["Education"].isin(sel_edu)
    & df["Marital_Status"].isin(sel_mar)
    & income_ok
)
mask &= df["Age"].between(*sel_age)
if kids_opt == "No children":
    mask &= df["Total_Children"] == 0
elif kids_opt == "Has children":
    mask &= df["Total_Children"] > 0
f = df[mask].copy()

st.markdown(
    f"""<div class="hero"><div class="t">Customer Segmentation Studio</div>
    <div class="s">Who your customers are, what they are worth, and how to reach each segment. 
    Showing <b>{len(f):,}</b> of {len(df):,} customers.</div></div>""",
    unsafe_allow_html=True,
)
if f.empty:
    st.warning("No customers match the current filters.")
    st.stop()

segs = [s for s in segs_all if s in f["Segment"].unique()]

seg_stats = (
    f.groupby("Segment")
    .agg(Customers=("ID", "count"), Revenue=("Total_Spending", "sum"),
         AvgSpend=("Total_Spending", "mean"), MedIncome=("Income", "median"),
         Response=("Response", "mean"), Purchases=("Total_Purchases", "mean"),
         Recency=("Recency", "mean"), WebVisits=("NumWebVisitsMonth", "mean"),
         CampAcc=("Total_Campaign_Acceptance", "mean"), Kids=("Total_Children", "mean"),
         Deals=("NumDealsPurchases", "mean"), Age=("Age", "mean"))
    .reindex(segs)
)

tab_over, tab_seg, tab_beh, tab_pred, tab_model, tab_data = st.tabs(
    ["📊 Overview", "🧩 Segments", "🎯 Campaigns & Behaviour", "🔮 Predict a Customer",
     "🧪 Model Quality", "🗂️ Data"]
)

# ============================================================================
# OVERVIEW
# ============================================================================
with tab_over:
    overall = df
    c = st.columns(6)
    c[0].metric("Customers", f"{len(f):,}", border=True)
    c[1].metric("Total revenue", money(f["Total_Spending"].sum(), CUR), border=True)
    c[2].metric("Avg spend / customer", money(f["Total_Spending"].mean(), CUR),
                delta=f"{(f['Total_Spending'].mean()/overall['Total_Spending'].mean()-1):+.0%} vs all"
                if len(f) != len(df) else None, border=True)
    c[3].metric("Median income", money(f["Income"].median(), CUR), border=True)
    c[4].metric("Campaign response", f"{f['Response'].mean():.1%}",
                delta=f"{(f['Response'].mean()-overall['Response'].mean())*100:+.1f} pts vs all"
                if len(f) != len(df) else None, border=True)
    c[5].metric("Avg recency", f"{f['Recency'].mean():.0f} days", border=True)

    # --- auto insights
    top_rev = seg_stats["Revenue"].idxmax()
    cust_share = seg_stats.loc[top_rev, "Customers"] / seg_stats["Customers"].sum()
    rev_share = seg_stats.loc[top_rev, "Revenue"] / seg_stats["Revenue"].sum()
    best_resp = seg_stats["Response"].idxmax()
    biggest = seg_stats["Customers"].idxmax()
    st.info(
        f"**{SEG_ICON.get(top_rev,'')} {top_rev}** are {cust_share:.0%} of customers but drive "
        f"**{rev_share:.0%}** of revenue.  \n"
        f"**{best_resp}** respond best to the latest campaign ({seg_stats.loc[best_resp,'Response']:.0%}).  \n"
        f"**{biggest}** are the largest group ({seg_stats.loc[biggest,'Customers']:,} customers, "
        f"avg spend {money(seg_stats.loc[biggest,'AvgSpend'], CUR)})."
    )

    a, b = st.columns([1, 1.25])
    with a:
        fig = px.pie(seg_stats.reset_index(), names="Segment", values="Customers", hole=0.58,
                     color="Segment", color_discrete_map=SEG_COLOR,
                     category_orders={"Segment": segs}, title="Customers by segment")
        fig.update_traces(textinfo="percent+label", textposition="outside", sort=False)
        plot(style(fig, 400, legend=False))
    with b:
        share = pd.DataFrame({
            "Segment": seg_stats.index,
            "Share of customers": seg_stats["Customers"] / seg_stats["Customers"].sum(),
            "Share of revenue": seg_stats["Revenue"] / seg_stats["Revenue"].sum(),
        }).melt("Segment", var_name="Metric", value_name="Share")
        fig = px.bar(share, x="Segment", y="Share", color="Metric", barmode="group", text_auto=".0%",
                     category_orders={"Segment": segs}, title="Customer share vs revenue share",
                     color_discrete_sequence=["#94a3b8", "#6366f1"])
        fig.update_yaxes(tickformat=".0%", title="")
        fig.update_xaxes(title="")
        plot(style(fig, 400))

    a, b = st.columns(2)
    with a:
        cat = (f.groupby("Segment")[SPEND].sum().reindex(segs).rename(columns=NICE)
               .T.reset_index().melt("index", var_name="Segment", value_name="Revenue")
               .rename(columns={"index": "Category"}))
        order = cat.groupby("Category")["Revenue"].sum().sort_values(ascending=False).index
        fig = px.bar(cat, x="Category", y="Revenue", color="Segment", color_discrete_map=SEG_COLOR,
                     category_orders={"Category": list(order), "Segment": segs},
                     title="Revenue by product category")
        fig.update_xaxes(title="")
        plot(style(fig))
    with b:
        ch = (f.groupby("Segment")[PURCH].mean().reindex(segs)
              .rename(columns={"NumWebPurchases": "Web", "NumCatalogPurchases": "Catalog",
                               "NumStorePurchases": "Store"})
              .reset_index().melt("Segment", var_name="Channel", value_name="Avg purchases"))
        fig = px.bar(ch, x="Channel", y="Avg purchases", color="Segment", barmode="group",
                     color_discrete_map=SEG_COLOR, category_orders={"Segment": segs},
                     title="Avg purchases per customer by channel", text_auto=".1f")
        fig.update_xaxes(title="")
        plot(style(fig))

# ============================================================================
# SEGMENTS
# ============================================================================
with tab_seg:
    cols = st.columns(len(segs))
    for col, s in zip(cols, segs):
        r = seg_stats.loc[s]
        col.markdown(
            f"""<div class="seg-card" style="--c:{SEG_COLOR[s]}">
            <div class="n">{SEG_ICON.get(s,'')} {s}</div>
            <div class="d">{SEG_STORY.get(s,'')}</div>
            <table>
            <tr><td>Customers</td><td>{int(r.Customers):,}</td></tr>
            <tr><td>Avg spend</td><td>{money(r.AvgSpend, CUR)}</td></tr>
            <tr><td>Median income</td><td>{money(r.MedIncome, CUR)}</td></tr>
            <tr><td>Avg purchases</td><td>{r.Purchases:.1f}</td></tr>
            <tr><td>Campaigns accepted</td><td>{r.CampAcc:.2f}</td></tr>
            <tr><td>Web visits / month</td><td>{r.WebVisits:.1f}</td></tr>
            <tr><td>Children at home</td><td>{r.Kids:.1f}</td></tr>
            </table></div>""",
            unsafe_allow_html=True,
        )
    st.write("")

    a, b = st.columns(2)
    with a:
        prof_all = df.groupby("Segment")[RADAR_COLS].mean().reindex(segs_all)
        nr = norm_for_radar(prof_all, df, RADAR_COLS)
        fig = go.Figure()
        for s in segs_all:
            radar(fig, nr.loc[s].values, s, SEG_COLOR[s])
        fig.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
                          title="Segment fingerprints (all customers)")
        plot(style(fig, 430))
    with b:
        means = df.groupby("Segment")[FEATURES].mean().reindex(segs_all)
        z = (means - scaler.mean_) / scaler.scale_
        fig = px.imshow(z.rename(columns=NICE).round(2), aspect="auto", text_auto=".1f",
                        color_continuous_scale="RdBu_r", zmin=-2, zmax=2,
                        title="Profile heatmap (z-scores vs average customer)")
        fig.update_xaxes(tickangle=-40)
        fig.update_coloraxes(showscale=False)
        plot(style(fig, 430, legend=False))
    st.caption("Radar and heatmap use all customers so the model profile stays stable; "
               "everything else on this page respects the sidebar filters.")

    a, b = st.columns(2)
    with a:
        num_opts = ["Income", "Total_Spending", "Total_Purchases", "Recency", "Age",
                    "NumWebVisitsMonth", "MntWines", "MntMeatProducts", "NumDealsPurchases"]
        x1, y1 = st.columns(2)
        xa = x1.selectbox("X axis", num_opts, index=0, format_func=lambda v: NICE.get(v, v))
        ya = y1.selectbox("Y axis", num_opts, index=1, format_func=lambda v: NICE.get(v, v))
        pf = f
        if "Income" in (xa, ya):
            pf = f[f["Income"].fillna(0) <= 200_000]
            if len(pf) < len(f):
                st.caption(f"{len(f)-len(pf)} extreme-income outlier(s) hidden from this chart.")
        fig = px.scatter(pf, x=xa, y=ya, color="Segment", color_discrete_map=SEG_COLOR,
                         category_orders={"Segment": segs}, opacity=0.7,
                         labels=NICE, title=f"{NICE.get(ya, ya)} vs {NICE.get(xa, xa)}")
        plot(style(fig, 420))
    with b:
        st.write("")
        st.write("")
        st.write("")
        fig = px.scatter(f, x="PC1", y="PC2", color="Segment", color_discrete_map=SEG_COLOR,
                         category_orders={"Segment": segs}, opacity=0.7,
                         title=f"PCA projection ({EVR.sum():.0%} of variance explained)")
        plot(style(fig, 420))

    st.markdown("#### Recommended playbook")
    pcols = st.columns(len(segs))
    for col, s in zip(pcols, segs):
        with col:
            with st.container(border=True):
                st.markdown(f"**{SEG_ICON.get(s,'')} {s}**")
                for tip in SEG_PLAYBOOK.get(s, ["Profile this segment to define a playbook."]):
                    st.markdown(f"- {tip}")

# ============================================================================
# CAMPAIGNS & BEHAVIOUR
# ============================================================================
with tab_beh:
    a, b = st.columns(2)
    with a:
        camp = f.groupby("Segment")[CAMP + ["Response"]].mean().reindex(segs) * 100
        camp.columns = [f"Campaign {i}" for i in range(1, 6)] + ["Latest campaign"]
        fig = px.imshow(camp.round(1), text_auto=".1f", aspect="auto", color_continuous_scale="Purples",
                        title="Acceptance rate by campaign (%)")
        fig.update_coloraxes(showscale=False)
        plot(style(fig, 360, legend=False))
    with b:
        fig = px.bar(seg_stats.reset_index(), x="Segment", y="Response", color="Segment",
                     color_discrete_map=SEG_COLOR, category_orders={"Segment": segs},
                     text_auto=".0%", title="Latest-campaign response rate")
        fig.update_yaxes(tickformat=".0%", title="")
        fig.update_xaxes(title="")
        plot(style(fig, 360, legend=False))

    a, b, c3 = st.columns(3)
    with a:
        fig = px.bar(seg_stats.reset_index(), x="Segment", y="WebVisits", color="Segment",
                     color_discrete_map=SEG_COLOR, category_orders={"Segment": segs},
                     text_auto=".1f", title="Web visits per month")
        fig.update_yaxes(title="")
        fig.update_xaxes(title="")
        plot(style(fig, 330, legend=False))
    with b:
        fig = px.bar(seg_stats.reset_index(), x="Segment", y="Deals", color="Segment",
                     color_discrete_map=SEG_COLOR, category_orders={"Segment": segs},
                     text_auto=".1f", title="Deal purchases per customer")
        fig.update_yaxes(title="")
        fig.update_xaxes(title="")
        plot(style(fig, 330, legend=False))
    with c3:
        fig = px.box(f, x="Segment", y="Recency", color="Segment", color_discrete_map=SEG_COLOR,
                     category_orders={"Segment": segs}, title="Recency (days since last purchase)")
        fig.update_xaxes(title="")
        fig.update_yaxes(title="")
        plot(style(fig, 330, legend=False))

    a, b = st.columns(2)
    with a:
        q = f.dropna(subset=["Dt_Customer"]).copy()
        q["Quarter"] = q["Dt_Customer"].dt.to_period("Q").astype(str)
        qq = q.groupby(["Quarter", "Segment"]).size().reset_index(name="Customers")
        fig = px.bar(qq, x="Quarter", y="Customers", color="Segment", color_discrete_map=SEG_COLOR,
                     category_orders={"Segment": segs}, title="Customer acquisition by quarter")
        fig.update_xaxes(title="", tickangle=-45)
        plot(style(fig, 380))
    with b:
        edu = f.groupby("Education")["Total_Spending"].agg(["mean", "count"]).reset_index()
        fig = px.bar(edu.sort_values("mean", ascending=False), x="Education", y="mean",
                     text_auto=".0f", title="Avg spend by education",
                     color_discrete_sequence=["#6366f1"], hover_data={"count": True})
        fig.update_xaxes(title="")
        fig.update_yaxes(title="")
        plot(style(fig, 380, legend=False))

    a, b = st.columns(2)
    with a:
        kd = f.groupby(["Total_Children", "Segment"])["Total_Spending"].mean().reset_index()
        fig = px.bar(kd, x="Total_Children", y="Total_Spending", color="Segment", barmode="group",
                     color_discrete_map=SEG_COLOR, category_orders={"Segment": segs},
                     title="Avg spend by number of children at home", labels=NICE)
        plot(style(fig, 380))
    with b:
        ms = f.groupby("Marital_Status")["Total_Spending"].agg(["mean", "count"]).reset_index()
        ms = ms[ms["count"] >= 10]
        fig = px.bar(ms.sort_values("mean", ascending=False), x="Marital_Status", y="mean",
                     text_auto=".0f", title="Avg spend by marital status (groups with 10+ customers)",
                     color_discrete_sequence=["#14b8a6"], hover_data={"count": True})
        fig.update_xaxes(title="")
        fig.update_yaxes(title="")
        plot(style(fig, 380, legend=False))

    corr_cols = ["Income", "Age", "Recency", "Total_Spending", "Total_Purchases", "NumWebVisitsMonth",
                 "NumDealsPurchases", "Total_Campaign_Acceptance", "Total_Children", "Response"]
    cm = f[corr_cols].corr().rename(index=NICE, columns=NICE)
    fig = px.imshow(cm.round(2), text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                    aspect="auto", title="Correlation matrix")
    plot(style(fig, 520, legend=False))

# ============================================================================
# PREDICT
# ============================================================================
with tab_pred:
    st.markdown("Describe a customer and the saved **K-Means model** assigns a segment. "
                "Totals are calculated automatically, exactly as in the notebook.")

    pres = {"Median customer": MEDIANS[INPUT_FEATURES].to_dict()}
    seg_means = df.groupby("Segment")[INPUT_FEATURES].mean()
    for s in segs_all:
        pres[f"Typical {s}"] = seg_means.loc[s].to_dict()

    def _cast(k, v):
        return float(round(v)) if k == "Income" else int(round(v))

    def apply_preset():
        for k, v in pres[st.session_state["preset"]].items():
            st.session_state[f"in_{k}"] = _cast(k, v)

    if "in_Income" not in st.session_state:
        for k, v in pres["Median customer"].items():
            st.session_state[f"in_{k}"] = _cast(k, v)

    st.selectbox("Start from a preset", list(pres), key="preset", on_change=apply_preset)

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("##### Customer profile")
        r1 = st.columns(3)
        r1[0].number_input("Income", 0, 700_000, step=1000, key="in_Income")
        r1[1].number_input("Recency (days)", 0, 99, key="in_Recency")
        r1[2].number_input("Web visits / month", 0, 30, key="in_NumWebVisitsMonth")
        st.markdown("##### Spending (last 2 years)")
        r2 = st.columns(3)
        r3 = st.columns(3)
        for i, k in enumerate(SPEND):
            (r2 if i < 3 else r3)[i % 3].number_input(NICE[k], 0, 3000, key=f"in_{k}")
        st.markdown("##### Purchases & campaigns")
        r4 = st.columns(4)
        for i, k in enumerate(PURCH):
            r4[i].number_input(NICE[k], 0, 50, key=f"in_{k}")
        r4[3].number_input("Campaigns accepted (0-5)", 0, 5, key="in_Total_Campaign_Acceptance")

    vals = {k: st.session_state[f"in_{k}"] for k in INPUT_FEATURES}
    row = dict(vals)
    row["Total_Spending"] = sum(vals[k] for k in SPEND)
    row["Total_Purchases"] = sum(vals[k] for k in PURCH)
    xrow = pd.DataFrame([row])[FEATURES]
    xs_row = scaler.transform(xrow)
    cl = int(km.predict(xs_row)[0])
    seg = SEG_NAMES[cl]
    dist = km.transform(xs_row)[0]
    closeness = pd.Series(1 / (dist ** 2 + 1e-9), index=[SEG_NAMES[i] for i in range(len(dist))])
    closeness = (closeness / closeness.sum()).reindex(segs_all)

    with right:
        st.markdown(
            f"""<div class="result" style="--c:{SEG_COLOR[seg]}"><div class="k">Predicted segment</div>
            <div class="v">{SEG_ICON.get(seg,'')} {seg}</div>
            <div style="margin-top:.3rem;font-size:.9rem">{SEG_STORY.get(seg,'')}</div></div>""",
            unsafe_allow_html=True,
        )
        st.write("")
        m = st.columns(3)
        m[0].metric("Total spending", money(row["Total_Spending"], CUR), border=True)
        m[1].metric("Total purchases", row["Total_Purchases"], border=True)
        m[2].metric("vs segment avg spend",
                    f"{row['Total_Spending']/df.loc[df.Segment==seg,'Total_Spending'].mean()-1:+.0%}", border=True)

        fig = go.Figure()
        pc = pd.DataFrame([row])[RADAR_COLS]
        radar(fig, norm_for_radar(pc, df, RADAR_COLS).iloc[0].values, "This customer", "#111827")
        seg_prof = df[df.Segment == seg][RADAR_COLS].mean().to_frame().T
        radar(fig, norm_for_radar(seg_prof, df, RADAR_COLS).iloc[0].values, f"{seg} average",
              SEG_COLOR[seg], dash="dash")
        fig.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
                          title="Customer vs segment average")
        plot(style(fig, 360))

    a, b = st.columns([1, 1])
    with a:
        fig = px.bar(closeness.reset_index(), y="index", x=0, orientation="h", color="index",
                     color_discrete_map=SEG_COLOR, text_auto=".0%",
                     title="Relative closeness to each segment centroid")
        fig.update_xaxes(tickformat=".0%", title="")
        fig.update_yaxes(title="", categoryorder="array", categoryarray=segs_all[::-1])
        plot(style(fig, 300, legend=False))
        st.caption("Inverse squared distance to each centroid, normalised. A high share on a "
                   "second segment means the customer sits near a boundary.")
    with b:
        with st.container(border=True):
            st.markdown(f"**Recommended actions for {seg}**")
            for tip in SEG_PLAYBOOK.get(seg, []):
                st.markdown(f"- {tip}")

    # ---- batch scoring
    with st.expander("📥 Score a whole file of customers"):
        st.caption("Upload an .xlsx/.csv with the original columns (Income, Recency, the six Mnt* columns, "
                   "the three Num*Purchases columns, NumWebVisitsMonth and AcceptedCmp1-5). "
                   "Totals are computed for you.")
        bf = st.file_uploader("Customer file", type=["xlsx", "csv"], key="batch")
        if bf is not None:
            bdf = read_table(bf.name, bf.getvalue())
            need = {"Income", "Recency", *SPEND, *PURCH, "NumWebVisitsMonth"}
            if not all(c in bdf.columns for c in CAMP) and "Total_Campaign_Acceptance" not in bdf.columns:
                need |= set(CAMP)
            miss = sorted(need - set(bdf.columns))
            if miss:
                st.error(f"Missing columns: {', '.join(miss)}")
            else:
                bdf = derive(bdf)
                xb = scaler.transform(bdf[FEATURES].fillna(MEDIANS))
                bdf["Cluster"] = km.predict(xb)
                bdf["Segment"] = bdf["Cluster"].map(SEG_NAMES)
                st.success(f"Scored {len(bdf):,} customers.")
                st.dataframe(bdf["Segment"].value_counts().rename("Customers").to_frame(), width="stretch")
                st.download_button("Download scored file (CSV)", bdf.to_csv(index=False).encode(),
                                   "scored_customers.csv", "text/csv")

# ============================================================================
# MODEL QUALITY
# ============================================================================
with tab_model:
    sil = silhouette_score(XS, df["Cluster"])
    dbi = davies_bouldin_score(XS, df["Cluster"])
    chi = calinski_harabasz_score(XS, df["Cluster"])
    m = st.columns(4)
    m[0].metric("Clusters (k)", km.n_clusters, border=True)
    m[1].metric("Silhouette", f"{sil:.3f}", help="Higher is better", border=True)
    m[2].metric("Davies-Bouldin", f"{dbi:.3f}", help="Lower is better", border=True)
    m[3].metric("Calinski-Harabasz", f"{chi:,.0f}", help="Higher is better", border=True)
    st.caption("Silhouette around 0.29 means moderately separated clusters: useful for marketing "
               "segmentation, but boundaries between segments are soft (see the closeness chart in "
               "the Predict tab).")

    sweep = k_sweep(XS, len(XS))
    a, b = st.columns(2)
    with a:
        fig = go.Figure(go.Scatter(x=sweep["k"], y=sweep["Inertia"], mode="lines+markers",
                                   line=dict(color="#6366f1", width=3)))
        fig.add_vline(x=km.n_clusters, line_dash="dash", line_color="#F43F5E",
                      annotation_text=f"k={km.n_clusters}")
        fig.update_layout(title="Elbow method", xaxis_title="Number of clusters", yaxis_title="Inertia")
        plot(style(fig, 340, legend=False))
    with b:
        fig = go.Figure(go.Scatter(x=sweep["k"], y=sweep["Silhouette"], mode="lines+markers",
                                   line=dict(color="#10B981", width=3)))
        fig.add_vline(x=km.n_clusters, line_dash="dash", line_color="#F43F5E",
                      annotation_text=f"k={km.n_clusters}")
        fig.update_layout(title="Silhouette score by k", xaxis_title="Number of clusters",
                          yaxis_title="Silhouette")
        plot(style(fig, 340, legend=False))
    best_k = int(sweep.loc[sweep["Silhouette"].idxmax(), "k"])
    if best_k != km.n_clusters:
        st.info(f"Silhouette peaks at k={best_k}, while the deployed model uses k={km.n_clusters}. "
                f"Four segments is a reasonable business choice (more actionable than two), "
                f"but it is worth knowing the metric prefers fewer groups.")

    st.markdown("#### Algorithm comparison (from your notebook)")
    comp = pd.DataFrame({
        "Model": ["K-Means", "Hierarchical", "BIRCH", "GMM"],
        "Silhouette": [0.287411, 0.268379, 0.241734, 0.080003],
        "Davies-Bouldin": [1.893764, 1.839451, 1.905494, 2.393245],
        "Calinski-Harabasz": [787.422964, 703.825304, 683.735550, 555.020264],
    })
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Silhouette (higher better)",
                                                        "Davies-Bouldin (lower better)",
                                                        "Calinski-Harabasz (higher better)"))
    colors = ["#6366f1" if mname == "K-Means" else "#cbd5e1" for mname in comp["Model"]]
    for i, col in enumerate(["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"], 1):
        fig.add_trace(go.Bar(x=comp["Model"], y=comp[col], marker_color=colors,
                             text=comp[col].round(3 if i < 3 else 0), textposition="outside",
                             showlegend=False), row=1, col=i)
    plot(style(fig, 360, legend=False))
    st.caption("K-Means wins on 2 of 3 metrics and is the deployed model. DBSCAN (eps=1.5, "
               "min_samples=5) did not produce a valid cluster structure for these metrics. "
               "These figures are copied from the notebook, not recomputed.")

# ============================================================================
# DATA
# ============================================================================
with tab_data:
    st.markdown("#### Data health")
    h = st.columns(5)
    h[0].metric("Rows", f"{len(df):,}", border=True)
    h[1].metric("Missing income", int(df["Income"].isna().sum()), help="Filled with the median for scoring",
                border=True)
    h[2].metric("Duplicate rows", int(raw.drop(columns=["ID"]).duplicated().sum()), border=True)
    h[3].metric("Income > 150k", int((df["Income"] > 150_000).sum()), border=True)
    h[4].metric("Age > 90", int((df["Age"] > 90).sum()), help="Age = 2014 - Year_Birth", border=True)
    const = [c for c in raw.columns if raw[c].nunique() <= 1]
    if const:
        st.caption(f"Constant columns with no information: {', '.join(const)}.")

    st.markdown("#### Customer table")
    show = ["ID", "Segment", "Age", "Education", "Marital_Status", "Income", "Total_Children", "Recency",
            "Total_Spending", "Total_Purchases", "Total_Campaign_Acceptance", "NumWebVisitsMonth",
            "NumDealsPurchases", "Response", "Dt_Customer"]
    st.dataframe(
        f[show].sort_values("Total_Spending", ascending=False),
        width="stretch", hide_index=True, height=480,
        column_config={
            "Income": st.column_config.NumberColumn(format="%d"),
            "Total_Spending": st.column_config.ProgressColumn(
                "Total spending", min_value=0, max_value=int(df["Total_Spending"].max()), format="%d"),
            "Dt_Customer": st.column_config.DateColumn("Customer since"),
        },
    )
    st.download_button("⬇️ Download filtered customers (CSV)",
                       f.drop(columns=["PC1", "PC2"]).to_csv(index=False).encode(),
                       "segmented_customers.csv", "text/csv")
