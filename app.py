"""
Customer Groups Dashboard (simple version)
Run:  streamlit run app.py
Keep these files in the same folder as app.py:
  customer_segmentation_kmeans.pkl, customer_segmentation_scaler.pkl, marketing_campaign*.xlsx
"""
import io
import warnings
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Customer Groups", page_icon="🛍️", layout="wide")
BASE = Path(__file__).parent

# ---------------------------------------------------------------- settings
SPEND = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
BUY = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
OFFERS = [f"AcceptedCmp{i}" for i in range(1, 6)]
FEATURES = ["Income", "Recency", *SPEND, *BUY, "NumWebVisitsMonth",
            "Total_Spending", "Total_Purchases", "Total_Campaign_Acceptance"]
INPUTS = ["Income", "Recency", *SPEND, *BUY, "NumWebVisitsMonth", "Total_Campaign_Acceptance"]

LABEL = {"MntWines": "Wine", "MntFruits": "Fruits", "MntMeatProducts": "Meat",
         "MntFishProducts": "Fish", "MntSweetProducts": "Sweets", "MntGoldProds": "Premium items",
         "NumWebPurchases": "Online", "NumCatalogPurchases": "Catalog", "NumStorePurchases": "In store"}

# Groups are named by how much they spend (highest first)
GROUPS = ["Big Spenders", "Steady Shoppers", "Bargain Hunters", "Light Shoppers"]
COLOR = {"Big Spenders": "#F5A623", "Steady Shoppers": "#3B82F6",
         "Bargain Hunters": "#10B981", "Light Shoppers": "#F43F5E"}
ICON = {"Big Spenders": "👑", "Steady Shoppers": "💎", "Bargain Hunters": "🏷️", "Light Shoppers": "🛒"}
ABOUT = {
    "Big Spenders": "High income. They spend the most and say yes to offers more than anyone else.",
    "Steady Shoppers": "High income and they spend a lot, but they rarely respond to offers.",
    "Bargain Hunters": "Middle income, usually with kids at home. They buy the most discounted items.",
    "Light Shoppers": "Lowest income and spend little, but visit the website a lot. The biggest group.",
}
TIPS = {
    "Big Spenders": ["Reward them with a VIP or loyalty club.", "Offer premium wine and meat bundles.",
                     "Ask them to refer friends - they respond well."],
    "Steady Shoppers": ["Send personal suggestions, not mass offers.", "Try premium products they may like.",
                        "Contact them if they have not bought for 60+ days."],
    "Bargain Hunters": ["Offer family bundles and multi-buy deals.", "Start a points or cash-back scheme.",
                        "Limit discounts to protect your profit."],
    "Light Shoppers": ["Use cheap online nudges: reminders, free-delivery offers.",
                       "Promote low-priced starter products.", "Do not overspend on discounts here."],
}

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1300px;}
.hero {padding: 1.3rem 1.6rem; border-radius: 16px; color: #fff; margin-bottom: 1rem;
       background: linear-gradient(120deg,#4f46e5,#7c3aed 55%,#db2777);}
.hero .t {font-size: 1.8rem; font-weight: 750;}
.hero .s {opacity: .93; margin-top: .2rem;}
.card {border: 1px solid rgba(128,128,128,.3); border-top: 6px solid var(--c); border-radius: 14px;
       padding: .9rem 1rem; height: 100%;}
.card .n {font-weight: 700; font-size: 1.1rem;}
.card .d {font-size: .88rem; opacity: .85; margin: .3rem 0 .6rem;}
.card table {width: 100%; font-size: .9rem; border-collapse: collapse;}
.card td:last-child {text-align: right; font-weight: 650;}
.answer {border-radius: 16px; padding: 1.1rem 1.3rem; color: #111; background: var(--c);}
.answer .v {font-size: 1.8rem; font-weight: 800;}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- loading
def find(pattern):
    hits = sorted(BASE.glob(pattern))
    return hits[0] if hits else None


@st.cache_resource(show_spinner=False)
def load_models():
    k, s = find("customer_segmentation_kmeans*.pkl"), find("customer_segmentation_scaler*.pkl")
    return (joblib.load(k), joblib.load(s)) if k and s else (None, None)


@st.cache_data(show_spinner=False)
def read_table(name, content):
    buf = io.BytesIO(content)
    return pd.read_csv(buf) if name.lower().endswith(".csv") else pd.read_excel(buf)


def add_totals(df):
    df = df.copy()
    df["Total_Spending"] = df[SPEND].sum(axis=1)
    df["Total_Purchases"] = df[BUY].sum(axis=1)
    if all(c in df.columns for c in OFFERS):
        df["Total_Campaign_Acceptance"] = df[OFFERS].sum(axis=1)
    return df


@st.cache_data(show_spinner="Sorting customers into groups...")
def prepare(raw):
    km, sc = load_models()
    df = add_totals(raw)
    med = df[FEATURES].median()
    df["Cluster"] = km.predict(sc.transform(df[FEATURES].fillna(med)))
    rank = df.groupby("Cluster")["Total_Spending"].mean().sort_values(ascending=False).index.tolist()
    names = {c: (GROUPS[i] if i < len(GROUPS) else f"Group {c}") for i, c in enumerate(rank)}
    df["Group"] = df["Cluster"].map(names)
    df["Age"] = 2014 - df["Year_Birth"]
    df["Kids at home"] = df["Kidhome"] + df["Teenhome"]
    return df, names, med


def money(x):
    return f"{CUR}{x/1e6:,.2f}M" if abs(x) >= 1e6 else f"{CUR}{x:,.0f}"


def show(fig, h=360, legend=True):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="",
                      showlegend=legend, title=dict(font=dict(size=16)))
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------- boot
km, scaler = load_models()
if km is None:
    st.error("Model files not found. Put the two .pkl files next to app.py.")
    st.stop()

with st.sidebar:
    st.markdown("### 🛍️ Customer Groups")
    CUR = st.text_input("Currency symbol", "$", max_chars=3)
    up = st.file_uploader("Use your own customer file (optional)", type=["xlsx", "csv"])
    st.caption("Leave this empty to use the built-in data.")

if up is not None:
    raw = read_table(up.name, up.getvalue())
else:
    path = find("marketing_campaign*.xlsx") or find("marketing_campaign*.csv")
    if path is None:
        st.warning("No customer file found. Upload one in the sidebar.")
        st.stop()
    raw = read_table(path.name, path.read_bytes())

df, NAMES, MEDIANS = prepare(raw)
groups = [g for g in GROUPS if g in df["Group"].unique()] or sorted(df["Group"].unique())
for g in groups:
    COLOR.setdefault(g, "#888888")

stats = df.groupby("Group").agg(
    Customers=("ID", "count"), Sales=("Total_Spending", "sum"), AvgSpend=("Total_Spending", "mean"),
    Income=("Income", "median"), YesRate=("Response", "mean"), Offers=("Total_Campaign_Acceptance", "mean"),
    Kids=("Kids at home", "mean"), WebVisits=("NumWebVisitsMonth", "mean"),
).reindex(groups)

st.markdown("""<div class="hero"><div class="t">🛍️ Customer Groups</div>
<div class="s">See who your customers are, who brings in the money, and how to talk to each group.</div></div>""",
            unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🏠 Summary", "👥 Meet the groups", "🔍 Which group is this customer?", "📋 Customer list"])

# ================================================================ 1. SUMMARY
with tab1:
    c = st.columns(4)
    c[0].metric("Customers", f"{len(df):,}", border=True)
    c[1].metric("Total sales", money(df["Total_Spending"].sum()), border=True)
    c[2].metric("Average spend per customer", money(df["Total_Spending"].mean()), border=True)
    c[3].metric("Said yes to the last offer", f"{df['Response'].mean():.0%}", border=True)

    top = stats["Sales"].idxmax()
    cs, ss = stats.loc[top, "Customers"] / stats["Customers"].sum(), stats.loc[top, "Sales"] / stats["Sales"].sum()
    big = stats["Customers"].idxmax()
    st.info(f"**Key finding:** {ICON.get(top,'')} **{top}** are only {cs:.0%} of your customers "
            f"but bring in **{ss:.0%}** of your sales.  \n"
            f"The biggest group is **{big}** ({stats.loc[big,'Customers']/stats['Customers'].sum():.0%} of customers).")

    a, b = st.columns(2)
    with a:
        share = pd.DataFrame({"Group": stats.index,
                              "Share of customers": stats["Customers"] / stats["Customers"].sum(),
                              "Share of sales": stats["Sales"] / stats["Sales"].sum()}
                             ).melt("Group", var_name="", value_name="Share")
        fig = px.bar(share, x="Group", y="Share", color="", barmode="group", text_auto=".0%",
                     category_orders={"Group": groups}, color_discrete_sequence=["#94a3b8", "#6366f1"],
                     title="Share of customers vs share of sales")
        fig.update_yaxes(tickformat=".0%", title="")
        fig.update_xaxes(title="")
        show(fig)
    with b:
        fig = px.bar(stats.reset_index(), x="Group", y="AvgSpend", color="Group", color_discrete_map=COLOR,
                     category_orders={"Group": groups}, text_auto=".0f",
                     title="Average spend per customer")
        fig.update_yaxes(title="")
        fig.update_xaxes(title="")
        show(fig, legend=False)

    a, b = st.columns(2)
    with a:
        prod = df.groupby("Group")[SPEND].mean().reindex(groups).rename(columns=LABEL)
        prod = prod.reset_index().melt("Group", var_name="Product", value_name="Avg spend")
        fig = px.bar(prod, x="Product", y="Avg spend", color="Group", barmode="group",
                     color_discrete_map=COLOR, category_orders={"Group": groups},
                     title="What each group buys (average spend)")
        fig.update_xaxes(title="")
        fig.update_yaxes(title="")
        show(fig)
    with b:
        chan = df.groupby("Group")[BUY].mean().reindex(groups).rename(columns=LABEL)
        chan = chan.reset_index().melt("Group", var_name="Where", value_name="Avg purchases")
        fig = px.bar(chan, x="Where", y="Avg purchases", color="Group", barmode="group",
                     color_discrete_map=COLOR, category_orders={"Group": groups},
                     title="Where each group shops (average purchases)")
        fig.update_xaxes(title="")
        fig.update_yaxes(title="")
        show(fig)

# ================================================================ 2. GROUPS
with tab2:
    st.markdown("Your customers fall into **four groups** of similar shoppers. Here is what each one looks like.")
    cols = st.columns(len(groups))
    for col, g in zip(cols, groups):
        r = stats.loc[g]
        col.markdown(f"""<div class="card" style="--c:{COLOR[g]}">
            <div class="n">{ICON.get(g,'')} {g}</div>
            <div class="d">{ABOUT.get(g,'')}</div>
            <table>
            <tr><td>Customers</td><td>{int(r.Customers):,}</td></tr>
            <tr><td>Average spend</td><td>{money(r.AvgSpend)}</td></tr>
            <tr><td>Typical income</td><td>{money(r.Income)}</td></tr>
            <tr><td>Said yes to last offer</td><td>{r.YesRate:.0%}</td></tr>
            <tr><td>Website visits / month</td><td>{r.WebVisits:.1f}</td></tr>
            <tr><td>Kids at home</td><td>{r.Kids:.1f}</td></tr>
            </table></div>""", unsafe_allow_html=True)

    st.markdown("### What to do with each group")
    cols = st.columns(len(groups))
    for col, g in zip(cols, groups):
        with col.container(border=True):
            st.markdown(f"**{ICON.get(g,'')} {g}**")
            for t in TIPS.get(g, []):
                st.markdown(f"- {t}")

    st.markdown("### Income vs spending")
    st.caption("Each dot is one customer. Customers with higher income tend to spend more.")
    plot_df = df[df["Income"].fillna(0) <= 200_000]
    fig = px.scatter(plot_df, x="Income", y="Total_Spending", color="Group", color_discrete_map=COLOR,
                     category_orders={"Group": groups}, opacity=0.7,
                     labels={"Income": "Income", "Total_Spending": "Total spending"})
    show(fig, h=430)
    hidden = len(df) - len(plot_df)
    if hidden:
        st.caption(f"{hidden} customer(s) with unusually high income are left out of this chart so it stays readable.")

# ================================================================ 3. PREDICT
with tab3:
    st.markdown("Fill in a customer's details and we will tell you which group they belong to.")

    seg_mean = df.groupby("Group")[INPUTS].mean()
    examples = {"Average customer": MEDIANS[INPUTS].to_dict()}
    for g in groups:
        examples[f"Example: {g}"] = seg_mean.loc[g].to_dict()

    def cast(k, v):
        return float(round(v)) if k == "Income" else int(round(v))

    def load_example():
        for k, v in examples[st.session_state["example"]].items():
            st.session_state[f"in_{k}"] = cast(k, v)

    if "in_Income" not in st.session_state:
        for k, v in examples["Average customer"].items():
            st.session_state[f"in_{k}"] = cast(k, v)

    st.selectbox("Start with an example (then change any number)", list(examples), key="example",
                 on_change=load_example)

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("**About the customer**")
        r = st.columns(3)
        r[0].number_input("Yearly income", 0, 700_000, step=1000, key="in_Income")
        r[1].number_input("Days since last purchase", 0, 99, key="in_Recency")
        r[2].number_input("Website visits per month", 0, 30, key="in_NumWebVisitsMonth")
        st.markdown("**Money spent in the last 2 years**")
        r = st.columns(3)
        for i, k in enumerate(SPEND):
            r[i % 3].number_input(LABEL[k], 0, 3000, key=f"in_{k}")
        st.markdown("**How they buy**")
        r = st.columns(4)
        for i, k in enumerate(BUY):
            r[i].number_input(f"Purchases {LABEL[k].lower()}", 0, 50, key=f"in_{k}")
        r[3].number_input("Past offers accepted (0-5)", 0, 5, key="in_Total_Campaign_Acceptance")

    vals = {k: st.session_state[f"in_{k}"] for k in INPUTS}
    row = dict(vals)
    row["Total_Spending"] = sum(vals[k] for k in SPEND)
    row["Total_Purchases"] = sum(vals[k] for k in BUY)
    xs = scaler.transform(pd.DataFrame([row])[FEATURES])
    grp = NAMES[int(km.predict(xs)[0])]
    dist = km.transform(xs)[0]
    fit = pd.Series(1 / (dist ** 2 + 1e-9), index=[NAMES[i] for i in range(len(dist))])
    fit = (fit / fit.sum()).reindex(groups)
    avg_in_group = df.loc[df["Group"] == grp, "Total_Spending"].mean()

    with right:
        st.markdown(f"""<div class="answer" style="--c:{COLOR[grp]}">
            <div>This customer belongs to</div>
            <div class="v">{ICON.get(grp,'')} {grp}</div>
            <div style="margin-top:.3rem">{ABOUT.get(grp,'')}</div></div>""", unsafe_allow_html=True)
        st.write("")
        diff = row["Total_Spending"] / avg_in_group - 1
        st.metric("Total spending", money(row["Total_Spending"]),
                  delta=f"{diff:+.0%} compared with the average {grp}", delta_color="off", border=True)
        with st.container(border=True):
            st.markdown(f"**What to do with {grp}**")
            for t in TIPS.get(grp, []):
                st.markdown(f"- {t}")

    fig = px.bar(fit.reset_index(), y="index", x=0, orientation="h", color="index",
                 color_discrete_map=COLOR, text_auto=".0%", title="How well this customer fits each group")
    fig.update_xaxes(tickformat=".0%", title="")
    fig.update_yaxes(title="", categoryorder="array", categoryarray=groups[::-1])
    show(fig, h=260, legend=False)
    st.caption("If two groups are close, the customer is on the border between them.")

    with st.expander("📥 Sort many customers at once (upload a file)"):
        st.caption("Upload an Excel or CSV file with the same columns as your customer data.")
        bf = st.file_uploader("Customer file", type=["xlsx", "csv"], key="batch")
        if bf is not None:
            b = read_table(bf.name, bf.getvalue())
            need = {"Income", "Recency", *SPEND, *BUY, "NumWebVisitsMonth"}
            if "Total_Campaign_Acceptance" not in b.columns:
                need |= set(OFFERS)
            miss = sorted(need - set(b.columns))
            if miss:
                st.error("These columns are missing: " + ", ".join(miss))
            else:
                b = add_totals(b)
                b["Group"] = km.predict(scaler.transform(b[FEATURES].fillna(MEDIANS)))
                b["Group"] = b["Group"].map(NAMES)
                st.success(f"Done! {len(b):,} customers sorted.")
                st.dataframe(b["Group"].value_counts().rename("Customers").to_frame(), width="stretch")
                st.download_button("Download results (CSV)", b.to_csv(index=False).encode(),
                                   "customers_with_groups.csv", "text/csv")

# ================================================================ 4. LIST
with tab4:
    pick = st.multiselect("Show only these groups", groups, default=groups)
    view = df[df["Group"].isin(pick)]
    cols = ["ID", "Group", "Age", "Income", "Kids at home", "Total_Spending", "Total_Purchases",
            "Recency", "Response"]
    st.caption(f"{len(view):,} customers. Click a column title to sort.")
    st.dataframe(
        view[cols].rename(columns={"Total_Spending": "Total spending", "Total_Purchases": "Total purchases",
                                   "Recency": "Days since last purchase",
                                   "Response": "Said yes to last offer (1 = yes)"}),
        width="stretch", hide_index=True, height=500)
    st.download_button("⬇️ Download this list (CSV)", view.drop(columns=["Cluster"]).to_csv(index=False).encode(),
                       "customer_groups.csv", "text/csv")
