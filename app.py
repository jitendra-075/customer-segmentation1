import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
from datetime import date

# ============================================================
# PAGE CONFIG & STYLING
# ============================================================
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🧑‍🤝‍🧑",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #00c6ff 100%); background-attachment: fixed; }
.block-container { background: rgba(255,255,255,0.96); border-radius: 20px; padding: 2.5rem; margin-top: 1.5rem; box-shadow: 0 8px 32px rgba(0,0,0,0.25); }

/* Force readable dark text inside the white card (works in Streamlit light AND dark mode) */
.block-container, .block-container * { color: #1e3c72 !important; }
.block-container h1 { font-weight: 800; text-align: center; }
.block-container h2, .block-container h3 { color: #2a5298 !important; }
.block-container button[role="tab"] { font-weight: 700; font-size: 16px; }

section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f2027, #203a43, #2c5364); }
section[data-testid="stSidebar"] * { color: #f0f0f0 !important; }
div.stButton > button { background: linear-gradient(90deg, #00c6ff, #0072ff); color: white; font-weight: 700; font-size: 18px; border-radius: 12px; padding: 0.6rem 1.5rem; border: none; width: 100%; }

/* Cards */
.block-container .result-card { background: linear-gradient(135deg, #11998e, #38ef7d); padding: 1.6rem; border-radius: 18px; text-align: center; margin: 1rem 0; }
.block-container .result-card * { color: #ffffff !important; margin: 0; }
.block-container .deploy-card { background: linear-gradient(135deg, #0f9b6e, #1fd18b); padding: 1.4rem 1.8rem; border-radius: 16px; margin: 0.8rem 0 1.2rem 0; }
.block-container .deploy-card * { color: #ffffff !important; }
.block-container .deploy-card h3 { margin: 0 0 0.3rem 0; }
.block-container .deploy-card p { margin: 0.15rem 0; font-size: 15px; }
.block-container .pill { display:inline-block; background: rgba(255,255,255,0.22); padding: 4px 12px; border-radius: 999px; margin: 6px 6px 0 0; font-weight: 700; font-size: 14px; }
.block-container .note-card { background: #fff7e6; border-left: 5px solid #f5a623; padding: 0.9rem 1.2rem; border-radius: 8px; margin: 0.8rem 0; }
.block-container .note-card * { color: #6b4500 !important; }

/* Tables */
.block-container table.tbl { width: 100%; border-collapse: collapse; font-size: 14.5px; margin: 0.4rem 0 1rem 0; }
.block-container table.tbl th { background: #1e3c72; padding: 9px 12px; text-align: left; color: #ffffff !important; }
.block-container table.tbl td { padding: 8px 12px; border-bottom: 1px solid #dbe4f3; }
.block-container table.tbl tr:nth-child(even) td { background: #f3f7ff; }
.block-container table.tbl tr.hl td { background: #d8f8e8 !important; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD SAVED ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load('segmentation_model.pkl')
    scaler = joblib.load('segmentation_scaler.pkl')
    pca = joblib.load('segmentation_pca.pkl')
    columns = joblib.load('segmentation_columns.pkl')
    reference_date = joblib.load('segmentation_reference_date.pkl')
    cluster_profile = joblib.load('segmentation_cluster_profile.pkl')
    report = joblib.load('segmentation_report.pkl')
    return model, scaler, pca, columns, reference_date, cluster_profile, report

model, scaler, pca, feature_columns, reference_date, cluster_profile, report = load_artifacts()

results_df = report["results"].copy()
viz = report["viz_data"]
deployed_name = report["deployed_model"]

# ============================================================
# HELPERS (naming, tables, charts)
# ============================================================
PALETTE = ["#0072ff", "#11998e", "#f7971e", "#e94560", "#8e44ad", "#16a085", "#7f8c8d", "#2c3e50"]
MNT_LABELS = {"MntWines": "Wine", "MntFruits": "Fruits", "MntMeatProducts": "Meat",
              "MntFishProducts": "Fish", "MntSweetProducts": "Sweets", "MntGoldProds": "Gold"}
CHANNEL_LABELS = {"NumWebPurchases": "Web", "NumCatalogPurchases": "Catalog",
                  "NumStorePurchases": "Store", "NumDealsPurchases": "Deals"}
PROFILE_LABELS = {"Income": "Annual Income ($)", "Age": "Age", "TotalSpend": "Total Spend ($)",
                  "TotalChildren": "Children at home", "TotalPurchases": "Total Purchases",
                  "TotalAcceptedCmp": "Campaigns Accepted", "Recency": "Days Since Last Purchase",
                  "Customer_For": "Customer Tenure (days)"}


def segment_names(profile):
    """Give each cluster a readable name based on how much its customers spend."""
    order = profile["TotalSpend"].sort_values(ascending=False).index.tolist()
    names = {}
    for rank, c in enumerate(order):
        if rank == 0:
            label = "High-Value Spenders"
        elif rank == len(order) - 1:
            label = "Budget-Conscious Shoppers"
        else:
            label = "Mid-Tier Customers"
        names[c] = f"Segment {c} · {label}"
    return names


def color_map(clusters):
    return {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(sorted(clusters))}


def _fmt(v, decimals=3):
    if isinstance(v, (bool, np.bool_)):
        return "✅ Yes" if v else "❌ No"
    if isinstance(v, (float, np.floating)):
        return "—" if np.isnan(v) else f"{v:,.{decimals}f}"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return str(v)


def html_table(df, highlight=None, decimals=3):
    """Render a DataFrame as a clean HTML table (readable on any Streamlit theme)."""
    head = "".join(f"<th>{c}</th>" for c in [df.index.name or ""] + list(df.columns))
    rows = []
    for idx, row in df.iterrows():
        cls = ' class="hl"' if highlight is not None and idx == highlight else ""
        cells = "".join(f"<td>{_fmt(v, decimals)}</td>" for v in row)
        rows.append(f"<tr{cls}><td><b>{idx}</b></td>{cells}</tr>")
    return f"<table class='tbl'><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def style_ax(ax):
    sns.despine(ax=ax)
    ax.grid(axis="y", alpha=0.25)


def fig_model_scores(res, deployed):
    d = res.copy()
    d["Silhouette Score"] = d["Silhouette Score"].fillna(0)
    d["Davies-Bouldin Index"] = d["Davies-Bouldin Index"].fillna(0)
    colors = ["#11998e" if m == deployed else ("#6fa8f5" if dep else "#cfd6e2")
              for m, dep in zip(d["Model"], d["Deployable"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, col, title in [(axes[0], "Silhouette Score", "Silhouette Score (higher = better)"),
                           (axes[1], "Davies-Bouldin Index", "Davies-Bouldin Index (lower = better)")]:
        bars = ax.barh(d["Model"], d[col], color=colors)
        ax.invert_yaxis()
        ax.set_title(title, fontweight="bold")
        for b, raw in zip(bars, res[col]):
            txt = "n/a (1 cluster)" if np.isnan(raw) else f"{raw:.3f}"
            ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2, txt, va="center", fontsize=9)
        ax.set_xlim(0, d[col].max() * 1.3 if d[col].max() > 0 else 1)
        sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def fig_k_selection(k_range, inertias, sils, best_k):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(k_range, inertias, marker="o", color="#0072ff")
    axes[0].set_title("Elbow Method", fontweight="bold")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[1].plot(k_range, sils, marker="o", color="#e94560")
    axes[1].axvline(best_k, color="#11998e", ls="--", label=f"Best k = {best_k}")
    axes[1].set_title("Silhouette Score by k", fontweight="bold")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette")
    axes[1].legend()
    for ax in axes:
        style_ax(ax)
    fig.tight_layout()
    return fig


def fig_pca_scatter(vdf, names, new_point=None):
    cmap = color_map(vdf["Cluster"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for c in sorted(vdf["Cluster"].unique()):
        sub = vdf[vdf["Cluster"] == c]
        ax.scatter(sub["PC1"], sub["PC2"], s=22, alpha=0.55, color=cmap[c], label=names[c])
    if new_point is not None:
        ax.scatter(new_point[0], new_point[1], marker="*", s=420, color="#ffd400",
                   edgecolor="black", linewidth=1.5, label="This customer", zorder=5)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("Customer Segments (PCA projection)", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=9, frameon=False)
    style_ax(ax)
    fig.tight_layout()
    return fig


def fig_segment_sizes(vdf, names):
    counts = vdf["Cluster"].value_counts().sort_index()
    cmap = color_map(counts.index)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.pie(counts.values, labels=[names[c].split(" · ")[0] for c in counts.index],
           colors=[cmap[c] for c in counts.index], autopct="%1.1f%%", startangle=90,
           wedgeprops=dict(width=0.45, edgecolor="white"), textprops={"fontsize": 11})
    ax.set_title("Customers per Segment", fontweight="bold")
    fig.tight_layout()
    return fig


def fig_grouped_means(vdf, cols, labels, names, title, ylabel):
    means = vdf.groupby("Cluster")[cols].mean()
    cmap = color_map(means.index)
    x = np.arange(len(cols))
    w = 0.8 / len(means)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for i, c in enumerate(means.index):
        ax.bar(x + i * w - 0.4 + w / 2, means.loc[c].values, w, color=cmap[c], label=names[c])
    ax.set_xticks(x)
    ax.set_xticklabels([labels[c] for c in cols])
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    style_ax(ax)
    fig.tight_layout()
    return fig


def fig_income_spend(vdf, names):
    cmap = color_map(vdf["Cluster"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for c in sorted(vdf["Cluster"].unique()):
        sub = vdf[vdf["Cluster"] == c]
        ax.scatter(sub["Income"], sub["TotalSpend"], s=20, alpha=0.5, color=cmap[c], label=names[c])
    ax.set_xlabel("Annual Income ($)")
    ax.set_ylabel("Total Spend ($)")
    ax.set_title("Income vs Total Spend", fontweight="bold")
    ax.legend(fontsize=9)
    style_ax(ax)
    fig.tight_layout()
    return fig


def fig_boxes(vdf):
    cmap = color_map(vdf["Cluster"].unique())
    order = sorted(vdf["Cluster"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, col, title in zip(axes, ["Income", "TotalSpend", "TotalPurchases"],
                              ["Income", "Total Spend", "Total Purchases"]):
        data = [vdf.loc[vdf["Cluster"] == c, col].values for c in order]
        bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color="black"))
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels([f"Seg {c}" for c in order])
        for patch, c in zip(bp["boxes"], order):
            patch.set_facecolor(cmap[c])
            patch.set_alpha(0.7)
        ax.set_title(f"{title} by Segment", fontweight="bold")
        style_ax(ax)
    fig.tight_layout()
    return fig


def fig_profile_heatmap(profile):
    p = profile.rename(columns=PROFILE_LABELS)
    span = (p.max() - p.min()).replace(0, 1)
    norm = (p - p.min()) / span
    fig, ax = plt.subplots(figsize=(11, 1.4 + 0.7 * len(p)))
    sns.heatmap(norm, annot=p, fmt=",.1f", cmap="YlGnBu", cbar=False, linewidths=2,
                linecolor="white", ax=ax, annot_kws={"fontsize": 11})
    ax.set_ylabel("Segment")
    ax.set_title("Segment Profile Heatmap (colour = relative level within each column)", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    return fig


def fig_customer_vs_segment(spend_dict, seg_means, seg_label):
    cats = list(spend_dict.keys())
    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.bar(x - 0.2, [spend_dict[c] for c in cats], 0.4, color="#ffb300", label="This customer")
    ax.bar(x + 0.2, [seg_means[c] for c in cats], 0.4, color="#0072ff", label=f"{seg_label} (avg)")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Spend ($)")
    ax.set_title("Spending: this customer vs segment average", fontweight="bold")
    ax.legend()
    style_ax(ax)
    fig.tight_layout()
    return fig


seg_names = segment_names(cluster_profile)

# ============================================================
# HEADER
# ============================================================
st.markdown("<h1>🧑‍🤝‍🧑 Customer Segmentation Dashboard</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Explore how our customers group together, compare the clustering models "
    "we trained, and find the segment of any new customer.</p>",
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR INPUTS
# ============================================================
st.sidebar.header("📋 Customer Profile")

birth_year = st.sidebar.number_input("Birth Year", min_value=1930, max_value=date.today().year, value=1980)
education = st.sidebar.selectbox("Education", ["Undergraduate", "Graduate", "Postgraduate"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Partnered"])
income = st.sidebar.number_input("Annual Income ($)", min_value=0, max_value=600000, value=50000, step=1000)
kidhome = st.sidebar.slider("Children at home", 0, 3, 0)
teenhome = st.sidebar.slider("Teenagers at home", 0, 3, 0)
customer_since = st.sidebar.date_input("Customer Since", value=date(2013, 1, 1))
recency = st.sidebar.slider("Days since last purchase", 0, 100, 30)
complain = st.sidebar.radio("Complained in last 2 years?", ["No", "Yes"], horizontal=True)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Spending (last 2 years)")
mnt_wines = st.sidebar.number_input("Wine ($)", 0, 3000, 300)
mnt_fruits = st.sidebar.number_input("Fruits ($)", 0, 500, 30)
mnt_meat = st.sidebar.number_input("Meat ($)", 0, 3000, 200)
mnt_fish = st.sidebar.number_input("Fish ($)", 0, 500, 50)
mnt_sweet = st.sidebar.number_input("Sweets ($)", 0, 500, 30)
mnt_gold = st.sidebar.number_input("Gold ($)", 0, 500, 50)

st.sidebar.markdown("---")
st.sidebar.subheader("🛒 Purchase Channels")
num_web = st.sidebar.slider("Web purchases", 0, 30, 4)
num_catalog = st.sidebar.slider("Catalog purchases", 0, 30, 2)
num_store = st.sidebar.slider("Store purchases", 0, 30, 5)
num_deals = st.sidebar.slider("Deal purchases", 0, 15, 2)
web_visits = st.sidebar.slider("Web visits/month", 0, 20, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("📣 Campaign Response")
accepted_cmps = st.sidebar.multiselect("Accepted past campaigns", ["1", "2", "3", "4", "5", "Last Campaign"])

predict_btn = st.sidebar.button("🔮 Find Segment")

# ============================================================
# BUILD FEATURE ROW
# ============================================================
def build_input_row():
    age = reference_date.year - birth_year
    customer_for = (reference_date - pd.Timestamp(customer_since)).days
    total_spend = mnt_wines + mnt_fruits + mnt_meat + mnt_fish + mnt_sweet + mnt_gold
    total_children = kidhome + teenhome
    family_size = (2 if marital_status == "Partnered" else 1) + total_children
    total_purchases = num_web + num_catalog + num_store + num_deals
    total_accepted = len(accepted_cmps)

    row = {
        "Income": income, "Kidhome": kidhome, "Teenhome": teenhome,
        "Recency": recency, "Complain": 1 if complain == "Yes" else 0,
        "NumWebVisitsMonth": web_visits, "Age": age, "Customer_For": customer_for,
        "TotalSpend": total_spend, "TotalChildren": total_children,
        "Family_Size": family_size, "TotalPurchases": total_purchases,
        "TotalAcceptedCmp": total_accepted,
        "Education_Postgraduate": education == "Postgraduate",
        "Education_Undergraduate": education == "Undergraduate",
        "Marital_Status_Single": marital_status == "Single",
    }
    input_df = pd.DataFrame([row]).reindex(columns=feature_columns, fill_value=0)
    scaled = scaler.transform(input_df)
    reduced = pca.transform(scaled)

    summary = {
        "Income": income, "Age": age, "TotalSpend": total_spend,
        "TotalChildren": total_children, "TotalPurchases": total_purchases,
        "TotalAcceptedCmp": total_accepted, "Recency": recency, "Customer_For": customer_for,
    }
    spend = {"Wine": mnt_wines, "Fruits": mnt_fruits, "Meat": mnt_meat,
             "Fish": mnt_fish, "Sweets": mnt_sweet, "Gold": mnt_gold}
    return reduced, summary, spend

# ============================================================
# PREDICTION (kept in session_state so it survives sidebar changes)
# ============================================================
if predict_btn:
    reduced, summary, spend = build_input_row()
    try:
        cluster = int(model.predict(reduced)[0])
        st.session_state["result"] = {
            "cluster": cluster, "summary": summary, "spend": spend,
            "point": (float(reduced[0, 0]), float(reduced[0, 1])),
        }
    except AttributeError:
        st.session_state.pop("result", None)
        st.error("The saved model doesn't support predicting new customers directly.")

# ============================================================
# DEPLOYMENT BANNER
# ============================================================
dep_row = results_df[results_df["Model"] == deployed_name].iloc[0]
st.markdown(f"""
<div class="deploy-card">
  <h3>✅ Best deployable model: {deployed_name} — trained, saved &amp; deployed</h3>
  <p>Out of {len(results_df)} clustering algorithms we compared, <b>{deployed_name}</b> is the best one that can
  assign <i>new</i> customers to a segment. It is saved as <code>segmentation_model.pkl</code> and is the model powering this app.</p>
  <span class="pill">Silhouette {dep_row['Silhouette Score']:.3f}</span>
  <span class="pill">Davies-Bouldin {dep_row['Davies-Bouldin Index']:.3f}</span>
  <span class="pill">Calinski-Harabasz {dep_row['Calinski-Harabasz']:,.0f}</span>
  <span class="pill">{int(dep_row['Clusters Found'])} segments</span>
  <span class="pill">{report['n_customers']:,} customers</span>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_eval, tab_insights = st.tabs(["🔮 Predict Segment", "📈 Model Evaluation", "📊 Segment Insights"])

# ------------------------------------------------------------
# TAB 1 — PREDICT
# ------------------------------------------------------------
with tab_predict:
    result = st.session_state.get("result")
    if result is None:
        st.info("👈 Fill in the customer's details in the sidebar and click **Find Segment**.")
    else:
        c = result["cluster"]
        st.markdown(f"""
        <div class="result-card">
            <p>This customer belongs to</p>
            <h2>{seg_names.get(c, f'Segment {c}')}</h2>
        </div>
        """, unsafe_allow_html=True)

        seg_avg = cluster_profile.loc[c]
        s = result["summary"]

        m = st.columns(4)
        m[0].metric("Total Spend", f"${s['TotalSpend']:,.0f}", f"{s['TotalSpend'] - seg_avg['TotalSpend']:+,.0f} vs segment avg")
        m[1].metric("Income", f"${s['Income']:,.0f}", f"{s['Income'] - seg_avg['Income']:+,.0f} vs segment avg")
        m[2].metric("Total Purchases", f"{s['TotalPurchases']}", f"{s['TotalPurchases'] - seg_avg['TotalPurchases']:+.1f} vs segment avg")
        m[3].metric("Campaigns Accepted", f"{s['TotalAcceptedCmp']}", f"{s['TotalAcceptedCmp'] - seg_avg['TotalAcceptedCmp']:+.1f} vs segment avg")

        left, right = st.columns(2)
        with left:
            st.pyplot(fig_pca_scatter(viz, seg_names, new_point=result["point"]))
        with right:
            seg_spend = viz[viz["Cluster"] == c][list(MNT_LABELS.keys())].mean()
            seg_spend.index = [MNT_LABELS[k] for k in seg_spend.index]
            st.pyplot(fig_customer_vs_segment(result["spend"], seg_spend,
                                              seg_names.get(c, f"Segment {c}").split(" · ")[0]))

        st.subheader("🧾 This customer vs the segment average")
        comp = pd.DataFrame({
            "This Customer": pd.Series(s).astype(float),
            "Segment Average": seg_avg[list(s.keys())].astype(float),
        })
        comp["Difference"] = comp["This Customer"] - comp["Segment Average"]
        comp.index = [PROFILE_LABELS[i] for i in comp.index]
        comp.index.name = "Metric"
        st.markdown(html_table(comp, decimals=1), unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB 2 — MODEL EVALUATION
# ------------------------------------------------------------
with tab_eval:
    st.subheader("🏆 Model comparison")
    st.markdown(
        "Customer segmentation is **unsupervised learning** — there is no correct label to compare against, so "
        "classic *accuracy* does not apply. Instead we judge how well-separated and compact the clusters are:"
    )
    st.markdown(
        "- **Silhouette Score** (−1 to 1): how well each customer fits its own cluster vs the nearest other one. *Higher is better.*\n"
        "- **Davies-Bouldin Index**: average similarity between clusters. *Lower is better.*\n"
        "- **Calinski-Harabasz**: ratio of between-cluster to within-cluster spread. *Higher is better.*\n"
        "- **Noise Points (%)**: customers a density-based model refuses to assign to any cluster."
    )

    table = results_df.copy()
    table["Status"] = np.where(
        table["Model"] == deployed_name, "🏆 Saved & Deployed",
        np.where(table["Deployable"], "Can predict new customers", "Cannot predict new customers"))
    table = table.drop(columns=["Deployable"]).set_index("Model")
    st.markdown(html_table(table, highlight=deployed_name), unsafe_allow_html=True)

    st.pyplot(fig_model_scores(results_df, deployed_name))
    st.caption("🟢 Deployed model · 🔵 other models that can predict new customers · ⚪ models that cannot predict new customers.")

    top = results_df.sort_values("Silhouette Score", ascending=False).iloc[0]
    if top["Model"] != deployed_name:
        st.markdown(f"""
        <div class="note-card">
        <b>Why {deployed_name} and not {top['Model']}?</b><br>
        {top['Model']} has the highest silhouette ({top['Silhouette Score']:.3f}), but it labels
        <b>{top['Noise Points (%)']:.0f}% of customers as noise</b> (no segment), its silhouette is computed only on the
        remaining points, and it cannot assign new customers to a segment. {deployed_name} segments
        <b>100% of customers</b>, has the highest Calinski-Harabasz score ({dep_row['Calinski-Harabasz']:,.0f}) and the best silhouette among
        deployable models — so it is the best model we could actually save and deploy.
        </div>
        """, unsafe_allow_html=True)

    st.subheader("🎯 How we chose the number of segments")
    st.pyplot(fig_k_selection(report["k_range"], report["inertias"], report["silhouette_by_k"], report["best_k"]))
    st.caption(f"The silhouette score peaked at k = {report['best_k']}, so we used {report['best_k']} segments for the centroid-based models.")

    st.subheader("⚙️ Pipeline")
    st.markdown(
        f"1. **Cleaned** the data (missing income imputed, outliers and duplicates removed) → {report['n_customers']:,} customers\n"
        f"2. **Engineered** features (age, tenure, total spend, family size, total purchases, campaigns accepted) → {report['n_features']} model inputs\n"
        f"3. **Scaled** with `StandardScaler` and reduced with **PCA** (3 components keep {report['pca_variance']*100:.1f}% of the variance)\n"
        f"4. **Trained & compared** {len(results_df)} clustering algorithms\n"
        f"5. **Saved** the best deployable model ({deployed_name}) with its scaler, PCA and metadata as `.pkl` files\n"
        f"6. **Deployed** it in this Streamlit app"
    )

# ------------------------------------------------------------
# TAB 3 — SEGMENT INSIGHTS
# ------------------------------------------------------------
with tab_insights:
    st.subheader("🗂️ Segment profiles")
    prof_tbl = cluster_profile.rename(columns=PROFILE_LABELS).copy()
    prof_tbl.insert(0, "Segment Name", [seg_names[i].split(" · ")[1] for i in prof_tbl.index])
    prof_tbl.insert(1, "Customers", viz["Cluster"].value_counts().reindex(prof_tbl.index).values)
    prof_tbl.index = [f"Segment {i}" for i in prof_tbl.index]
    prof_tbl.index.name = "Segment"
    st.markdown(html_table(prof_tbl, decimals=1), unsafe_allow_html=True)

    st.pyplot(fig_profile_heatmap(cluster_profile))

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(fig_pca_scatter(viz, seg_names))
    with c2:
        st.pyplot(fig_segment_sizes(viz, seg_names))

    c3, c4 = st.columns(2)
    with c3:
        st.pyplot(fig_grouped_means(viz, list(MNT_LABELS.keys()), MNT_LABELS, seg_names,
                                    "Average spend by product category", "Avg spend ($)"))
    with c4:
        st.pyplot(fig_grouped_means(viz, list(CHANNEL_LABELS.keys()), CHANNEL_LABELS, seg_names,
                                    "Average purchases by channel", "Avg purchases"))

    c5, c6 = st.columns(2)
    with c5:
        st.pyplot(fig_income_spend(viz, seg_names))
    with c6:
        st.pyplot(fig_boxes(viz))
