import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(page_title="Telecom Retention Engine", layout="wide")

# ... imports remain the same ...

# ==========================================
# 1. CACHED DATA & MODEL PIPELINE
# ==========================================
# We use @st.cache so the model doesn't retrain every time you move a slider.
@st.cache_resource(show_spinner=False)
def load_and_train_model():
    # START: Use st.status for a stable loading UI
    with st.status("🚀 Training AI Model & Running Simulations...", expanded=True) as status:

        # --- A. Load Data ---
        st.write("📂 Loading Dataset...")
        df = pd.read_csv("telecom_churn_data.csv")

        # --- B. Preprocessing ---
        st.write("🧹 Cleaning & Imputing Missing Values...")
        rech_cols = [col for col in df.columns if 'rech' in col or 'mou' in col or 'vol' in col or 'arpu' in col]
        df[rech_cols] = df[rech_cols].fillna(0)

        cat_cols = ['fb_user_6', 'fb_user_7', 'fb_user_8', 'night_pck_user_6', 'night_pck_user_7', 'night_pck_user_8']
        existing_cat = [c for c in cat_cols if c in df.columns]
        df[existing_cat] = df[existing_cat].fillna(0)

        missing_percent = df.isnull().mean() * 100
        df.drop(columns=missing_percent[missing_percent > 70].index, inplace=True)

        # --- C. Pareto Optimization ---
        st.write("⚙️ Running Pareto Optimization...")
        df['total_rev_6_7'] = df['total_rech_amt_6'] + df['total_rech_amt_7']
        sorted_rev = df.sort_values('total_rev_6_7', ascending=False)['total_rev_6_7']
        cum_rev = sorted_rev.cumsum() / sorted_rev.sum() * 100
        cum_pop = np.arange(1, len(sorted_rev) + 1) / len(sorted_rev) * 100
        optimal_idx = np.argmax(cum_rev - cum_pop)
        optimal_pop_pct = cum_pop[optimal_idx]

        cutoff_val = df['total_rev_6_7'].quantile(1 - (optimal_pop_pct / 100))
        df_high_val = df[df['total_rev_6_7'] >= cutoff_val].copy()

        df_high_val['churn'] = np.where(
            (df_high_val['total_ic_mou_9'] == 0) & (df_high_val['total_og_mou_9'] == 0) &
            (df_high_val['vol_2g_mb_9'] == 0) & (df_high_val['vol_3g_mb_9'] == 0), 1, 0
        )
        cols_9 = [col for col in df_high_val.columns if '_9' in col]
        df_final = df_high_val.drop(columns=cols_9)

        # --- D. Feature Engineering ---
        st.write("🧠 Engineering Features...")
        metrics = ['arpu', 'total_og_mou', 'total_ic_mou', 'total_rech_amt']
        for metric in metrics:
            col_8, col_6, col_7 = f"{metric}_8", f"{metric}_6", f"{metric}_7"
            if col_8 in df_final.columns:
                df_final[f'{metric}_diff'] = df_final[col_8] - ((df_final[col_6] + df_final[col_7]) / 2)

        # --- E. Modeling ---
        st.write("🤖 Training Logistic Regression & Random Forest...")
        df_model = df_final.select_dtypes(include=[np.number]).fillna(0)
        X = df_model.drop(columns=['churn', 'mobile_number', 'circle_id'], errors='ignore')
        y = df_model['churn']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pca = PCA(n_components=0.95, random_state=42)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
        lr.fit(X_train_pca, y_train)

        y_prob = lr.predict_proba(X_test_pca)[:, 1]
        arpu_test = df_final.loc[y_test.index, 'arpu_8'].values

        rf = RandomForestClassifier(class_weight='balanced', n_estimators=50, random_state=42)
        rf.fit(X_train, y_train)
        importances = rf.feature_importances_

        # Update final status
        status.update(label="✅ System Ready", state="complete", expanded=False)

    return y_test, y_prob, arpu_test, df_final, importances, X.columns

# ... rest of the code remains the same ...

# Load Data (Only runs once)
y_test, y_prob, arpu_test, df_final, importances, feature_names = load_and_train_model()

# ==========================================
# 2. THE FRONTEND UI (Streamlit)
# ==========================================

# --- Sidebar: Business Assumptions ---
st.sidebar.header("⚙️ Simulation Controls")
st.sidebar.markdown("Adjust these sliders to update the strategy.")

cost_input = st.sidebar.slider("Cost of Retention Offer (% of ARPU)", 0.05, 0.90, 0.10, 0.01)
success_input = st.sidebar.slider("Offer Acceptance Rate", 0.10, 0.90, 0.50, 0.05)
ltv_input = st.sidebar.slider("Months of Revenue Saved (LTV)", 1, 12, 6, 1)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Logic:** \n"
    "Profit = (Saved Revenue) - (Cost of Offers)\n"
    "*Optimizing for ROI, not just Accuracy.*"
)

# --- Main Page ---
st.title("📡 Telecom Retention AI Engine")
st.markdown("### Interactive Profit Optimization & Churn Strategy")

# 1. High-Level Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total High-Value Customers", f"{len(df_final):,}")
col2.metric("Churn Rate (Imbalance)", f"{df_final['churn'].mean()*100:.1f}%")
col3.metric("Features Analyzed", f"{len(feature_names)}")

st.markdown("---")

# 2. The Financial Simulation (Dynamic)
st.subheader("💰 ROI Optimization Simulator")

# Run Simulation Loop based on SLIDER inputs
thresholds = np.arange(0.1, 0.9, 0.05)
profits = []

eval_df = pd.DataFrame({'Actual': y_test, 'Prob': y_prob, 'ARPU': arpu_test})

for t in thresholds:
    eval_df['Pred'] = (eval_df['Prob'] > t).astype(int)
    cost = eval_df[eval_df['Pred'] == 1]['ARPU'].sum() * cost_input
    saved_revenue = eval_df[(eval_df['Pred'] == 1) & (eval_df['Actual'] == 1)]['ARPU'].sum()
    gain = saved_revenue * ltv_input * success_input
    profits.append(gain - cost)

best_idx = np.argmax(profits)
best_t = thresholds[best_idx]
max_profit = profits[best_idx]

# Plotting the Curve
fig_roi, ax = plt.subplots(figsize=(10, 4))
ax.plot(thresholds, profits, marker='o', color='#2ecc71', linewidth=2)
ax.axvline(best_t, color='#e74c3c', linestyle='--', label=f'Optimal: {best_t:.2f}')
ax.set_title(f"Projected Monthly Profit: INR {max_profit:,.0f}", fontsize=14, fontweight='bold')
ax.set_xlabel("Probability Threshold")
ax.set_ylabel("Net Profit (INR)")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig_roi)

st.success(f"✅ **Recommendation:** Set trigger threshold at **{best_t:.2f}**. This maximizes profit at **INR {max_profit:,.0f}**.")

# 3. Strategy Segmentation (Dynamic Clustering)
st.markdown("---")
st.subheader("🎯 Intelligent Customer Segmentation")
st.markdown("Analyzing High-Risk customers to generate targeted actions.")

# Filter High Risk based on NEW Threshold
high_risk_indices = eval_df[eval_df['Prob'] > best_t].index
seg_cols = ['arpu_8', 'total_og_mou_8', 'total_ic_mou_8', 'vol_3g_mb_8', 'aon']
if len(high_risk_indices) > 0:
    X_seg = df_final.loc[y_test.index].loc[high_risk_indices, seg_cols].fillna(0)
    scaler_seg = StandardScaler()
    X_seg_scaled = scaler_seg.fit_transform(X_seg)

    # K-Means (Fixed k=4 for Demo Stability)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_seg_scaled)
    X_seg['Cluster'] = clusters
    profile = X_seg.groupby('Cluster').mean()

    # Heatmap
    fig_heat, ax2 = plt.subplots(figsize=(10, 5))
    sns.heatmap(profile.T, annot=True, cmap='Blues', fmt='.1f', ax=ax2)
    ax2.set_title("Customer Personas (Mean Values)")
    st.pyplot(fig_heat)

    st.markdown("""
    **Strategy Key:**
    * **High Data Usage:** Offer Speed Upgrade.
    * **High Tenure (AON):** Offer Loyalty Rewards (Prevent Fatigue).
    * **High Voice Usage:** Offer Unlimited Calling.
    * **Low Usage:** Offer Validity Extension.
    """)
else:
    st.warning("Threshold too high! No customers selected for segmentation. Lower the threshold.")

# 4. Explainable AI (Drivers)
st.markdown("---")
with st.expander("🔍 See Top Churn Drivers"):
    indices = np.argsort(importances)[::-1][:10]
    top_features = [feature_names[i] for i in indices]
    fig_feat, ax3 = plt.subplots(figsize=(8, 4))
    ax3.barh(range(10), importances[indices][::-1], color='#3498db')
    ax3.set_yticks(range(10))
    ax3.set_yticklabels(top_features[::-1])
    ax3.set_xlabel("Impact Score")
    st.pyplot(fig_feat)