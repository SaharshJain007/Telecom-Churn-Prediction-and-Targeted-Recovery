import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, recall_score, roc_curve, auc, precision_recall_curve

# Set professional style
sns.set(style="whitegrid")

# ==========================================
# PART 1: Data Preparation & Optimization
# ==========================================
print("--- Step 1: Loading and Preprocessing Data ---")
try:
    df = pd.read_excel(r'C:\Users\Dell\Downloads\telecom_churn_data.xlsx')
except:
    print("Excel not found, using CSV fallback...")
    df = pd.read_csv('telecom_churn_data.csv')

# ---------------------------------------------------------
# CRITICAL FIX: Robust Imputation before Dropping
# ---------------------------------------------------------
# 1. Identify Usage Columns (Numerical)
rech_cols = [col for col in df.columns if 'rech' in col or 'mou' in col or 'vol' in col or 'arpu' in col]
df[rech_cols] = df[rech_cols].fillna(0)

# 2. Identify Categorical Indicators (FB/Night Pack)
# We treat NaN as 0 (Not a user) so we don't drop these valuable signals
cat_cols = ['fb_user_6', 'fb_user_7', 'fb_user_8', 'night_pck_user_6', 'night_pck_user_7', 'night_pck_user_8']
# Only fill if they exist in the dataset
existing_cat_cols = [c for c in cat_cols if c in df.columns]
df[existing_cat_cols] = df[existing_cat_cols].fillna(0)

# 3. Drop Sparse Columns (>70% Missing)
# Now safe because meaningful zeros are already filled
missing_percent = df.isnull().mean() * 100
cols_to_drop = missing_percent[missing_percent > 70].index
df.drop(columns=cols_to_drop, inplace=True)
print(f"Dropped {len(cols_to_drop)} sparse columns (mostly dates/IDs).")

# ---------------------------------------------------------
# OPTIMIZATION 1: High-Value Cutoff (Lorenz Curve)
# ---------------------------------------------------------
print("--- Step 1.5: Optimizing High-Value Cutoff (Lorenz Curve) ---")
df['total_rev_6_7'] = df['total_rech_amt_6'] + df['total_rech_amt_7']
total_revenue = df['total_rev_6_7'].sum()

# Sort & Calculate Cumulative
sorted_rev = df.sort_values('total_rev_6_7', ascending=False)['total_rev_6_7']
cum_revenue = sorted_rev.cumsum() / total_revenue * 100
cum_population = np.arange(1, len(sorted_rev) + 1) / len(sorted_rev) * 100

# OPTIMIZATION: Max Separation Point
optimal_idx = np.argmax(cum_revenue - cum_population)
optimal_pop_pct = cum_population[optimal_idx]

print(f"\n[Auto-Optimization 1: Revenue]")
print(f"Optimal High-Value Cutoff: Top {optimal_pop_pct:.1f}%")

# CHART 1: Pareto (Optimized)
plt.figure(figsize=(8, 5))
plt.plot(cum_population, cum_revenue, color='#2c3e50', linewidth=2, label='Lorenz Curve')
plt.plot([0, 100], [0, 100], color='gray', linestyle='--', label='Equality Line')
plt.scatter(optimal_pop_pct, cum_revenue.iloc[optimal_idx], color='#e74c3c', zorder=5)
plt.axvline(optimal_pop_pct, color='#e74c3c', linestyle='--', label=f'Optimal Cutoff ({optimal_pop_pct:.1f}%)')
plt.title(f'Chart 1: Optimized Revenue Cutoff (Top {optimal_pop_pct:.1f}%)')
plt.xlabel('% Customers'); plt.ylabel('% Revenue')
plt.legend(); plt.show()

# Apply Optimal Filter
cutoff_value = df['total_rev_6_7'].quantile(1 - (optimal_pop_pct / 100))
df_high_val = df[df['total_rev_6_7'] >= cutoff_value].copy()
print(f"High-Value Customers Selected: {df_high_val.shape[0]} rows")

# Tag Churners (Month 9 Usage)
df_high_val['churn'] = np.where(
    (df_high_val['total_ic_mou_9'] == 0) &
    (df_high_val['total_og_mou_9'] == 0) &
    (df_high_val['vol_2g_mb_9'] == 0) &
    (df_high_val['vol_3g_mb_9'] == 0),
    1, 0
)

# Remove Churn Phase
cols_9 = [col for col in df_high_val.columns if '_9' in col]
df_final = df_high_val.drop(columns=cols_9)

# ==========================================
# PART 2: Deep Dive EDA (Visuals Only)
# ==========================================
print("--- Step 2: Generating Visual Insights ---")

# Chart 2: Churn Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='churn', data=df_final, palette='pastel')
plt.title('Chart 2: Churn Distribution')
plt.show()

# Chart 3: Loyalty Analysis (KDE)
plt.figure(figsize=(8, 5))
sns.kdeplot(df_final[df_final['churn'] == 0]['aon'], label='Retained', fill=True, color='green', alpha=0.3)
sns.kdeplot(df_final[df_final['churn'] == 1]['aon'], label='Churned', fill=True, color='red', alpha=0.3)
plt.title('Chart 3: Impact of Loyalty (Age on Network)')
plt.xlabel('Tenure (Days)'); plt.legend(); plt.show()

# Chart 4: Usage Map (Voice vs Data)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='total_og_mou_8', y='vol_3g_mb_8', hue='churn', data=df_final,
                alpha=0.6, palette={0:'#95a5a6', 1:'#e74c3c'}, legend=False)
plt.title('Chart 4: Usage Map (Voice vs Data) - Where do Churners sit?')
plt.xlim(0, 3000); plt.ylim(0, 3000)
plt.show()

# Chart 5: Lifecycle Crash Grid (15 Plots)
metrics = ['arpu', 'total_og_mou', 'total_ic_mou', 'vol_2g_mb', 'vol_3g_mb']
plt.figure(figsize=(15, 12))
plt.suptitle('Chart 5: Customer Lifecycle & Usage Crash in Month 8', fontsize=16)
for i, metric in enumerate(metrics):
    for j, month in enumerate([6, 7, 8]):
        col_name = f"{metric}_{month}"
        plt.subplot(5, 3, i*3 + j + 1)
        sns.boxplot(x='churn', y=col_name, data=df_final, showfliers=False, palette="Set2")
        plt.title(col_name); plt.xlabel('')
plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

# Chart 6: Correlation Heatmap
numeric_df = df_final.select_dtypes(include=[np.number])
corr_with_churn = numeric_df.corr()['churn']
# Top 10 Pos and Top 10 Neg
top_cols = list(set(corr_with_churn.nlargest(10).index.tolist() + corr_with_churn.nsmallest(10).index.tolist()))
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(numeric_df[top_cols].corr(), dtype=bool))
sns.heatmap(numeric_df[top_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', mask=mask)
plt.title('Chart 6: Correlation Heatmap (Top Predictors Only)')
plt.show()

# ==========================================
# PART 3: Feature Engineering & Modelling
# ==========================================
print("--- Step 3: Feature Engineering & Modelling ---")
metrics = ['arpu', 'total_og_mou', 'total_ic_mou', 'total_rech_amt']
for metric in metrics:
    col_8, col_6, col_7 = f"{metric}_8", f"{metric}_6", f"{metric}_7"
    if col_8 in df_final.columns:
        df_final[f'{metric}_diff'] = df_final[col_8] - ((df_final[col_6] + df_final[col_7]) / 2)

df_model = df_final.select_dtypes(include=[np.number]).drop(columns=['mobile_number', 'circle_id'], errors='ignore').fillna(0)
X = df_model.drop('churn', axis=1)
y = df_model['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# PCA + Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr.fit(X_train_pca, y_train)
y_prob = lr.predict_proba(X_test_pca)[:, 1]

# ---------------------------------------------------------
# OPTIMIZATION 2: Financial Threshold Tuning
# ---------------------------------------------------------
print("\n--- Auto-Optimization 2: Financial Threshold Tuning ---")
# Business Assumptions (Configurable)
COST_OF_OFFER = 0.10   # 10% Discount cost
ACCEPTANCE_RATE = 0.50 # 50% success rate
LTV_MULT = 6           # 6 months revenue save

eval_df = pd.DataFrame({'Actual': y_test, 'Prob': y_prob})
eval_df['ARPU'] = df_final.loc[y_test.index, 'arpu_8'].values
thresholds = np.arange(0.1, 0.9, 0.05)
profits = []

for t in thresholds:
    eval_df['Pred'] = (eval_df['Prob'] > t).astype(int)
    # Cost: We pay for every offer sent
    cost = eval_df[eval_df['Pred'] == 1]['ARPU'].sum() * COST_OF_OFFER
    # Gain: Revenue saved from True Positives
    saved_revenue = eval_df[(eval_df['Pred'] == 1) & (eval_df['Actual'] == 1)]['ARPU'].sum()
    gain = saved_revenue * LTV_MULT * ACCEPTANCE_RATE
    profits.append(gain - cost)

best_t = thresholds[np.argmax(profits)]
print(f"Optimal Probability Threshold: {best_t:.2f}")
print(f"Projected ROI at this threshold: INR {max(profits):,.0f}")

# Final Predictions
y_pred_opt = (y_prob > best_t).astype(int)
print("\nModel Accuracy (at Optimized Threshold):")
print(classification_report(y_test, y_pred_opt))

# Chart 7: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('Chart 7: ROC Curve'); plt.legend(); plt.show()

# Chart 8: Precision-Recall Curve (Better for Imbalance)
precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, color='#8e44ad', lw=2, label=f'PR AUC = {pr_auc:.2f}')
plt.title('Chart 8: Precision-Recall Curve (Imbalance Robust)')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend(); plt.show()

# Chart 9: Profit Optimization Curve
plt.figure(figsize=(8, 5))
plt.plot(thresholds, profits, marker='o', color='green')
plt.axvline(best_t, color='red', linestyle='--', label=f'Optimal Threshold {best_t:.2f}')
plt.title('Chart 9: Financial Optimization (Profit vs Probability Threshold)')
plt.xlabel('Churn Probability Threshold')
plt.ylabel('Projected Profit (INR)')
plt.legend(); plt.show()

# ==========================================
# PART 4: Drivers & Strategy
# ==========================================
print("--- Recommendation: Drivers & Clustering ---")

# Feature Importance
rf = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
top_indices = np.argsort(importances)[::-1][:10]
marketing_map = {
    'total_ic_mou_8': 'Social Connectivity (Incoming)',
    'loc_ic_mou_8': 'Local Community (Incoming)',
    'total_og_mou_8': 'Voice Usage Level',
    'arpu_diff': 'Revenue Degradation Rate',
    'total_rech_amt_diff': 'Spending Drop Velocity',
    'vol_3g_mb_8': 'High-Speed Data Usage',
    'aon': 'Loyalty (Tenure)'
}
top_features = [X.columns[i] for i in top_indices]
top_labels = [marketing_map.get(f, f) for f in top_features]

# Chart 10: Marketing Drivers
plt.figure(figsize=(10, 6))
plt.title("Chart 10: Key Marketing Drivers (Feature Importance)")
plt.barh(range(10), importances[top_indices][::-1], color='#3498db', align="center")
plt.yticks(range(10), top_labels[::-1])
plt.xlabel("Impact Score"); plt.show()

# ---------------------------------------------------------
# OPTIMIZATION 3: Elbow Method for K
# ---------------------------------------------------------
print("--- Auto-Optimizing Segments (Elbow) ---")
high_risk_indices = eval_df[eval_df['Prob'] > best_t].index
seg_cols = ['arpu_8', 'total_og_mou_8', 'total_ic_mou_8', 'vol_3g_mb_8', 'aon']
X_seg = df_final.loc[high_risk_indices, seg_cols].fillna(0)
X_seg_scaled = StandardScaler().fit_transform(X_seg)

inertia = []
K_range = range(2, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_seg_scaled)
    inertia.append(km.inertia_)

# Auto-Detect Elbow
p1 = np.array([K_range[0], inertia[0], 0])
p2 = np.array([K_range[-1], inertia[-1], 0])
optimal_k = 3
max_dist = 0
for i, k in enumerate(K_range):
    p = np.array([k, inertia[i], 0])
    dist = np.linalg.norm(np.cross(p2-p1, p1-p)) / np.linalg.norm(p2-p1)
    if dist > max_dist:
        max_dist = dist
        optimal_k = k

print(f"Optimal Clusters detected: k={optimal_k}")

# Chart 11: Elbow Method
plt.figure(figsize=(6, 4))
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.axvline(optimal_k, color='red', linestyle='--')
plt.title('Chart 11: Elbow Method'); plt.legend(); plt.show()

# Final Clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
X_seg['Cluster'] = kmeans.fit_predict(X_seg_scaled)
cluster_profile = X_seg.groupby('Cluster').mean()

print("\n[Strategic Segments - Cluster Profile]")
print(cluster_profile)

# Chart 12: Strategy Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_profile.T, annot=True, cmap='YlGnBu', fmt='.1f')
plt.title(f'Chart 12: Strategic Segments (k={optimal_k})')
plt.show()
