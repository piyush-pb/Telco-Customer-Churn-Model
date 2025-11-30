import json
import os

notebook_path = 'notebook/01_data_exploration.ipynb'

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [source]
    }

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True)
    }

# Cell 11: Model Evaluation
cell_11_code = """
# Cell 11: Model Evaluation - Logistic Regression

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

print("="*60)
print("STEP 1: Calculate Metrics")
print("="*60)

# Calculate probabilities if not already done
if 'y_pred_proba' not in locals():
    y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

print(f"Accuracy:      {accuracy:.4f}")
print(f"Precision:     {precision:.4f}")
print(f"Recall:        {recall:.4f}")
print(f"F1-Score:      {f1:.4f}")
print(f"ROC-AUC:       {roc_auc:.4f}")
print(f"Specificity:   {specificity:.4f}")

print("\\nConfusion Matrix Breakdown:")
print(f"True Negatives (Retained correctly identified): {tn}")
print(f"False Positives (Retained predicted as Churn):  {fp}")
print(f"False Negatives (Churn predicted as Retained):  {fn}")
print(f"True Positives (Churn correctly identified):    {tp}")

print("\\n" + "="*60)
print("STEP 2: Visualization")
print("="*60)

# Create 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 1. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], cbar=False,
            xticklabels=['Retained', 'Churned'], yticklabels=['Retained', 'Churned'])
axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Predicted', fontsize=12)
axes[0, 0].set_ylabel('Actual', fontsize=12)

# Add percentages
total = np.sum(cm)
for i in range(2):
    for j in range(2):
        axes[0, 0].text(j + 0.5, i + 0.7, f"({cm[i, j]/total:.1%})", 
                        ha='center', va='center', color='black', fontsize=10)

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0, 1].plot(fpr, tpr, color='#2ecc71', lw=3, label=f'ROC Curve (AUC = {roc_auc:.3f})')
axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0, 1].set_xlim([0.0, 1.0])
axes[0, 1].set_ylim([0.0, 1.05])
axes[0, 1].set_xlabel('False Positive Rate', fontsize=12)
axes[0, 1].set_ylabel('True Positive Rate', fontsize=12)
axes[0, 1].set_title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
axes[0, 1].legend(loc="lower right", fontsize=12)
axes[0, 1].grid(alpha=0.3)

# 3. Precision-Recall Curve
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall_curve, precision_curve)
axes[1, 0].plot(recall_curve, precision_curve, color='#e74c3c', lw=3, label=f'PR Curve (AUC = {pr_auc:.3f})')
axes[1, 0].set_xlabel('Recall', fontsize=12)
axes[1, 0].set_ylabel('Precision', fontsize=12)
axes[1, 0].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
axes[1, 0].legend(loc="lower left", fontsize=12)
axes[1, 0].grid(alpha=0.3)

# 4. Metrics Summary Table
metrics_data = [
    ['Accuracy', f"{accuracy:.4f}"],
    ['Precision', f"{precision:.4f}"],
    ['Recall', f"{recall:.4f}"],
    ['F1-Score', f"{f1:.4f}"],
    ['ROC-AUC', f"{roc_auc:.4f}"],
    ['Specificity', f"{specificity:.4f}"]
]
table = axes[1, 1].table(cellText=metrics_data, colLabels=['Metric', 'Value'], loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1, 2)
axes[1, 1].axis('off')
axes[1, 1].set_title('Model Performance Metrics', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../outputs/05_model_evaluation.png', dpi=300, bbox_inches='tight')
print("Visualization saved to outputs/05_model_evaluation.png")
plt.show()

print("\\n" + "="*60)
print("STEP 3: Detailed Classification Report")
print("="*60)
print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))
"""

# Cell 12: Random Forest
cell_12_code = """
# Cell 12: Random Forest Model & Comparison

from sklearn.ensemble import RandomForestClassifier

print("="*60)
print("STEP 1: Train Random Forest Classifier")
print("="*60)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)
print("Random Forest model trained successfully!")

print("\\n" + "="*60)
print("STEP 2: Evaluation & Comparison")
print("="*60)

# Predictions
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# Calculate RF Metrics
rf_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_rf),
    'Precision': precision_score(y_test, y_pred_rf),
    'Recall': recall_score(y_test, y_pred_rf),
    'F1-Score': f1_score(y_test, y_pred_rf),
    'ROC-AUC': roc_auc_score(y_test, y_pred_proba_rf)
}

# Get LR Metrics (from previous cell variables)
lr_metrics = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'ROC-AUC': roc_auc
}

# Create Comparison DataFrame
comparison_df = pd.DataFrame([lr_metrics, rf_metrics], index=['Logistic Regression', 'Random Forest'])
print("Model Performance Comparison:")
print(comparison_df.round(4))

print("\\n" + "="*60)
print("STEP 3: Feature Importance")
print("="*60)

# Extract Feature Importances
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Top 15 Most Important Features (Random Forest):")
print(importances.head(15).to_string(index=False))

print("\\n" + "="*60)
print("STEP 4: Visualization")
print("="*60)

# Create 1x2 subplot
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# 1. Feature Importance Bar Chart
sns.barplot(x='Importance', y='Feature', data=importances.head(15), 
            palette='viridis', ax=axes[0])
axes[0].set_title('Top 15 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Importance Score', fontsize=12)
axes[0].grid(axis='x', alpha=0.3)

# 2. Comparison Table
# Prepare cell text
cell_text = []
for row in comparison_df.itertuples():
    cell_text.append([f"{x:.4f}" for x in row[1:]])

# Add table
table = axes[1].table(cellText=cell_text, rowLabels=comparison_df.index, colLabels=comparison_df.columns, 
                     loc='center', cellLoc='center', bbox=[0.1, 0.3, 0.8, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(14)
axes[1].axis('off')
axes[1].set_title('Model Comparison', fontsize=14, fontweight='bold')

# Add Recommendation Text
recommendation = (
    "RECOMMENDATION:\\n"
    "Random Forest typically achieves higher accuracy and ROC-AUC due to its ability\\n"
    "to capture non-linear relationships. However, Logistic Regression offers\\n"
    "better interpretability (coefficients). If the goal is pure prediction performance,\\n"
    "choose Random Forest. If explaining 'why' is critical, stick with Logistic Regression."
)
axes[1].text(0.5, 0.1, recommendation, ha='center', va='center', fontsize=12, 
             bbox=dict(facecolor='#f0f0f0', alpha=0.5, boxstyle='round,pad=1'))

plt.tight_layout()
plt.savefig('../outputs/06_model_comparison.png', dpi=300, bbox_inches='tight')
print("Visualization saved to outputs/06_model_comparison.png")
plt.show()

print("\\n" + "="*60)
print("FINAL RECOMMENDATION")
print("="*60)
print(recommendation)
"""

# Cell 13: Business Impact
cell_13_code = """
# Cell 13: Business Impact & ROI Analysis

print("="*60)
print("BUSINESS IMPACT & ROI ANALYSIS")
print("="*60)

# 1. Calculate Customer Lifetime Value (CLV)
avg_monthly_charges = df['MonthlyCharges'].mean()
churn_rate = df['Churn_Binary'].mean()
avg_lifetime_months = 1 / churn_rate if churn_rate > 0 else 0
clv = avg_monthly_charges * avg_lifetime_months

print(f"Average Monthly Charges: ${avg_monthly_charges:.2f}")
print(f"Overall Churn Rate:      {churn_rate:.1%}")
print(f"Avg Customer Lifetime:   {avg_lifetime_months:.1f} months")
print(f"Customer Lifetime Value: ${clv:.2f}")

print("\\n" + "-"*60)
print("2. RISK SEGMENTATION (Test Set Sample)")
print("-" * 60)

# Create DataFrame for analysis using test set predictions
risk_df = pd.DataFrame({'Probability': y_pred_proba})

# Define segments
def get_risk_segment(prob):
    if prob < 0.3: return 'Low Risk'
    elif prob < 0.6: return 'Medium Risk'
    elif prob < 0.8: return 'High Risk'
    else: return 'Critical Risk'

risk_df['Segment'] = risk_df['Probability'].apply(get_risk_segment)

# Count and Revenue at Risk
segment_counts = risk_df['Segment'].value_counts()
# Estimate revenue at risk (assuming these customers have avg CLV)
# Note: This is a projection based on the test set sample size
revenue_at_risk = {
    'High Risk': segment_counts.get('High Risk', 0) * clv,
    'Critical Risk': segment_counts.get('Critical Risk', 0) * clv
}

print("Customer Distribution by Risk Segment:")
print(segment_counts)
print(f"\\nPotential Revenue at Risk (High + Critical Segments in Sample):")
print(f"High Risk:     ${revenue_at_risk['High Risk']:,.2f}")
print(f"Critical Risk: ${revenue_at_risk['Critical Risk']:,.2f}")
print(f"Total At Risk: ${sum(revenue_at_risk.values()):,.2f}")

print("\\n" + "-"*60)
print("3. RETENTION CAMPAIGN ROI ANALYSIS")
print("-" * 60)

campaign_cost_per_customer = 50.0
success_rate = 0.30  # 30% of targeted customers are saved

def calculate_roi(threshold):
    target_customers = risk_df[risk_df['Probability'] > threshold]
    n_target = len(target_customers)
    
    if n_target == 0:
        return 0, 0, 0, 0, 0
    
    total_cost = n_target * campaign_cost_per_customer
    customers_saved = n_target * success_rate
    revenue_saved = customers_saved * clv
    net_benefit = revenue_saved - total_cost
    roi = (net_benefit / total_cost) * 100
    
    return n_target, total_cost, customers_saved, revenue_saved, net_benefit, roi

# Calculate for target threshold > 0.6
n_target, cost, saved, rev_saved, net, roi = calculate_roi(0.6)

print(f"Campaign Assumptions:")
print(f" - Target: Probability > 0.6")
print(f" - Cost per customer: ${campaign_cost_per_customer}")
print(f" - Success Rate: {success_rate:.0%}")
print(f"\\nResults:")
print(f" - Customers Targeted: {n_target}")
print(f" - Total Campaign Cost: ${cost:,.2f}")
print(f" - Customers Saved:    {saved:.1f}")
print(f" - Revenue Saved:      ${rev_saved:,.2f}")
print(f" - Net Benefit:        ${net:,.2f}")
print(f" - ROI:                {roi:.1f}%")

print("\\n" + "-"*60)
print("4. STRATEGIC RECOMMENDATIONS")
print("-" * 60)

recommendations = [
    {
        "Rec": "1. Target Month-to-Month Customers",
        "Target": "Customers with tenure < 12 months on month-to-month contracts.",
        "Action": "Offer a 15% discount for switching to a 1-year contract.",
        "Impact": "Reduces churn probability by ~40% (based on odds ratio)."
    },
    {
        "Rec": "2. Payment Method Incentives",
        "Target": "Customers paying via Electronic Check.",
        "Action": "Give $5 bill credit for setting up Auto-Pay (Credit Card/Bank Transfer).",
        "Impact": "Moves customers from high-churn segment to lower-churn segment."
    },
    {
        "Rec": "3. First-Year Care Program",
        "Target": "New customers (0-6 months tenure).",
        "Action": "Proactive check-in calls and personalized onboarding emails.",
        "Impact": "Increases early engagement and reduces early-stage churn."
    }
]

for item in recommendations:
    print(f"\\n{item['Rec']}")
    print(f"Target: {item['Target']}")
    print(f"Action: {item['Action']}")
    print(f"Impact: {item['Impact']}")

print("\\n" + "-"*60)
print("5. VISUALIZATION")
print("-" * 60)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# 1. Risk Distribution
sns.histplot(risk_df['Probability'], bins=20, kde=True, color='#3498db', ax=axes[0])
axes[0].axvline(0.6, color='red', linestyle='--', label='Campaign Threshold (0.6)')
axes[0].set_title('Customer Churn Probability Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Churn Probability', fontsize=12)
axes[0].set_ylabel('Number of Customers', fontsize=12)
axes[0].legend()

# 2. ROI by Threshold
thresholds = [0.5, 0.6, 0.7, 0.8]
rois = [calculate_roi(t)[5] for t in thresholds]

sns.barplot(x=thresholds, y=rois, palette='Greens_d', ax=axes[1])
axes[1].set_title('Campaign ROI by Probability Threshold', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Target Probability Threshold', fontsize=12)
axes[1].set_ylabel('ROI (%)', fontsize=12)

# Add labels
for i, v in enumerate(rois):
    axes[1].text(i, v + 5, f"{v:.0f}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../outputs/07_business_impact.png', dpi=300, bbox_inches='tight')
print("Visualization saved to outputs/07_business_impact.png")
plt.show()
"""

# Cell 14: Executive Summary
cell_14_code = """
# Cell 14: Executive Summary & Dashboard

print("="*60)
print("EXECUTIVE SUMMARY GENERATION")
print("="*60)

# Prepare Data for Summary
top_drivers = importances.head(3)
driver_1 = f"{top_drivers.iloc[0]['Feature']} ({top_drivers.iloc[0]['Importance']:.3f})"
driver_2 = f"{top_drivers.iloc[1]['Feature']} ({top_drivers.iloc[1]['Importance']:.3f})"
driver_3 = f"{top_drivers.iloc[2]['Feature']} ({top_drivers.iloc[2]['Importance']:.3f})"

summary_text = f\"\"\"
EXECUTIVE SUMMARY: CUSTOMER CHURN ANALYTICS
===========================================

1. PROJECT OVERVIEW
-------------------
Objective: Analyze customer churn drivers and build a predictive model to improve retention.
Data: Telco customer dataset (7,043 records).
Current Churn Rate: {churn_rate:.1%}

2. KEY FINDINGS
---------------
The top 3 drivers of customer churn are:
1. {top_drivers.iloc[0]['Feature']}: Strongest predictor.
2. {top_drivers.iloc[1]['Feature']}: Significant impact.
3. {top_drivers.iloc[2]['Feature']}: Notable factor.

Insight: Customers on month-to-month contracts and those using fiber optic services are at highest risk.
Tenure is a major retention factor; risk drops significantly after 12 months.

3. MODEL PERFORMANCE
--------------------
Model Selected: Logistic Regression (for interpretability)
Accuracy: {accuracy:.1%}
ROC-AUC:  {roc_auc:.3f}
Performance: The model successfully identifies {recall:.1%} of actual churners (Recall).

4. BUSINESS RECOMMENDATIONS
---------------------------
1. Target Month-to-Month Customers: Incentivize switching to 1-year contracts (15% discount).
2. Payment Method Optimization: Drive adoption of Auto-Pay to reduce involuntary churn.
3. New Customer Onboarding: Implement "First 90 Days" care program to boost early tenure.

5. FINANCIAL IMPACT
-------------------
Campaign Target: High-risk customers (>60% probability).
Projected Net Benefit: ${net:,.2f} (Test Sample)
Projected ROI: {roi:.1f}%
\"\"\"

print(summary_text)

# Save to file
with open('../outputs/Executive_Summary.txt', 'w') as f:
    f.write(summary_text)
print("Summary saved to outputs/Executive_Summary.txt")

print("\\n" + "="*60)
print("DASHBOARD VISUALIZATION")
print("="*60)

# Create Dashboard
fig = plt.figure(figsize=(18, 10))
fig.suptitle('Customer Churn Analytics Dashboard', fontsize=24, fontweight='bold', y=0.95)

# Grid layout
gs = fig.add_gridspec(2, 4)

# KPI 1: Churn Rate
ax1 = fig.add_subplot(gs[0, 0])
ax1.text(0.5, 0.6, f"{churn_rate:.1%}", ha='center', va='center', fontsize=40, fontweight='bold', color='#e74c3c')
ax1.text(0.5, 0.3, "Overall Churn Rate", ha='center', va='center', fontsize=14, color='gray')
ax1.axis('off')

# KPI 2: Model Accuracy
ax2 = fig.add_subplot(gs[0, 1])
ax2.text(0.5, 0.6, f"{accuracy:.1%}", ha='center', va='center', fontsize=40, fontweight='bold', color='#2ecc71')
ax2.text(0.5, 0.3, "Model Accuracy", ha='center', va='center', fontsize=14, color='gray')
ax2.axis('off')

# KPI 3: ROI
ax3 = fig.add_subplot(gs[0, 2])
ax3.text(0.5, 0.6, f"{roi:.0f}%", ha='center', va='center', fontsize=40, fontweight='bold', color='#3498db')
ax3.text(0.5, 0.3, "Projected ROI", ha='center', va='center', fontsize=14, color='gray')
ax3.axis('off')

# KPI 4: Net Savings (Sample)
ax4 = fig.add_subplot(gs[0, 3])
ax4.text(0.5, 0.6, f"${net/1000:.1f}k", ha='center', va='center', fontsize=40, fontweight='bold', color='#f1c40f')
ax4.text(0.5, 0.3, "Est. Savings (Sample)", ha='center', va='center', fontsize=14, color='gray')
ax4.axis('off')

# Chart: Top 3 Drivers
ax5 = fig.add_subplot(gs[1, :])
sns.barplot(x='Importance', y='Feature', data=top_drivers, palette='viridis', ax=ax5)
ax5.set_title('Top 3 Churn Drivers', fontsize=16, fontweight='bold')
ax5.set_xlabel('Importance Score', fontsize=12)
ax5.set_ylabel('', fontsize=12)

# Add values to bars
for i, v in enumerate(top_drivers['Importance']):
    ax5.text(v + 0.005, i, f"{v:.3f}", va='center', fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig('../outputs/08_executive_summary.png', dpi=300, bbox_inches='tight')
print("Dashboard saved to outputs/08_executive_summary.png")
plt.show()
"""

# Cell 15: Export Data
cell_15_code = """
# Cell 15: Export Data for Dashboard

print("="*60)
print("EXPORT DATA FOR DASHBOARD")
print("="*60)

# 1. Prepare Customer Data Sheet
# ------------------------------
# We need to predict on the full dataset. 
# First, apply the same preprocessing (dummies, scaling) to the full df.
# Note: In a production pipeline, we would use a pipeline object. 
# Here, we'll manually replicate the steps for the full df.

# Re-create dummies for full df (df_model was created in Cell 10)
# Ensure we have the same columns as X_train
full_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)
# Align columns with training data (add missing cols with 0, drop extra cols)
full_encoded = full_encoded.reindex(columns=X.columns, fill_value=0)

# Scale
full_scaled = pd.DataFrame(scaler.transform(full_encoded), columns=X.columns)

# Predict
full_probs = log_reg.predict_proba(full_scaled)[:, 1]
full_preds = log_reg.predict(full_scaled)

# Create export dataframe
export_df = df.copy()
export_df['Risk_Score'] = full_probs
export_df['Predicted_Churn'] = full_preds
export_df['Risk_Category'] = export_df['Risk_Score'].apply(get_risk_segment)

print(f"Customer Data prepared: {export_df.shape[0]} rows")

# 2. Prepare Segment Summary Sheet
# --------------------------------
segment_summary = []
segments_to_analyze = ['Contract', 'PaymentMethod', 'InternetService', 'TechSupport']

for seg in segments_to_analyze:
    summary = df.groupby(seg)['Churn_Binary'].agg(['count', 'sum', 'mean']).reset_index()
    summary.columns = ['Segment_Value', 'Total_Customers', 'Churned_Customers', 'Churn_Rate']
    summary['Segment_Name'] = seg
    summary = summary[['Segment_Name', 'Segment_Value', 'Total_Customers', 'Churned_Customers', 'Churn_Rate']]
    segment_summary.append(summary)

segment_summary_df = pd.concat(segment_summary, ignore_index=True)
print(f"Segment Summary prepared: {segment_summary_df.shape[0]} rows")

# 3. Prepare Model Performance Sheet
# ----------------------------------
# Combine LR and RF metrics
model_perf_df = comparison_df.reset_index().rename(columns={'index': 'Model'})
print("Model Performance prepared")

# 4. Prepare Financial Impact Sheet
# ---------------------------------
financial_data = {
    'Metric': [
        'Average Monthly Charges',
        'Overall Churn Rate',
        'Avg Customer Lifetime (Months)',
        'Customer Lifetime Value (CLV)',
        'High Risk Revenue at Risk (Sample)',
        'Critical Risk Revenue at Risk (Sample)',
        'Campaign ROI (Projected)',
        'Net Benefit (Sample)'
    ],
    'Value': [
        avg_monthly_charges,
        churn_rate,
        avg_lifetime_months,
        clv,
        revenue_at_risk['High Risk'],
        revenue_at_risk['Critical Risk'],
        roi,
        net
    ]
}
financial_df = pd.DataFrame(financial_data)
print("Financial Impact prepared")

# 5. Save to Excel
# ----------------
output_file = '../outputs/churn_dashboard_data.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    export_df.to_excel(writer, sheet_name='Customer_Data', index=False)
    segment_summary_df.to_excel(writer, sheet_name='Segment_Summary', index=False)
    model_perf_df.to_excel(writer, sheet_name='Model_Performance', index=False)
    financial_df.to_excel(writer, sheet_name='Financial_Impact', index=False)

print(f"\\nData successfully exported to: {output_file}")
print("Sheets created:")
print(" - Customer_Data")
print(" - Segment_Summary")
print(" - Model_Performance")
print(" - Financial_Impact")
"""

# Read notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Append cells
cells_to_add = [
    ("# Cell 11: Model Evaluation", cell_11_code),
    ("# Cell 12: Random Forest Model & Comparison", cell_12_code),
    ("# Cell 13: Business Impact & ROI Analysis", cell_13_code),
    ("# Cell 14: Executive Summary", cell_14_code),
    ("# Cell 15: Export Data for Dashboard", cell_15_code)
]

for title, code in cells_to_add:
    nb['cells'].append(create_markdown_cell(title))
    nb['cells'].append(create_code_cell(code))

# Write notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Restored Cells 11-15 successfully!")
