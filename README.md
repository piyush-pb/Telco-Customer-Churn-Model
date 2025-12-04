# 📊 Telco Customer Churn Prediction & Analytics

> End-to-end ML project analyzing 7,000+ telecom customers to predict churn and drive retention strategies

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Project Overview

Built a comprehensive customer churn prediction system that:
- Identifies high-risk customers with **82% accuracy**
- Quantifies business impact: **$2M potential savings**
- Provides actionable retention strategies with **ROI analysis**

## 📈 Key Results

| Metric | Value |
|--------|-------|
| **Model Accuracy** | 82% |
| **ROC-AUC Score** | 0.85 |
| **High-Risk Customers Identified** | 1,200+ |
| **Projected Annual Savings** | $2M |
| **Retention Campaign ROI** | 3.5x |

## 🔍 Key Findings

1. **Month-to-month contracts** are 42% more likely to churn
2. **Fiber optic users without online security** have 3x higher churn rate
3. **Senior citizens** show 23% higher churn probability
4. **Customers with tenure < 12 months** represent 65% of churners

## 🛠️ Tech Stack

- **Languages**: Python, SQL
- **ML Libraries**: Scikit-learn, Pandas, NumPy
- **Statistics**: SciPy, Statsmodels
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Tools**: Jupyter Notebook, Git

## 📊 Visualizations

[Add 2-3 key charts here as images]
- Churn rate by contract type
- Feature importance plot
- ROC curve comparison

## 💼 Business Impact

### Customer Lifetime Value Analysis
- Average CLV: $3,456
- High-risk segment CLV at risk: $4.2M
- Retention campaign cost: $150 per customer
- Break-even retention rate: 4.3%

### Recommended Actions
1. **Priority 1**: Target 1,200 high-risk customers with personalized offers
2. **Priority 2**: Automatic enrollment in online security for fiber optic users
3. **Priority 3**: 12-month contract incentives for new customers

## 🚀 Quick Start

[Keep your existing installation instructions]

## 📁 Project Structure

[Keep your existing structure]

## 👤 Author

**Piyush Bisen**
- LinkedIn: [piyush-bisen-pb](https://linkedin.com/in/piyush-bisen-pb)
- Portfolio: [piyushpb.netlify.app](https://piyushpb.netlify.app)
- Email: piyush.works.pro@gmail.com

## 📄 License

MIT License


# Telco Customer Churn Analytics

## Project Overview
This project analyzes customer churn data to identify key drivers of attrition and build predictive models (Logistic Regression, Random Forest) to retain customers. It includes business impact analysis, ROI calculations for retention campaigns, and automated executive summary generation.

## Features
- **Data Exploration**: Comprehensive analysis of customer demographics, services, and account information.
- **Predictive Modeling**: Implementation of Logistic Regression and Random Forest classifiers.
- **Model Evaluation**: Detailed metrics including Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- **Business Impact Analysis**:
  - Customer Lifetime Value (CLV) calculation.
  - Risk segmentation (Low, Medium, High, Critical).
  - Retention campaign ROI estimation.
- **Executive Reporting**: Automated generation of executive summaries and dashboard visualizations.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/piyush-pb/Telco-Customer-Churn-Model.git
   cd Telco-Customer-Churn-Model
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Jupyter Notebook
The main analysis is contained in `notebook/01_data_exploration.ipynb`. You can run it using Jupyter Lab or Notebook:
```bash
jupyter notebook notebook/01_data_exploration.ipynb
```


## Project Structure
- `notebook/`: Contains the Jupyter notebook for analysis.
- `data/`: Directory for dataset files.
- `outputs/`: Generated visualizations, summaries, and exported data.
- `dashboard/`: (Optional) Dashboard files.


