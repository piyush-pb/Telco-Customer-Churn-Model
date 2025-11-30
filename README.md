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

