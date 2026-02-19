# Telco Customer Churn Analysis & Prediction (IBM Dataset)

## Problem
Telecom churn reduces recurring revenue and increases acquisition costs. This project analyzes churn drivers and builds ML models to predict which customers are likely to churn, enabling targeted retention strategies.

## Data
IBM Telco Customer Churn dataset (7,043 customers, 21 columns). Target variable: `Churn` (converted to binary `churn` 0/1).

## Method
The project is organized into 5 notebooks:

1. **Notebook 01:** Data loading, cleaning, type fixes, target definition  
2. **Notebook 02:** EDA (churn vs tenure, contract, payment method, charges, and services)  
3. **Notebook 03:** Feature engineering and preprocessing using `Pipeline` + `ColumnTransformer` (no leakage) and stratified train/test split  
4. **Notebook 04:** Baseline Logistic Regression with class imbalance comparison (`class_weight='balanced'`) + metrics + threshold discussion + stratified CV  
5. **Notebook 05:** Model comparison (Random Forest, optional XGBoost), feature importance, interpretation, and business recommendations  

Key modeling practices:
- Reproducibility via `random_state`
- Stratified splitting and `StratifiedKFold` CV
- Leakage prevention: preprocessing fit only on training folds inside pipelines
- Metrics: ROC-AUC, PR-AUC, F1, confusion matrix, threshold tuning

## Results
Logistic Regression (balanced):
- ROC-AUC: 0.842
- PR-AUC: 0.633
- F1-score: 0.614
- Cross-validated ROC-AUC: 0.846 ± 0.012

Random Forest models did not outperform Logistic Regression and showed lower recall and F1. Therefore, Logistic Regression was selected as the final model due to better performance and interpretability.


## Business Impact
Using the model’s churn probabilities, the business can:
- Prioritize retention outreach for high-risk customers
- Reduce churn in month-to-month customers via contract incentives
- Improve early-tenure onboarding to reduce early churn
- Encourage autopay adoption to reduce payment-related churn
- Bundle value-added services (e.g., tech support/security) to increase stickiness

## Limitations
- Observational dataset: findings are correlational, not causal
- Performance may drift if pricing/plans/customer behavior changes
- Threshold choice depends on retention cost vs churn cost

## How to Run
1. Create environment and install dependencies:
   ```bash
   pip install -r requirements.txt

2. Run notebooks in order from notebooks/01_... to 05_...

3. Optional Streamlit demo:
   ```bash
   python -m streamlit run app/streamlit_app.py
