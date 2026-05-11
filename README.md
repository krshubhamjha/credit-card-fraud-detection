# Credit Card Fraud Detection

## Live App
[Click here to try the live app](https://credit-card-fraud-detection-tj4smrcmwxd6fvhfr2cdqd.streamlit.app/)

## Overview
A machine learning project to detect fraudulent credit card transactions using XGBoost, SMOTE, and SHAP explainability. Built with a complete MLOps pipeline including automated retraining when new data arrives.

## Problem Statement
Credit card fraud causes billions in losses every year. This project builds a model that catches as many frauds as possible while minimising false alarms. Only 0.17% of transactions are fraud making this a highly imbalanced classification problem.

## Dataset
- Source: Kaggle — European credit card transactions
- Size: 284,807 transactions
- Fraud rate: 0.17% (492 fraud cases)
- Features: V1-V28 (PCA transformed), Amount, Time

## Tech Stack
- Python
- XGBoost
- SMOTE
- SHAP
- Streamlit
- Pandas
- Scikit-learn
- GitHub Actions

## Project Structure
credit-card-fraud-detection/
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_shap_explainability.ipynb
├── src/
│   └── train.py
├── model/
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   └── feature_names.json
├── tests/
│   └── test_preprocess.py
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── retrain.yml
├── app.py
└── requirements.txt

## Approach
1. EDA — analysed class imbalance, amount patterns, time patterns
2. Feature Engineering — scaled Amount and Time, created Hour feature
3. Modelling — compared Logistic Regression, Random Forest, XGBoost
4. SMOTE — applied to balance fraud vs legitimate classes
5. SHAP — explained model decisions with waterfall plots
6. Streamlit — deployed interactive fraud detection app
7. CI/CD — automated testing and deployment with GitHub Actions

## Model Performance
| Model | F1 Score | Recall | ROC-AUC |
|-------|----------|--------|---------|
| Logistic Regression | 0.72 | 0.63 | 0.84 |
| Random Forest | 0.87 | 0.81 | 0.90 |
| XGBoost without SMOTE | 0.86 | 0.85 | 0.92 |
| XGBoost with SMOTE | 0.78 | 0.86 | 0.93 |

Final model: XGBoost with SMOTE — chosen for highest recall since catching maximum fraud is the business priority.

## MLOps Pipeline
- Every code push triggers CI/CD pipeline
- Tests run automatically on every push
- New transaction data pushed to data/new_data/ triggers automatic retraining
- New model deployed only if recall improves over existing model
- Append log tracks all data updates with date, rows added and fraud cases

## How to Run Locally
1. Clone the repo
git clone https://github.com/krshubhamjha/credit-card-fraud-detection.git

2. Create virtual environment
python -m venv venv
venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Run app
streamlit run app.py

## Test with these fraud values
Amount = 1.0
Hour = 15
V1 = -1.271244
V2 = 2.462675
V3 = -2.851395
V4 = 2.32448
V7 = -3.065234
V10 = -4.881143
V14 = -5.0
V17 = -3.0
rest = 0.0

## Author
Shubham Kumar
Data Analyst 
GitHub: https://github.com/krshubhamjha