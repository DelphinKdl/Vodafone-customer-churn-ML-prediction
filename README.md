# Vodafone Customer Churn Prediction
**Domain:** Telecom / Customer Retention

**Project Type:** Classification, Business-Focused Machine Learning

## Business Problem

### What problem did I solve?:

Telecom companies like Vodafone lose significant revenue due to customer churn, when customers leave for competitors. Each lost customer decreases lifetime value, raises acquisition costs, and weakens brand loyalty.

### Why does it matter?:
Retaining existing customers is far more cost-effective than acquiring new ones. Early churn prediction allows the business to take proactive actions (e.g., retention campaigns, personalized offers) to reduce revenue loss and improve satisfaction.

## Project Goal:
**Build a machine learning model that predicts whether a customer is likely to churn** enabling Vodafone to take proactive action to retain them.

## My Approach
### Exploratory Data Analysis (EDA):
   -  Identified churn trends across demographics, service types, and account behavior.
### Feature Engineering & Cleaning:
  - Converted categorical features into numerical formats
  - Handled class imbalance
  - Scaled features for modeling
### Modeling Techniques Used:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Cross-validated performance using accuracy, precision, recall, and ROC-AUC
### Deployment
  - Built (and will deploy) a Streamlit app for Vodafone staff to input customer data and get real-time churn predictions.

## Key Findings
 - **Contract type**, **tenure**, and **monthly charges** were the top churn predictors.
 - Customers on **month-to-month contracts** with **high charges** had the highest churn risk.
 - **Random Forest** and **XGBoost** achieved **ROC-AUC > 0.82**, delivering reliable predictive performance.

## Impact
This project delivers a functional prototype that Vodafone can use to:
 - Proactively target at-risk customers
 - Prioritize retention campaigns
 - Reduce churn-related revenue loss by enabling data-driven action.
**This isn’t just a model, it’s a deployable business tool.**
## Tools Used
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn
- Streamlit (App interface)
- Git for version control
## Try It Out
 [Coming SOON](#)
## Future Improvements
- Integrate with Vodafone’s CRM for real-time predictions
- A/B test retention strategies on model-flagged customers
- Incorporate call-center interaction data for deeper insights  
## License
IT License, free to use with attribution.
