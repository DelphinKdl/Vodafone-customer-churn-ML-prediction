# Vodafone Customer Churn Prediction
**Domain:** Telecom / Customer Retention
**Project Type:** Classification, Business-Focused Machine Learning

## **Business Problem**

### **What problem did I solve?**:

Telecom companies like Vodafone face significant losses from customer churn, when customers leave for competitors. Each lost customer reduces lifetime value, increases acquisition costs, and hurts brand loyalty.

### **Why does it matter?**:
Retaining customers is far more cost-effective than acquiring new ones. Predicting churn early allows the business to take targeted actions (e.g., retention campaigns, personalized offers) and significantly reduce churn-related revenue loss.

## **Project Goal **:
**Build a machine learning model that predicts whether a customer is likely to churn** allowing Vodafone to take proactive action to retain them.

## **My Approach**
### **Exploratory Data Analysis (EDA):**
   -  Identified correlations between churn and customer demographics, service subscriptions, and account status.
### **Feature Engineering & Cleaning:**
  - Converted categorical features into numerical formats
  - Handled class imbalance
  - Scaled features for modeling
### **Modeling Techniques Used:**
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Cross-validated performance using accuracy, precision, recall, and ROC-AUC
### **Deployment**
  - Built and will be deploying a Streamlit app to allow Vodafone staff to input customer data and get instant churn predictions.
## **Key Findings**
 - Contract type, tenure, and monthly charges were among the top churn predictors.
 - Customers with month-to-month contracts and high charges were significantly more likely to churn.
 - Random Forest and XGBoost performed best with ROC-AUC > 0.82.
## **Impact**
###  What changed because of my work?
The project provides Vodafone with a functional prototype to:
### What changed because of my work?
The project provides Vodafone with a functional prototype to:
 - Proactively target at-risk customers
 - Prioritize retention campaigns
 - Reduce churn-related revenue loss by enabling data-driven action.

**This isn’t just a model, it’s a deployable business tool.**
## **Tools Used** 
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn
- Streamlit (App interface)
- Git for version control
## **Try It Out** 
 [Coming SOON](LINK)
## **Future Improvements**
- Integrate with Vodafone’s CRM for real-time predictions
- A/B test retention strategies on model-flagged customers
- Incorporate call-center interaction data for deeper insights
  
## **License**
IT License, free to use with attribution.




