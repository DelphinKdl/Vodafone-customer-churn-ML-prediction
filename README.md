# Customer Churn prediction Using Machine Learning in subscription business.
Every company aims to increase its profit or revenue margin; customer retention is a critical area where industry players focus their resources. In today's world of machine learning, most companies build classification models to perform churn analysis on their customers.

This repository contains a project on customer churn analysis using machine learning techniques. The primary goal is to build a predictive model to predict customer churn, helping companies to identify customers who are likely to leave and to implement strategies to retain them.

## **Business Undestanding**

### **Problem statements**:

Vodafone is currently facing a substantial challenge with customer churn, which negatively impacts its revenue and operational efficiency. The existing approaches to manage churn are predominantly reactive and lack the precision to effectively identify customers at risk of leaving. This inability to predict churn accurately hinders Vodafone's efforts to implement timely and effective retention strategies, resulting in increased costs associated with acquiring new customers and managing churn.

### **Project Goal and Objectives**:

**Goal:**
Develop a robust machine learning model to accurately predict customer churn at Vodafone, enabling the company to implement preemptive retention strategies and reduce the overall churn rate.

**Objectives:**

1. **Data Analysis:** Conduct a thorough analysis of customer data to identify patterns and factors contributing to churn, including demographics, usage patterns, service quality, and customer support interactions.
2. **Feature Engineering:** Create meaningful features from raw data to improve the predictive power of the churn model.
3. **Model Development:** Build and evaluate various machine learning models to predict customer churn, selecting the best-performing model based on accuracy, precision, recall, and other relevant metrics.
4. **Model Evaluation:** Rigorously assess the performance of the models using appropriate validation techniques and metrics to ensure robustness and reliability.
5. **Actionable Insights:** Provide actionable insights and recommendations based on the model’s predictions to help Vodafone develop targeted retention campaigns and improve customer satisfaction.

##### **Stakeholder**

1. **Customer Retention Team**:
   **Role**: Implement retention strategies based on the model's predictions.

2. **Marketing Team**:
   **Role**: Develop and execute marketing strategies to enhance customer loyalty.
3. **Customer Service Team**:
   **Role**: Address customer concerns and improve service quality.
4. **Data Science and Analytics Team**:
   **Role**: Develop, refine, and validate the churn prediction model.
5. **Vodafone Senior Management**:
   **Role**: Make strategic decisions based on model insights.

These stakeholders will actively use the model to drive decisions and actions that enhance customer retention and loyalty, ultimately contributing to Vodafone's overall business success.

##### **Key Metrics and Success Criteria**

1. **Accuracy**: Proportion of correct predictions.
   **Target**: ≥ 85% (Balanced data)

2. **F1 Score**:
   **Definition**: Harmonic mean of precision and recall.
   **Target**: ≥ 80%.

3. **AUC-ROC**: Discriminative power of the model.
   **Target**: ≥ 0.80.

4. At least 4 baseline models
5. All hyperparameter uning should be only to basedline model if they excedd there F1 score

### **Features**

There are 17 categorical features:

- CustomerID : Unique identifier for each customer.
- Gender : Gender of the customer (Male, Female).
- SeniorCitizen : Indicates if the customer is a senior citizen (1: Yes, 0: No).
- Partner : Indicates if the customer has a partner (Yes, No).
- Dependents : Indicates if the customer has dependents (Yes, No).
- PhoneService : Indicates if the customer has a phone service (Yes, No).
- MultipleLines : Indicates if the customer has multiple lines (Yes, No, No phone service).
- InternetService : Type of internet service the customer has (DSL, Fiber optic, No).
- OnlineSecurity : Indicates if the customer has online security service (Yes, No, No internet service).
- OnlineBackup : Indicates if the customer has an online backup service (Yes, No, No internet service).
- DeviceProtection : Indicates if the customer has device protection service (Yes, No, No internet service).
- TechSupport : Indicates if the customer has tech support service (Yes, No, No internet service).
- StreamingTV : Indicates if the customer streams TV (Yes, No, No internet service).
- StreamingMovies : Indicates if the customer streams movies (Yes, No, No internet service).
- Contract : The contract term of the customer (Month-to-month, One year, Two years).
- PaperlessBilling: Indicates if the customer uses paperless billing (True, False).
- PaymentMethod: The customer's payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)).

Next, there are 3 numerical features:

- Tenure: Number of months the customer has stayed with the company
- MonthlyCharges: The amount charged to the customer monthly
- TotalCharges: The total amount charged to the customer

Finally, there’s a prediction feature:

- Churn: Whether the customer churned or not (Yes or No)

### **Hypotheses**

**Hypothesis 2: Total Charges**
Higher monthly charges might lead to dissatisfaction among customers, especially if they do not perceive a corresponding value in the service, leading to higher churn rates.

**Null Hypothesis (H0):**
Total charges have no effect on the likelihood of customer churn.

**Alternative Hypothesis (H1):**
Total charges have a significant effect on the likelihood of customer churn.

### Analtytics Questions

1. Overall Churn Rate:

   What is the overall churn rate in the dataset?

2. Contract Type and Churn:

   What is the distribution of contract types among customers?
   How does the contract type affect the churn rate?

3. Tenure and Churn:

4. What is the distribution of tenure among customers?
   How does tenure affect the churn rate?
5. Monthly Charges and Churn:

   What is the distribution of monthly charges among customers?
   How do monthly charges affect the churn rate?

6. Payment Method and Churn:

   What are the different payment methods used by customers?
   How does the payment method affect the churn rate?

7. Internet Service and Churn:

   What types of internet services are customers using?
   How does the type of internet service affect the churn rate?

Additional Services and Churn (Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies):

How does the usage of these additional services affect the churn rate?
Senior Citizens and Churn:

What percentage of customers are senior citizens?
How does the churn rate compare between senior citizens and non-senior citizens?
Partners and Churn:

How many customers have partners?
Is there a significant difference in churn rates between customers with and without partners?
Dependents and Churn:

How many customers have dependents?
How does the presence of dependents affect the churn rate?
