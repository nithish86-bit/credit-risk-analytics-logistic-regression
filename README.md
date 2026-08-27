# Credit Risk Analytics using Logistic Regression

## 📌 Project Overview

This project uses **Logistic Regression** to predict the probability of loan default for bank customers.

The analysis follows an end-to-end data science workflow including data understanding, exploratory data analysis (EDA), data cleaning, correlation analysis, variable selection, multicollinearity analysis, model building, model evaluation, cutoff selection, decile analysis, and prediction of new customers.

## 🎯 Objective

To develop a Logistic Regression model that predicts whether a customer is likely to default on a loan and identify customers with higher default risk.

## 📊 Dataset

The dataset contains information about bank customers and their financial and demographic characteristics.

The target variable is:

* `default` – indicates whether the customer defaulted on the loan

The dataset includes variables related to customer financial information, credit history, employment, debt, and other customer characteristics.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy
* Scikit-learn
* Statsmodels
* Patsy

## 🔎 Analysis Workflow

### 1. Data Understanding & EDA

* Inspected dataset structure and dimensions
* Checked data types and missing values
* Examined numerical variables
* Calculated cardinality of the target variable
* Analyzed the distribution of default and non-default customers
* Performed descriptive analysis
* Analyzed correlations between variables

### 2. Data Cleaning

* Treated extreme values using percentile-based clipping
* Examined missing values
* Prepared the dataset for model development

### 3. Train-Test Split

The existing customer dataset was divided into:

* 70% Training Data
* 30% Testing Data

A random state was used to make the split reproducible.

### 4. Logistic Regression Model

A Logistic Regression model was developed using **Statsmodels**.

The initial model was built using all available predictor variables.

The model was then refined using variable significance and multicollinearity analysis.

### 5. Variable Selection using Somer's D

Bivariate Logistic Regression was performed for each independent variable.

ROC-AUC was calculated for each variable and Somer's D was derived using:

`Somers' D = 2 × ROC-AUC - 1`

Variables with Somer's D of at least 0.20 were considered for further model development.

### 6. Multicollinearity Analysis

Variance Inflation Factor (VIF) was used to identify multicollinearity among the selected predictor variables.

Variables were reviewed to build a more stable Logistic Regression model.

### 7. Final Model Development

A final Logistic Regression model was developed using the selected variables.

The final model was used to calculate the probability of default for both training and testing datasets.

### 8. Model Evaluation

Model performance was evaluated using:

* ROC-AUC
* Accuracy
* Confusion Matrix
* Classification Report
* Sensitivity
* Specificity

The model was evaluated on both training and testing datasets.

### 9. Cutoff Selection

Different probability cutoff values between 0 and 1 were evaluated.

The cutoff was selected based on the highest combined:

* Sensitivity
* Specificity

The selected cutoff was **0.23**.

Customers were classified as:

* `1` → Default / High Risk
* `0` → Non-default / Lower Risk

### 10. Decile Analysis

Customers were divided into ten groups based on their predicted probability of default.

Decile analysis was used to understand how default risk is distributed across customer segments.

The decile results were exported into:

* `train_deciles.csv`
* `test_deciles.csv`

### 11. New Customer Prediction

The final Logistic Regression model was applied to new customers whose default status was not available.

The model calculates the probability of default and classifies customers based on the selected cutoff.

In this analysis:

* **64 customers** were identified as high risk
* **86 customers** were identified as lower risk

## 📈 Final Model Features

The final Logistic Regression model uses the following variables:

* `address`
* `creddebt`
* `debtinc`
* `employ`

These variables were selected through the variable selection and model refinement process.

## 📊 Model Results

| Metric   |  Train |   Test |
| -------- | -----: | -----: |
| ROC-AUC  |  0.837 |  0.886 |
| Accuracy | 72.86% | 74.76% |

### Classification Cutoff

**Best Cutoff: 0.23**

### Test Classification Performance

At the selected cutoff of 0.23:

* Default Recall / Sensitivity: **84%**
* Non-default Recall / Specificity: **72%**
* Overall Accuracy: **74.76%**

The test ROC-AUC of **0.886** indicates that the model has good ability to distinguish between default and non-default customers.

## 💼 Business Interpretation

The model can help a bank identify customers with a higher probability of loan default.

High-risk customers can be subjected to additional verification or scrutiny before approving a loan.

Lower-risk customers can be processed with fewer restrictions.

The model can therefore support risk-based loan approval and proactive credit risk management.

## ⚠️ Model Limitations

The model may have limitations due to:

* Limited dataset size
* Possible multicollinearity
* Model assumptions
* Potential overfitting
* Limited number of predictive variables
* Changes in customer behavior over time
* Model performance depending on the selected cutoff

## 📁 Repository Structure

```text
credit-risk-analytics-logistic-regression/
│
├── bankloans.csv
├── credit_risk_analytics.py
├── train_deciles.csv
├── test_deciles.csv
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/nithish86-bit/credit-risk-analytics-logistic-regression.git
```

### 2. Navigate to the project folder

```bash
cd credit-risk-analytics-logistic-regression
```

### 3. Install required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Python script

```bash
python credit_risk_analytics.py
```

The script performs the complete analysis including data preparation, model development, evaluation, decile analysis, and prediction of new customers.

## 👨‍💻 Author

**Nithish Ramesh**

GitHub: [nithish86-bit](https://github.com/nithish86-bit)

LinkedIn: [Nithish Ramesh](https://www.linkedin.com/in/nithish-rbn)
