# 🏦 Risk Alert Classifier – Customer Risk Prediction

**Live Application:**  
https://hlkrnerpkphbqtro4yniaq.streamlit.app/

## 📌 Project Overview

This project applies supervised machine learning classification techniques to identify high-risk customers in a digital banking environment.

The project focuses on **class imbalance handling, classification performance evaluation, model comparison, and hyperparameter optimization** to develop an early-warning system for identifying customers who may default on payments or exhibit risky financial behavior.

A Streamlit application is also deployed to demonstrate the model's predictions.

---

## 🎯 Objectives

- Perform exploratory data analysis and understand customer risk patterns.
- Implement Logistic Regression as a baseline classifier.
- Evaluate classification performance using multiple metrics.
- Handle class imbalance using different sampling techniques.
- Implement Decision Tree and Random Forest classifiers.
- Analyze overfitting and model generalization.
- Apply Randomized Search CV and Grid Search CV for hyperparameter tuning.
- Compare models using ROC curves and AUC scores.
- Select a suitable classification model based on minimizing false negatives.

---

## 📊 Dataset

The dataset contains **4,600 customer records** with demographic, financial, behavioral, and transaction-related information.

### Target Variable

`risk_status`

- `0` – Low Risk
- `1` – High Risk

### Main Features

- `age`
- `gender`
- `region`
- `employment_type`
- `annual_income_inr`
- `credit_score`
- `credit_utilization_ratio`
- `missed_payments_12m`
- `avg_late_payment_days`
- `monthly_transaction_count`
- `monthly_spend_inr`
- `cash_advance_count_6m`
- `complaints_last_6m`
- `failed_login_attempts_3m`
- `account_tenure_months`
- `last_transaction_date`
- `debt_balance_inr`

`customer_id` is used only as an identifier and is excluded from model training.

The transaction date is converted into `transaction_recency_days` before removing the original date column.

---

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Streamlit
- Jupyter Notebook

---

## 🔄 Data Preprocessing

- Train-test split using stratified sampling.
- Missing value identification and treatment.
- KNN Imputation for numerical features.
- Most Frequent Imputation for categorical features.
- Standard Scaling for numerical features.
- One-Hot Encoding for categorical features.
- Feature engineering using transaction dates.

---

## 🤖 Models Implemented

### Logistic Regression
- Baseline classification model.

### Tree-Based Models
- Decision Tree Classifier.
- Random Forest Classifier.

---

## ⚖️ Handling Class Imbalance

The following sampling techniques are applied to address class imbalance:

- Random Under-Sampling
- Random Over-Sampling
- SMOTE (Synthetic Minority Over-sampling Technique)
- ADASYN (Adaptive Synthetic Sampling)

Model performance is compared before and after applying these techniques.

---

## 🔧 Hyperparameter Tuning

The following optimization techniques are used:

- Randomized Search CV
- Grid Search CV

Hyperparameter tuning is performed for Decision Tree and Random Forest classifiers to improve model performance and generalization.

---

## 📏 Evaluation Metrics

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- True Positive Rate (TPR)
- False Positive Rate (FPR)
- AUC-ROC

Additional evaluation includes confusion matrices, classification reports, and ROC curves.

---

## 📊 Comprehensive Model Performance Comparison

| Model | Test Accuracy | Minority Recall | Precision | F1-Score | AUC-ROC | Overfitting Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Logistic Regression** | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 | 0.00% |
| **SMOTE Logistic Regression** | 99.78% | 100.00% | 98.23% | 99.11% | 1.0000 | ~0.00% |
| **Decision Tree (Untuned)** | 96.96% | 84.68% | 89.52% | 87.04% | 0.9166 | 3.04% |
| **Decision Tree (Tuned)** | 97.50% | 92.79% | 87.29% | 89.96% | 0.9848 | 1.50% |
| **Random Forest (Untuned)** | 99.89% | 99.10% | 100.00% | 99.55% | 1.0000 | 0.11% |
| **Random Forest (Tuned) [Champion]** | **99.78%** | **98.20%** | **100.00%** | **99.09%** | **0.9999** | **0.08%** |

---

## 🔍 Key Findings

- Late payment days and missed payments show strong positive relationships with customer risk.
- Credit score has a negative relationship with risk status.
- Credit utilization and cash advance activity are important risk indicators.
- Several features show considerable distribution differences between Low Risk and High Risk customers.
- The initial Logistic Regression model achieved perfect test-set evaluation metrics, requiring further validation.
- Model selection considers minority-class recall and the cost of false negatives.

---

## 🚀 Streamlit Application

The project includes a deployed Streamlit application that provides an interactive interface for customer risk prediction.

---

## 🏁 Conclusion

This project demonstrates an end-to-end machine learning classification workflow for customer risk prediction.

It explores different classification algorithms, sampling strategies, evaluation metrics, and hyperparameter optimization techniques to understand their impact on model performance.

The final model is selected based on its ability to identify high-risk customers while minimizing false negatives, helping support early risk detection in digital banking.