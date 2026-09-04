# 💳 Credit Card Default Prediction & Risk Analytics

An end-to-end Machine Learning classification pipeline built on the UCI Credit Card dataset (30,000 records) to identify default risks, engineer custom financial stress indicators, and evaluate classification algorithms for risk assessment.

---

## 📌 Project Overview
Credit card default prediction is a crucial task for financial institutions to manage credit risk and minimize financial losses. This project analyzes customer demographic details, historical payment behavior, and bill statements to predict whether a client will default on their credit card payment next month.

Special focus is given to **Recall**, as failing to detect a potential defaulter (False Negative) carries a significantly higher financial risk for lenders than misclassifying a non-defaulter.

---

##  Key Features & Highlights
* **Comprehensive Exploratory Data Analysis (EDA):** Identified key risk factors including payment delays (`PAY_0` to `PAY_6`), credit limits (`LIMIT_BAL`), and age distributions.
* **Domain-Specific Feature Engineering:** Engineered 10 custom financial stress features:
  * `financial_stress_score`: Aggregated score combining payment delays, high utilization, and late payments.
  * `credit_utilization`: Ratio of billing amount to the total credit limit.
  * `late_payment_count`: Number of delayed payments across historical cycles.
  * `avg_pay_delay` & `max_delay`: Average and maximum delay in months.
  * `payment_efficiency` & `debt_growth`: Cash flow and debt movement indicators.
* **Class Imbalance Handling:** Addressed data imbalance (~78% non-defaulters vs. ~22% defaulters) using `class_weight='balanced'` across models.
* **Model Benchmark & Comparison:** Evaluated Logistic Regression, Support Vector Classifier (SVC), and Random Forest.

---

## 📊 Dataset & Features
* **Dataset:** [UCI Credit Card Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) (30,000 samples, 25 original columns expanded to 34 columns via feature engineering).
* **Target Variable:** `default.payment.next.month` (0 = No Default, 1 = Default).

---

##  Model Performance & Results

| Model | Accuracy | Defaulter Recall (Class 1) | Defaulter Precision (Class 1) | F1-Score (Class 1) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | **82.01%** | 0.35 | **0.68** | 0.46 |
| **SVM (SVC)** | 76.96% | 0.59 | 0.48 | **0.53** |
| **Logistic Regression** | 75.48% | **0.61** | 0.46 | 0.52 |

### Key Findings & Feature Importance:
1. **Top Predictive Features:** According to Random Forest feature importance, engineered features dominated the top signals:
   * **`financial_stress_score`** (~0.052 importance)
   * **`credit_utilization`** (~0.044 importance)
   * **`total_payment`** (~0.043 importance)
   * **`late_payment_count`** (~0.043 importance)
2. **Model Selection Strategy:** 
   * **Logistic Regression** achieved the highest **Recall (61%)** for defaulters, making it ideal if the primary goal is capturing max risk.
   * **Random Forest** achieved the highest overall **Accuracy (82.01%)** and highest **Precision (68%)**, minimizing false alarms.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Processing & EDA:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (`LogisticRegression`, `RandomForestClassifier`, `SVC`, `StandardScaler`, `train_test_split`)

---

##  How to Run
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/MaiAlhusseini/credit-card-default-prediction.git](https://github.com/MaiAlhusseini/credit-card-default-prediction.git)
   cd credit-card-default-prediction
