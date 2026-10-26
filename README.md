# 🏦 Customer Churn Prediction & Business Threshold Tuning



A complete End-to-End Machine Learning project predicting bank customer churn using **Logistic Regression**, addressing class imbalance, and applying **Custom Probability Threshold Tuning** to align with business costs.



---



## 📌 Project Overview

Customer churn is a critical challenge for retail banks. This project builds a predictive model to identify customers likely to leave the bank, moving beyond raw accuracy to optimize **Precision, Recall, and Decision Thresholds** based on real-world business constraints.



---



## 🛠️ Key Pipeline Steps



1. **Data Preprocessing & Cleaning:**

&#x20;  * Dropped high-cardinality/irrelevant identifiers (`CustomerId`, `Surname`).

&#x20;  * One-hot encoded categorical features (e.g., `Geography`).

2. **Feature Scaling (Preventing Data Leakage):**

&#x20;  * Implemented Scikit-Learn's `StandardScaler` fitted **only** on the training set (`X_train`) and transformed onto the testing set (`X_test`) to ensure a clean evaluation.

3. **Mitigating Class Imbalance:**

&#x20;  * Diagnosed the "accuracy trap" where default models miss minority churners.

&#x20;  * Applied `class_weight='balanced'` to penalize missed churns and boost initial recall.

4. **Custom Probability Threshold Tuning:**

&#x20;  * Moved away from the rigid default 0.5 decision threshold.

&#x20;  * Explored probabilities using `model.predict_proba()` and tuned the threshold to **0.6** to drastically reduce false alarms (false positives) while maintaining a strong capture rate of actual churners.



---



## 📊 Final Model Results (Threshold = 0.6)

* **Accuracy:** 78%

* **Churn Recall (Class 1):** 58% (Balanced to catch critical at-risk customers)

* **False Positives Reduced:** Optimized via threshold tuning from 448 down to 269, protecting the bank's marketing/retention budget from wasted outreach.



---



## ⚙️ Tech Stack

* **Language:** Python 3.x

* **Libraries:** 

&#x20; * `pandas`, `numpy` (Data manipulation)

&#x20; * `scikit-learn` (Preprocessing, Logistic Regression, Metrics)

