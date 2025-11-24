# 🍷 Wine Quality Classification Project (White Wine)

## Overview

This project implements a **Random Forest Classifier** to predict the quality of white wines based on their physicochemical properties. The goal is to obtain the best possible performance on unseen data by applying robust training methods, including stratified splitting and hyperparameter tuning.

The problem is modeled as a **binary classification task**.

---

## 📊 Dataset: Wine Quality (White)

* **Source:** UCI Machine Learning Repository
* **Target Variable (Y):** The original continuous 'quality' score (3-9) was transformed into a binary class for classification:
    * **Class 0 (Standard Quality):** Quality score $\le 6$
    * **Class 1 (High Quality):** Quality score $> 6$
* **Features (X):** 11 physicochemical variables (e.g., fixed acidity, residual sugar, pH, alcohol content, etc.).
* **Challenge:** The dataset presents a class imbalance, with the 'Standard Quality' class being the majority.

---

## ⚙️ Methodology

### 1. Preprocessing and Data Preparation

* **Handling Missing Values:** The dataset was verified to have no significant missing values.
* **Target Transformation:** The continuous 'quality' variable was transformed into the binary `quality_label` (0 or 1).
* **Data Splitting:** The data was split into **80% Training** and **20% Test** sets. **Stratified splitting** (`stratify=y`) was used to ensure the class distribution is maintained in both sets, which is crucial for imbalanced datasets.
* **Feature Scaling:** All features were standardized using `StandardScaler` to ensure they have a mean of 0 and a standard deviation of 1. This helps the model converge and perform better.

### 2. Model Implementation: Random Forest Classifier

The **Random Forest Classifier** was chosen due to its robustness, efficiency in handling non-linear data relationships, and its ability to manage numerical features without requiring extensive feature scaling (though scaling is still a good practice).

### 3. Training and Optimization Methods (Hyperparameter Tuning)

To achieve the "best performance on unseen data," the following strategy was employed:

* **Technique:** **Grid Search** (`GridSearchCV` from scikit-learn).
* **Validation:** **$k$-fold Cross-Validation** ($k=5$) was performed on the training set.
* **Optimization Metric:** The primary metric used for optimization was **Accuracy**. This metric measures the proportion of correctly classified instances.

* **Parameters Explored:** Key hyperparameters tuned included:
    * `n_estimators` (number of trees)
    * `max_depth` (maximum depth of the trees)
    * `min_samples_split` (minimum samples to split a node)
    * `min_samples_leaf` (minimum samples required at a leaf node)
    * `max_features` (number of features to consider for the best split)

---

## 📈 Results and Evaluation

The final optimized model was evaluated on the unseen **Test Set** (20% of the data).

### 1. Key Metrics

The results are reported using the **Classification Report**, which provides Precision, Recall, and F1-Score for each class, along with the overall accuracy and ROC-AUC score.

| Metric | Goal |
| :--- | :--- |
| **Accuracy** | Primary optimization metric; a high value is desired. |
| **ROC-AUC Score** | Measures the model's ability to distinguish between classes. A value close to **1.0** is excellent. |
| **Confusion Matrix** | Visual validation of classification accuracy for both classes (Standard and High). | 

### 2. Best Hyperparameters Found

(Placeholder: Insert the actual results from `grid_search.best_params_` here, e.g.)
* `n_estimators`: 200
* `max_depth`: 20
* `min_samples_leaf`: 1
* `min_samples_split`: 5
* `max_features`: 'sqrt'

### 3. Final Performance (Test Set)

(Placeholder: Insert the final results from the script's output here, e.g.)
* **Accuracy:** 0.88
* **ROC-AUC Score:** 0.85

---

## 📦 Requirements

This project requires the following Python libraries (typically managed via Anaconda or installed via pip):

* `pandas`
* `numpy`
* `scikit-learn` (sklearn)
* `matplotlib` / `seaborn`