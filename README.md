# 🍷 Wine Quality Classification Project (White Wine)

## Overview

This project implements a **Random Forest Classifier** to predict the quality category of white wines based on their physicochemical properties. The goal is to obtain the best possible performance on unseen data by applying robust training methods, including stratified splitting and hyperparameter tuning.

---

## 📊 Dataset: Wine Quality (White)

* **Source:** UCI Machine Learning Repository
* **Target Variable (Y):** The original continuous 'quality' score (3-9) was transformed into three discrete classes for classification:
    * **Low Quality:** Quality $\le 5$
    * **Medium Quality:** Quality $= 6$
    * **High Quality:** Quality $\ge 7$
* **Features (X):** 11 physicochemical variables (e.g., fixed acidity, residual sugar, pH, alcohol content, etc.).
* **Challenge:** The dataset presents a significant **class imbalance**, with the 'Medium Quality' class being the majority.

---

## ⚙️ Methodology

### 1. Preprocessing and Data Preparation

* **Handling Missing Values:** The dataset was verified to have no significant missing values.
* **Target Transformation:** The continuous 'quality' variable was transformed into the three-level categorical variable (Low, Medium, High).
* **Data Splitting:** The data was split into **80% Training** and **20% Test** sets. **Stratified splitting** was used to ensure the class distribution is maintained in both sets.
* **Addressing Imbalance (Optional/Recommended):** Techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) were applied to the training set to mitigate the class imbalance and prevent the model from ignoring minority classes.

### 2. Model Implementation: Random Forest Classifier

The **Random Forest Classifier** was chosen due to its robustness, efficiency in handling non-linear data relationships, and its ability to manage numerical features without requiring extensive feature scaling.

### 3. Training and Optimization Methods (Hyperparameter Tuning)

To achieve the "best performance on unseen data," the following strategy was employed:

* **Technique:** **Grid Search** (`GridSearchCV` from scikit-learn).
* **Validation:** **$k$-fold Cross-Validation** ($k=5$) was performed on the training set.
* **Optimization Metric:** The primary metric used for optimization was the **Weighted F1-Score**. This metric provides a balanced measure of Precision and Recall, and its *weighted* average accounts for the class imbalance. 

[Image of the F1 Score formula]

* **Parameters Explored:** Key hyperparameters tuned included:
    * `n_estimators` (number of trees)
    * `max_depth` (maximum depth of the trees)
    * `min_samples_leaf` (minimum samples required at a leaf node)

---

## 📈 Results and Evaluation

The final optimized model was evaluated on the unseen **Test Set** (20% of the data).

### 1. Key Metrics

The results are reported using the **Classification Report**, which provides Precision, Recall, and F1-Score for each class, along with the overall weighted average.

| Metric | Goal |
| :--- | :--- |
| **Weighted F1-Score** | Primary metric; close to **1.0** indicates excellent performance. |
| **Confusion Matrix** | Visual validation of classification accuracy across all three classes (Low, Medium, High). | 

### 2. Best Hyperparameters Found

(Placeholder: Insert the actual results from `grid_search.best_params_` here, e.g.)
* `n_estimators`: 200
* `max_depth`: 15
* `min_samples_split`: 5

### 3. Final Performance (Test Set)

(Placeholder: Insert the final results here, e.g.)
* **Weighted F1-Score:** 0.785
* **Accuracy:** 76.5%

---

## 📦 Requirements

This project requires the following Python libraries (typically managed via Anaconda or installed via pip):

* `pandas`
* `numpy`
* `scikit-learn` (sklearn)
* `matplotlib` / `seaborn`
* `imblearn` (optional, but highly recommended for SMOTE)