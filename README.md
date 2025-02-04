# Machine Learning Projects

## Overview
This repository contains various machine learning models applied to different datasets, focusing on:
- **House Price Prediction** using Linear Regression
- **Country Clustering** using K-Means and Mean Shift
- **NBA Player Classification** using Logistic Regression, Naïve Bayes, and Neural Networks

Each project involves:
1. **Data Preprocessing**
2. **Feature Selection**
3. **Model Training and Evaluation**
4. **Results Interpretation and Visualization**

---

## 📂 Project Structure

---

## 🔍 Detailed Findings

### **1️⃣ House Price Prediction**
- **Files:** `TASK1 -2.py`, `TASK 1 MULTIPLE VARIABLES-2.py`
- **Dataset:** `houseprice_data.csv`
- **Methods:** Linear Regression (Single & Multiple Variables)
- **Findings:**
  - **Single Variable Model:**
    - Explored the relationship between `sqft_lot` (size of the room) and `house price`.
    - **Results:** Larger apartments tend to have higher value, as seen from the regression coefficient.
  - **Multiple Variable Model:**
    - Included `grade` as an additional feature.
    - **Key Result:** 
      - The model predicted that **for every 1-unit increase in grade, house price increases by $98,680.**
      - **For every 1-unit increase in `sqft_lot`, price increases by $177.**
      - **R² Score: 0.54**, indicating a moderate relationship.
  - **Improvement Suggestions:**
    - Normalizing the dataset
    - Adding more independent variables (economic conditions, crime rates)
    - Using polynomial regression for better accuracy

---

### **2️⃣ Country Clustering Analysis**
- **Files:** `TASK 2-2.py`, `TASK 2 (3D) -2.py`, `TASK 2 Mean SHIFT-1.py`
- **Dataset:** `country_data.csv`
- **Methods:** K-Means & Mean Shift Clustering
- **Findings:**
  - **K-Means Clustering (2 Variables)**
    - Created 3 clusters using `Health` and `Child Mortality`
    - The **inverse relationship** between child mortality and health expenditure was evident.
    - **Key Result:** Countries investing more in healthcare had significantly lower child mortality rates.
  - **K-Means (3 Variables)**
    - Included `Income` as a third variable.
    - Clusters became **denser**, indicating income is a major factor in health and mortality.
  - **Mean Shift Clustering**
    - Generated 5 centroids.
    - Showed better separation than K-Means.
  - **Insights:**
    - Countries with **higher income and health expenditure** have **lower child mortality**.
    - The addition of more features made clusters more distinct.

---

### **3️⃣ NBA Player Classification**
- **Files:** `TASK 3-2.py`
- **Dataset:** `nba_rookie_data.csv`
- **Methods:** Logistic Regression, Naïve Bayes, Neural Networks
- **Findings:**
  - **Neural Networks:**
    - Used `Games Played`, `Minutes Played`, `Points Per Game`, `Field Goals Made`, `Free Throws Made`.
    - **Best Accuracy: 68%**.
    - **Mislabelled Points:** 168.
  - **Gaussian Naïve Bayes (GNB):**
    - Accuracy: **68%**, Mislabelled Points: 168.
  - **Logistic Regression:**
    - Accuracy: **67%**, slightly lower than GNB and Neural Networks.
  - **Conclusion:**
    - The models performed similarly with **accuracy between 62%-68%**.
    - **Games Played** was the strongest predictor of NBA success.
    - **Possible Improvements:**
      - Feature engineering (adding new variables).
      - Hyperparameter tuning.

---

## 📌 Key Insights from Report

### **🏠 House Prices**
- The condition (`grade`) and size (`sqft_lot`) significantly influence price.
- The model **underperformed** with an **R² of 0.54**, suggesting more variables are needed.
- **External factors (economic conditions, crime rates, amenities) affect housing prices.**

### **🌍 Country Clustering**
- Countries with **higher healthcare spending have lower child mortality**.
- **Income levels are a major determinant** of both health and mortality.
- **Mean Shift clustering was more effective** than K-Means in this dataset.

### **🏀 NBA Classification**
- **Games Played is the most important predictor** of career longevity.
- Models had **low predictive power (max accuracy 68%)**, meaning more features are needed.

---

## 🚀 Setup Instructions

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/shittukhadijatyetunde/Machine-Learning-Projects.git
cd Machine-Learning-Projects
