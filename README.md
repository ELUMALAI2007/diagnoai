# AI-Based Disease Prediction System

## 1. Introduction

The **AI-Based Disease Prediction System** is an advanced healthcare-oriented web application designed to assist in early-stage disease identification using Machine Learning techniques. The system integrates two independent predictive modules:

1. **Symptom-Based Disease Prediction**
   This module accepts multiple user-input symptoms and predicts possible diseases among **200+ medical conditions** using a high-performance **Random Forest Classifier**.

2. **Direct Disease Risk Prediction**
   This module evaluates structured medical parameters such as Age, Blood Pressure, Sugar Level, BMI, and Cholesterol to predict the likelihood of **50 major diseases**, including cardiovascular, metabolic, and organ-related disorders.

The system aims to provide a **preliminary diagnostic support tool** and enhance accessibility to health insights.

---

## 2. Problem Statement

Accurate disease diagnosis based on multiple symptoms and medical parameters is a complex and time-consuming process. Manual analysis may lead to inconsistencies due to human limitations, especially when dealing with hundreds of diseases.

There is a need for an **automated, scalable, and intelligent system** that can:

* Analyze large combinations of symptoms
* Process structured health data
* Provide quick and reliable preliminary predictions

---

## 3. Objectives

* Develop a **multi-class classification model** capable of predicting over 200 diseases from symptom inputs.
* Design a **structured prediction system** for 50 major diseases using medical attributes.
* Generate **large-scale, medically consistent datasets** for training (up to 100,000 samples).
* Build a **secure and user-friendly web interface** with authentication.
* Deliver **top-3 predictions with probability scores** and health suggestions.

---

## 4. Theoretical Background

### 4.1 Random Forest Classifier

Random Forest is an ensemble learning algorithm that constructs multiple decision trees and combines their outputs to improve prediction accuracy and reduce overfitting.

**Mathematical Representation:**
[
f(x) = \text{mode} { T_1(x), T_2(x), ..., T_n(x) }
]

Where:

* (T_i(x)) represents individual decision trees
* Final output is the majority vote of all trees

**Justification:**

* Handles high-dimensional data efficiently
* Suitable for multi-class classification (200+ diseases)
* Reduces overfitting compared to single decision trees

---

### 4.2 Logistic Regression

Logistic Regression is used to model the probability of a categorical dependent variable.

**Equation:**
[
p(X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + ... + \beta_nX_n)}}
]

**Justification:**

* Efficient for structured numerical datasets
* Provides probabilistic outputs
* Interpretable model for medical data

---

## 5. Dataset Description

### Dataset 1: Symptom-Based Dataset (`symptom_data.csv`)

* **Size**: ~60,000 samples
* **Diseases**: 200+ classes
* **Features**: 120 symptoms (binary encoded)
* **Type**: Multi-label symptom representation

### Dataset 2: Direct Disease Dataset (`direct_disease_data.csv`)

* **Size**: ~30,000 samples
* **Diseases**: 50 major diseases
* **Features**:

  * Age
  * Gender
  * Blood Pressure (Systolic)
  * Sugar Level
  * Cholesterol
  * BMI
  * Heart Rate

**Data Characteristics:**

* Balanced distribution
* Medically logical relationships
* No random or noisy data

---

## 6. System Architecture & Implementation

* **Backend Framework**: Flask
* **Machine Learning Library**: Scikit-learn
* **Database**: SQLite (for authentication)
* **Frontend**: HTML, CSS, Bootstrap

### Workflow:

1. User Login → Authentication via SQLite
2. Input Symptoms / Medical Data
3. Data Preprocessing
4. Model Prediction
5. Output Display (Top-3 diseases + probability + suggestions)

---

## 7. Model Training & Evaluation

### Models Used:

* Random Forest (Primary Model)
* Decision Tree (Comparison)
* Naive Bayes (Baseline)
* Logistic Regression (Structured Data)

### Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1 Score

### Visualization:

* Confusion Matrix
* Feature Importance Graph
* Performance Comparison Chart

---

## 8. Results

* Random Forest achieved the **highest accuracy and stability** for symptom-based predictions.
* Logistic Regression provided **interpretable results** for structured medical data.
* The system successfully predicts:

  * **Top 3 diseases**
  * **Probability scores**
  * **Risk levels**

---

## 9. Future Enhancements

* Integration with **real hospital datasets**
* Deployment using **cloud platforms (AWS/Render)**
* Adding **deep learning models (Neural Networks)**
* Real-time **API integration with wearable devices**
* Multi-language support for accessibility

---

## 10. Execution Steps

1. Navigate to the project folder
2. Run the Flask server:

   ```
   python app.py
   ```
3. Open browser:

   ```
   http://127.0.0.1:5000/
   ```

---

## 11. Conclusion

The project demonstrates how Machine Learning can be effectively used to build a scalable and intelligent healthcare support system. It combines high-dimensional classification with structured prediction models, offering a practical solution for preliminary disease diagnosis.
