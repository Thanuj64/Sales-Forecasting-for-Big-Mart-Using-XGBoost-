# 🛒 BigMart Sales Prediction Dashboard

An AI-powered **Retail Sales Forecasting Web Application** developed using **Machine Learning (XGBoost Regression)** and **Streamlit**.

This system predicts product sales based on product and outlet characteristics through an interactive industry-level dashboard.

---

## 🚀 Project Overview

Retail businesses require accurate demand forecasting to manage inventory and improve profitability.  
The **BigMart Sales Prediction Dashboard** provides an intelligent machine learning solution that predicts future product sales using historical retail data.

This project demonstrates a complete **End-to-End Machine Learning Pipeline**, including:

- Data Preprocessing
- Feature Engineering
- Model Training using XGBoost
- Model Serialization
- Interactive Dashboard Development
- Batch Prediction System

---

## 🎯 Features

✅ Manual Sales Prediction  
✅ CSV Batch Prediction  
✅ Automatic Sales Categorization (Low / Medium / High)  
✅ Industry-Level Streamlit UI  
✅ Download Prediction Results  
✅ Fast Model Loading using Cache  

---

## 🧠 Machine Learning Model

**Algorithm Used:** XGBoost Regression  

**Learning Type:** Supervised Learning  
**Problem Type:** Regression  

### Why XGBoost?
- Handles missing values efficiently
- Prevents overfitting using regularization
- Captures complex nonlinear relationships
- High prediction accuracy
- Fast training and inference

---

## 🖥️ Application Modules

### 🔹 Manual Entry Prediction
Users manually enter product and outlet details to instantly predict sales.

### 🔹 CSV Upload Prediction
Users can upload datasets to:
- Predict multiple product sales
- Automatically classify sales into:

| Sales Value | Category |
|-------------|----------|
| < 2000 | Low |
| 2000 – 4000 | Medium |
| > 4000 | High |
