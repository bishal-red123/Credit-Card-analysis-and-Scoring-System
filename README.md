# 💳 Credit Card Analysis & Recommendation System

## 📌 Project Overview

This project combines data analytics and machine learning to offer a comprehensive solution for analyzing credit card usage and recommending financial products. Initially built using Power BI for customer behavior analysis and segmentation, the project has now been extended to include backend machine learning models for credit score prediction, fraud detection, and credit card recommendation.

---

## 🎯 Key Features

### 🧠 Machine Learning Models
- **Credit Score Prediction**  
  Predicts a customer’s creditworthiness based on their profile and behavior.

- **Fraud Detection**  
  Detects anomalies in transactions that may indicate fraudulent activity.

- **Credit Card Recommendation**  
  Suggests personalized credit cards using a trained recommendation system with scaled features.

### 📊 Power BI Dashboards
- **Customer Segmentation & Spending Trends**  
  Visual dashboards showing demographic trends, card usage patterns, and risk zones.

- **Interactive Filters**  
  Filter data by income group, transaction type, geography, and credit card usage.

---

## 🛠 Technologies Used

- **SQL** – Data extraction and transformation
- **Power BI** – Dashboard creation and data visualization
- **Python + scikit-learn** – Model development and serialization
- **Joblib** – Model and scaler persistence
- **Docker** – For deploying machine learning services

---

## 📁 Project Files

| File                             | Description                                                           |
|----------------------------------|-----------------------------------------------------------------------|
| `credit_score_model.joblib`      | Trained model for predicting credit scores                           |
| `fraud_detection_model.joblib`   | Trained fraud detection model                                        |
| `credit_card_recommender_scaler.joblib` | Scaler used for normalizing features in the recommendation model |
| `Dockerfile`                     | Environment and deployment setup for containerized service           |
| `credit_dashboard.pbix`          | Power BI file for interactive visualization of customer insights     |

---

## 🚀 How to Use

1. **Run Power BI Dashboard**  
   Open `credit_dashboard.pbix` in Power BI Desktop.

2. **Use Models in Python**  
   ```python
   import joblib
   model = joblib.load("credit_score_model.joblib")
   prediction = model.predict([user_data])
Deploy via Docker

bash
Copy
Edit
docker build -t credit-analyzer .
docker run -p 5000:5000 credit-analyzer
🧩 Future Enhancements
Integrate with a Streamlit UI for real-time model interaction

Connect Power BI to live model APIs

Implement explainable AI using SHAP or LIME

Deploy to AWS/GCP with CI/CD pipelines

👨‍💻 Author
Bishal Mondal
GitHub: bishal-red123
