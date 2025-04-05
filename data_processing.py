import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

def load_data():
    """
    Load credit card and customer data from CSV files
    """
    try:
        # Load credit card transaction data
        credit_card_data = pd.read_csv('attached_assets/credit_card.csv')
        cc_add_data = pd.read_csv('attached_assets/cc_add.csv')
        
        # Load customer data
        customer_data = pd.read_csv('attached_assets/customer.csv')
        cust_add_data = pd.read_csv('attached_assets/cust_add.csv')
        
        # Combine the datasets
        cc_data = pd.concat([credit_card_data, cc_add_data], ignore_index=True)
        cust_data = pd.concat([customer_data, cust_add_data], ignore_index=True)
        
        # Merge credit card and customer data on Client_Num
        merged_data = pd.merge(cc_data, cust_data, on='Client_Num', how='inner')
        
        return merged_data
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(data):
    """
    Preprocess the credit card and customer data
    """
    if data is None:
        return None, None, None
    
    # Convert date column to datetime
    data['Week_Start_Date'] = pd.to_datetime(data['Week_Start_Date'], format='%d-%m-%Y', errors='coerce')
    
    # Handle missing values
    data = data.fillna({
        'Total_Revolving_Bal': 0, 
        'Avg_Utilization_Ratio': 0,
        'Delinquent_Acc_x': 0,
        'Personal_loan': 'no'
    })
    
    # Create a target variable for credit scoring based on multiple factors
    # Higher score means better creditworthiness
    data['credit_score'] = 0
    
    # Factors that positively affect credit score
    data.loc[data['Income'] > 50000, 'credit_score'] += 10  # Higher income
    data.loc[data['Total_Revolving_Bal'] < 1000, 'credit_score'] += 10  # Low revolving balance
    data.loc[data['Avg_Utilization_Ratio'] < 0.3, 'credit_score'] += 10  # Low utilization ratio
    data.loc[data['Delinquent_Acc_x'] == 0, 'credit_score'] += 20  # No delinquent accounts
    data.loc[data['House_Owner'] == 'yes', 'credit_score'] += 5  # Homeowner
    data.loc[data['Personal_loan'] == 'no', 'credit_score'] += 5  # No personal loans
    data.loc[data['Customer_Age'] > 40, 'credit_score'] += 5  # Mature customer
    data.loc[data['Cust_Satisfaction_Score'] >= 4, 'credit_score'] += 5  # High satisfaction

    # Create credit score bands
    data['credit_score_band'] = pd.cut(
        data['credit_score'], 
        bins=[0, 20, 40, 60, 70], 
        labels=['Poor', 'Fair', 'Good', 'Excellent']
    )
    
    # Create fraud flags based on suspicious patterns
    # This is a simplified version - in reality, we would use more sophisticated methods
    data['fraud_flag'] = 0
    
    # Suspicious patterns
    data.loc[(data['Total_Trans_Amt'] > 10000) & (data['Total_Trans_Ct'] < 20), 'fraud_flag'] = 1  # High amount, low count
    data.loc[(data['Avg_Utilization_Ratio'] > 0.9), 'fraud_flag'] = 1  # Very high utilization
    data.loc[(data['Total_Trans_Amt'] > 3 * data['Income']), 'fraud_flag'] = 1  # Transactions much higher than income
    
    # Feature engineering
    data['Trans_Avg'] = data['Total_Trans_Amt'] / data['Total_Trans_Ct'].replace(0, 1)
    data['Income_to_Credit_Ratio'] = data['Income'] / data['Credit_Limit'].replace(0, 1)
    data['Revolving_to_Credit_Ratio'] = data['Total_Revolving_Bal'] / data['Credit_Limit'].replace(0, 1)
    
    # Convert categorical variables to dummy variables
    categorical_cols = ['Card_Category', 'Use_Chip', 'Exp_Type', 'Gender', 
                        'Education_Level', 'Marital_Status', 'Customer_Job']
    
    # Convert categorical columns
    for col in categorical_cols:
        if col in data.columns:
            # Use pandas get_dummies for one-hot encoding
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
            data = pd.concat([data, dummies], axis=1)
    
    # Select features for modeling
    credit_score_features = [
        'Customer_Age', 'Dependent_Count', 'Income', 'Credit_Limit',
        'Total_Revolving_Bal', 'Total_Trans_Amt', 'Total_Trans_Ct',
        'Avg_Utilization_Ratio', 'Trans_Avg', 'Income_to_Credit_Ratio',
        'Revolving_to_Credit_Ratio'
    ]
    
    # Add dummy variables to features
    for col in data.columns:
        if any(cat in col for cat in categorical_cols) and col not in categorical_cols:
            credit_score_features.append(col)
    
    # Create feature matrix for credit scoring
    X_credit = data[credit_score_features].copy()
    y_credit = data['credit_score_band'].copy()
    
    # Create feature matrix for fraud detection
    X_fraud = data[credit_score_features].copy()
    y_fraud = data['fraud_flag'].copy()
    
    # Standardize the features
    scaler = StandardScaler()
    X_credit_scaled = scaler.fit_transform(X_credit)
    X_fraud_scaled = scaler.fit_transform(X_fraud)
    
    # Convert back to DataFrame to keep column names
    X_credit_scaled = pd.DataFrame(X_credit_scaled, columns=X_credit.columns)
    X_fraud_scaled = pd.DataFrame(X_fraud_scaled, columns=X_fraud.columns)
    
    # Handle imbalanced data for fraud detection with SMOTE
    smote = SMOTE(random_state=42)
    X_fraud_resampled, y_fraud_resampled = smote.fit_resample(X_fraud_scaled, y_fraud)
    
    # Prepare data splits for both models
    X_credit_train, X_credit_test, y_credit_train, y_credit_test = train_test_split(
        X_credit_scaled, y_credit, test_size=0.2, random_state=42
    )
    
    X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(
        X_fraud_resampled, y_fraud_resampled, test_size=0.2, random_state=42
    )
    
    credit_data = {
        'X_train': X_credit_train,
        'X_test': X_credit_test,
        'y_train': y_credit_train,
        'y_test': y_credit_test,
        'scaler': scaler,
        'features': credit_score_features
    }
    
    fraud_data = {
        'X_train': X_fraud_train,
        'X_test': X_fraud_test,
        'y_train': y_fraud_train,
        'y_test': y_fraud_test,
        'scaler': scaler,
        'features': credit_score_features
    }
    
    return data, credit_data, fraud_data

def get_customer_profile(data, client_num):
    """
    Get a customer profile by client number
    """
    if client_num in data['Client_Num'].values:
        return data[data['Client_Num'] == client_num].iloc[0]
    return None

def get_recent_transactions(data, client_num, n=10):
    """
    Get recent transactions for a customer
    """
    if client_num in data['Client_Num'].values:
        customer_data = data[data['Client_Num'] == client_num]
        return customer_data.sort_values('Week_Start_Date', ascending=False).head(n)
    return None

def get_aggregate_stats(data):
    """
    Get aggregate statistics from the data
    """
    stats = {
        'total_customers': data['Client_Num'].nunique(),
        'total_transactions': data['Total_Trans_Ct'].sum(),
        'total_transaction_amount': data['Total_Trans_Amt'].sum(),
        'avg_credit_limit': data['Credit_Limit'].mean(),
        'avg_utilization': data['Avg_Utilization_Ratio'].mean(),
        'fraud_rate': data['fraud_flag'].mean() * 100,
        'credit_score_distribution': data['credit_score_band'].value_counts().to_dict()
    }
    return stats
