import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_credit_score_distribution(data):
    """
    Plot credit score distribution
    """
    credit_score_counts = data['credit_score_band'].value_counts().reset_index()
    credit_score_counts.columns = ['Credit Score Band', 'Count']
    
    # Create a color map
    color_map = {
        'Poor': 'red',
        'Fair': 'orange',
        'Good': 'lightgreen',
        'Excellent': 'darkgreen'
    }
    
    # Create a bar plot with Plotly
    fig = px.bar(
        credit_score_counts, 
        x='Credit Score Band', 
        y='Count',
        color='Credit Score Band',
        color_discrete_map=color_map,
        title='Credit Score Distribution',
        labels={'Count': 'Number of Customers'}
    )
    
    fig.update_layout(xaxis_title='Credit Score Band', yaxis_title='Number of Customers')
    
    return fig

def plot_fraud_distribution(data):
    """
    Plot fraud distribution
    """
    fraud_counts = data['fraud_flag'].value_counts().reset_index()
    fraud_counts.columns = ['Fraud Flag', 'Count']
    
    # Map the fraud flag values
    fraud_counts['Fraud Flag'] = fraud_counts['Fraud Flag'].map({0: 'Legitimate', 1: 'Fraudulent'})
    
    # Create a pie chart with Plotly
    fig = px.pie(
        fraud_counts,
        values='Count',
        names='Fraud Flag',
        title='Distribution of Fraudulent vs Legitimate Transactions',
        color='Fraud Flag',
        color_discrete_map={'Legitimate': 'green', 'Fraudulent': 'red'}
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig

def plot_transaction_trends(data):
    """
    Plot transaction trends over time
    """
    # Group by week and calculate averages
    weekly_trends = data.groupby('Week_Start_Date').agg({
        'Total_Trans_Amt': 'mean',
        'Total_Trans_Ct': 'mean',
        'fraud_flag': 'mean'
    }).reset_index()
    
    weekly_trends['fraud_percentage'] = weekly_trends['fraud_flag'] * 100
    
    # Create a figure with two y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    fig.add_trace(
        go.Scatter(
            x=weekly_trends['Week_Start_Date'],
            y=weekly_trends['Total_Trans_Amt'],
            name='Avg Transaction Amount',
            line=dict(color='blue')
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=weekly_trends['Week_Start_Date'],
            y=weekly_trends['Total_Trans_Ct'],
            name='Avg Transaction Count',
            line=dict(color='green')
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=weekly_trends['Week_Start_Date'],
            y=weekly_trends['fraud_percentage'],
            name='Fraud Percentage',
            line=dict(color='red')
        ),
        secondary_y=True
    )
    
    # Set titles
    fig.update_layout(
        title_text='Transaction Trends Over Time',
        xaxis_title='Date'
    )
    
    fig.update_yaxes(title_text='Amount / Count', secondary_y=False)
    fig.update_yaxes(title_text='Fraud Percentage (%)', secondary_y=True)
    
    return fig

def plot_credit_score_by_age(data):
    """
    Plot credit score by age group
    """
    # Create age groups
    data['Age_Group'] = pd.cut(
        data['Customer_Age'],
        bins=[0, 30, 40, 50, 60, 100],
        labels=['<30', '30-40', '40-50', '50-60', '60+']
    )
    
    # Group by age group and credit score
    age_credit = data.groupby(['Age_Group', 'credit_score_band']).size().reset_index(name='Count')
    
    # Create a grouped bar plot
    fig = px.bar(
        age_credit,
        x='Age_Group',
        y='Count',
        color='credit_score_band',
        title='Credit Score Distribution by Age Group',
        barmode='group',
        color_discrete_map={
            'Poor': 'red',
            'Fair': 'orange',
            'Good': 'lightgreen',
            'Excellent': 'darkgreen'
        }
    )
    
    fig.update_layout(xaxis_title='Age Group', yaxis_title='Number of Customers')
    
    return fig

def plot_income_credit_limit_relation(data):
    """
    Plot relationship between income and credit limit
    """
    # Create a sample if the dataset is too large
    if len(data) > 1000:
        sample_data = data.sample(1000, random_state=42)
    else:
        sample_data = data
    
    # Create a scatter plot
    fig = px.scatter(
        sample_data,
        x='Income',
        y='Credit_Limit',
        color='credit_score_band',
        title='Relationship Between Income and Credit Limit',
        labels={'Income': 'Income', 'Credit_Limit': 'Credit Limit'},
        color_discrete_map={
            'Poor': 'red',
            'Fair': 'orange',
            'Good': 'lightgreen',
            'Excellent': 'darkgreen'
        }
    )
    
    fig.update_layout(xaxis_title='Income', yaxis_title='Credit Limit')
    
    return fig

def plot_model_metrics(model_evaluation, model_type='credit'):
    """
    Plot model evaluation metrics
    """
    if not model_evaluation:
        return None
    
    if model_type == 'credit':
        title = 'Credit Scoring Model Performance'
    else:
        title = 'Fraud Detection Model Performance'
    
    # Extract classification report
    report = model_evaluation['classification_report']
    
    # Prepare data for plotting
    metrics = []
    for class_name, values in report.items():
        if isinstance(values, dict):  # Skip the 'accuracy' key
            metrics.append({
                'Class': class_name,
                'Precision': values['precision'],
                'Recall': values['recall'],
                'F1-Score': values['f1-score']
            })
    
    metrics_df = pd.DataFrame(metrics)
    
    # Create a grouped bar plot for metrics
    fig = px.bar(
        metrics_df,
        x='Class',
        y=['Precision', 'Recall', 'F1-Score'],
        barmode='group',
        title=title
    )
    
    fig.update_layout(xaxis_title='Class', yaxis_title='Score')
    
    return fig

def plot_feature_importance(model_evaluation):
    """
    Plot feature importance
    """
    if not model_evaluation or 'feature_importance' not in model_evaluation:
        return None
    
    feature_importance = model_evaluation['feature_importance']
    
    # Take the top 10 features
    top_features = feature_importance.head(10).sort_values('importance')
    
    # Create a horizontal bar plot
    fig = px.bar(
        top_features,
        y='feature',
        x='importance',
        orientation='h',
        title='Top 10 Feature Importance'
    )
    
    fig.update_layout(xaxis_title='Importance', yaxis_title='Feature')
    
    return fig

def plot_confusion_matrix(model_evaluation):
    """
    Plot confusion matrix
    """
    if not model_evaluation or 'confusion_matrix' not in model_evaluation:
        return None
    
    conf_matrix = model_evaluation['confusion_matrix']
    
    # Create a heatmap
    fig = px.imshow(
        conf_matrix,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Class 0', 'Class 1', 'Class 2', 'Class 3'],
        y=['Class 0', 'Class 1', 'Class 2', 'Class 3'],
        title='Confusion Matrix'
    )
    
    fig.update_layout(xaxis_title='Predicted', yaxis_title='Actual')
    
    return fig

def plot_customer_credit_profile(customer_data, credit_pred, fraud_pred):
    """
    Plot customer credit profile
    """
    # Create a figure with subplots
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
    
    # Fraud probability gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=fraud_pred['fraud_probability'] * 100,
            title={'text': "Fraud Risk"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "red"}
                ]
            }
        ),
        row=1, col=1
    )
    
    # Credit score gauge
    credit_score_value = list(credit_pred['all_probabilities'].values())
    credit_score_labels = list(credit_pred['all_probabilities'].keys())
    
    fig.add_trace(
        go.Pie(
            values=credit_score_value,
            labels=credit_score_labels,
            hole=0.3,
            title="Credit Score Probability"
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="Customer Credit Profile",
        height=400
    )
    
    return fig
