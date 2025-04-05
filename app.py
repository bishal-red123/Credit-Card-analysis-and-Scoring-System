import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import threading
import time
import os
import json
import requests

# Import local modules
import data_processing
import credit_scoring
import fraud_detection
import visualization
import api

# Set page configuration
st.set_page_config(
    page_title="Credit Scoring & Fraud Detection System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'credit_data' not in st.session_state:
    st.session_state.credit_data = None
if 'fraud_data' not in st.session_state:
    st.session_state.fraud_data = None
if 'credit_model' not in st.session_state:
    st.session_state.credit_model = credit_scoring.CreditScoreModel()
if 'fraud_model' not in st.session_state:
    st.session_state.fraud_model = fraud_detection.FraudDetectionModel()
if 'credit_evaluation' not in st.session_state:
    st.session_state.credit_evaluation = None
if 'fraud_evaluation' not in st.session_state:
    st.session_state.fraud_evaluation = None
if 'api_running' not in st.session_state:
    st.session_state.api_running = False
if 'api_thread' not in st.session_state:
    st.session_state.api_thread = None

# Function to load and process data
def load_data():
    with st.spinner("Loading and processing data..."):
        # Load data
        data = data_processing.load_data()
        
        if data is not None:
            # Preprocess data
            processed_data, credit_data, fraud_data = data_processing.preprocess_data(data)
            
            # Store in session state
            st.session_state.data = processed_data
            st.session_state.credit_data = credit_data
            st.session_state.fraud_data = fraud_data
            
            st.success("Data loaded and processed successfully!")
            
            # Train or load models
            with st.spinner("Loading models..."):
                if not st.session_state.credit_model.load_model():
                    st.session_state.credit_model.train(credit_data)
                
                if not st.session_state.fraud_model.load_model():
                    st.session_state.fraud_model.train(fraud_data)
                
                # Evaluate models
                st.session_state.credit_evaluation = st.session_state.credit_model.evaluate(credit_data)
                st.session_state.fraud_evaluation = st.session_state.fraud_model.evaluate(fraud_data)
                
                st.success("Models loaded and evaluated successfully!")
        else:
            st.error("Failed to load data. Please check the file paths.")

# Function to start API thread
def start_api_thread():
    if not st.session_state.api_running:
        st.session_state.api_thread = threading.Thread(target=api.start_api)
        st.session_state.api_thread.daemon = True
        st.session_state.api_thread.start()
        st.session_state.api_running = True
        st.success("API started successfully!")
        
        # Wait for API to start
        time.sleep(2)
    else:
        st.info("API is already running.")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Credit Scoring", "Fraud Detection", "Customer Analysis", "Power BI Integration"])

# Main title
st.title("Credit Scoring & Fraud Detection System")

# Home page
if page == "Home":
    st.write("## Welcome to the Credit Scoring & Fraud Detection System")
    st.write("""
    This system helps financial institutions make data-driven decisions about credit applications and 
    detect potential fraudulent activities. Navigate using the sidebar to explore different features of the system.
    """)
    
    st.write("### Features:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Credit Scoring")
        st.write("- Evaluate creditworthiness of applicants")
        st.write("- Predict credit score bands")
        st.write("- Analyze factors affecting credit scores")
    
    with col2:
        st.write("#### Fraud Detection")
        st.write("- Identify suspicious transactions")
        st.write("- Calculate fraud risk probabilities")
        st.write("- Monitor fraud patterns over time")
    
    # Load data button
    if st.button("Load Data and Train Models"):
        load_data()
    
    # Display system status
    st.write("### System Status")
    data_status = "Loaded" if st.session_state.data is not None else "Not Loaded"
    credit_model_status = "Trained" if st.session_state.credit_model.is_trained else "Not Trained"
    fraud_model_status = "Trained" if st.session_state.fraud_model.is_trained else "Not Trained"
    api_status = "Running" if st.session_state.api_running else "Not Running"
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Data", data_status)
    col2.metric("Credit Model", credit_model_status)
    col3.metric("Fraud Model", fraud_model_status)
    col4.metric("API Status", api_status)
    
    # Start API button
    if st.button("Start API for Power BI"):
        start_api_thread()
    
    # API Information
    if st.session_state.api_running:
        st.write("### API Endpoints")
        st.code("""
        GET /api/load_data - Load and preprocess data
        GET /api/customer/<client_num> - Get customer information
        GET /api/credit_score/<client_num> - Get credit score for a customer
        GET /api/fraud_detection/<client_num> - Get fraud detection results for a customer
        GET /api/model_metrics - Get model metrics for Power BI
        GET /api/export_data - Export processed data for Power BI
        """)
        
        st.write("API is available at: http://localhost:8000")

# Data Exploration page
elif page == "Data Exploration":
    st.write("## Data Exploration")
    
    if st.session_state.data is None:
        st.warning("Please load the data first.")
        if st.button("Load Data"):
            load_data()
    else:
        # Display data overview
        st.write("### Data Overview")
        st.write(f"Number of records: {len(st.session_state.data)}")
        st.write(f"Number of unique customers: {st.session_state.data['Client_Num'].nunique()}")
        
        # Display data sample
        st.write("### Data Sample")
        st.dataframe(st.session_state.data.head())
        
        # Display basic statistics
        st.write("### Basic Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Credit Limit", f"${st.session_state.data['Credit_Limit'].mean():.2f}")
            st.metric("Average Revolving Balance", f"${st.session_state.data['Total_Revolving_Bal'].mean():.2f}")
        
        with col2:
            st.metric("Average Transaction Amount", f"${st.session_state.data['Total_Trans_Amt'].mean():.2f}")
            st.metric("Average Transaction Count", f"{st.session_state.data['Total_Trans_Ct'].mean():.2f}")
        
        with col3:
            st.metric("Average Utilization Ratio", f"{st.session_state.data['Avg_Utilization_Ratio'].mean():.2%}")
            st.metric("Fraud Rate", f"{st.session_state.data['fraud_flag'].mean():.2%}")
        
        # Display visualizations
        st.write("### Visualizations")
        tab1, tab2, tab3, tab4 = st.tabs(["Credit Score Distribution", "Fraud Distribution", "Transaction Trends", "Demographics"])
        
        with tab1:
            st.plotly_chart(visualization.plot_credit_score_distribution(st.session_state.data), use_container_width=True)
            st.plotly_chart(visualization.plot_income_credit_limit_relation(st.session_state.data), use_container_width=True)
        
        with tab2:
            st.plotly_chart(visualization.plot_fraud_distribution(st.session_state.data), use_container_width=True)
        
        with tab3:
            st.plotly_chart(visualization.plot_transaction_trends(st.session_state.data), use_container_width=True)
        
        with tab4:
            st.plotly_chart(visualization.plot_credit_score_by_age(st.session_state.data), use_container_width=True)

# Credit Scoring page
elif page == "Credit Scoring":
    st.write("## Credit Scoring Model")
    
    if st.session_state.data is None or not st.session_state.credit_model.is_trained:
        st.warning("Please load the data and train the models first.")
        if st.button("Load Data and Train Models"):
            load_data()
    else:
        # Display model information
        st.write("### Model Information")
        st.write("The credit scoring model evaluates customers' creditworthiness based on their financial behaviors, demographics, and transaction patterns.")
        
        # Display model evaluation metrics
        st.write("### Model Evaluation")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Accuracy", f"{st.session_state.credit_evaluation['accuracy']:.2%}")
        
        with col2:
            avg_f1 = np.mean([metrics['f1-score'] for cls, metrics in st.session_state.credit_evaluation['classification_report'].items() if isinstance(metrics, dict)])
            st.metric("Average F1 Score", f"{avg_f1:.2%}")
        
        # Display feature importance
        st.write("### Feature Importance")
        st.plotly_chart(visualization.plot_feature_importance(st.session_state.credit_evaluation), use_container_width=True)
        
        # Display confusion matrix
        st.write("### Confusion Matrix")
        st.plotly_chart(visualization.plot_confusion_matrix(st.session_state.credit_evaluation), use_container_width=True)
        
        # Score a customer
        st.write("### Score a Customer")
        client_num = st.number_input("Enter Client Number", min_value=1, step=1)
        
        if st.button("Get Credit Score"):
            customer = data_processing.get_customer_profile(st.session_state.data, client_num)
            
            # Get credit score prediction (the model will handle None values)
            prediction = st.session_state.credit_model.predict(customer)
            
            # Check if there was an error in prediction
            if 'error' in prediction:
                st.error(prediction['error'])
            elif customer is None:
                st.error(f"Customer with client number {client_num} not found.")
            else:
                # Customer found and no errors in prediction
                
                # Display customer information
                st.write("#### Customer Information")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Customer Age", customer['Customer_Age'])
                    st.metric("Income", f"${customer['Income']:,}")
                
                with col2:
                    st.metric("Credit Limit", f"${customer['Credit_Limit']:,.2f}")
                    st.metric("Revolving Balance", f"${customer['Total_Revolving_Bal']:,}")
                
                with col3:
                    st.metric("Transaction Amount", f"${customer['Total_Trans_Amt']:,}")
                    st.metric("Transaction Count", customer['Total_Trans_Ct'])
                
                # Display credit score
                st.write("#### Credit Score Prediction")
                st.metric("Credit Score Band", prediction['credit_score_band'])
                st.metric("Confidence", f"{prediction['probability']:.2%}")
                
                # Display credit score probabilities
                proba_df = pd.DataFrame({
                    'Credit Score Band': list(prediction['all_probabilities'].keys()),
                    'Probability': list(prediction['all_probabilities'].values())
                })
                
                fig = px.bar(
                    proba_df,
                    x='Credit Score Band',
                    y='Probability',
                    color='Credit Score Band',
                    title='Credit Score Probabilities',
                    color_discrete_map={
                        'Poor': 'red',
                        'Fair': 'orange',
                        'Good': 'lightgreen',
                        'Excellent': 'darkgreen'
                    }
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Fraud Detection page
elif page == "Fraud Detection":
    st.write("## Fraud Detection Model")
    
    if st.session_state.data is None or not st.session_state.fraud_model.is_trained:
        st.warning("Please load the data and train the models first.")
        if st.button("Load Data and Train Models"):
            load_data()
    else:
        # Display model information
        st.write("### Model Information")
        st.write("The fraud detection model identifies potentially fraudulent transactions based on transaction patterns, customer behavior, and account characteristics.")
        
        # Display model evaluation metrics
        st.write("### Model Evaluation")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{st.session_state.fraud_evaluation['accuracy']:.2%}")
        
        with col2:
            st.metric("AUC Score", f"{st.session_state.fraud_evaluation['auc_score']:.2%}")
        
        with col3:
            precision = st.session_state.fraud_evaluation['classification_report']['1']['precision']
            recall = st.session_state.fraud_evaluation['classification_report']['1']['recall']
            st.metric("Fraud Precision", f"{precision:.2%}")
        
        # Display feature importance
        st.write("### Feature Importance")
        st.plotly_chart(visualization.plot_feature_importance(st.session_state.fraud_evaluation), use_container_width=True)
        
        # Display PR curve
        st.write("### Precision-Recall Curve")
        pr_curve_data = st.session_state.fraud_evaluation['pr_curve_data']
        fig = px.line(
            pr_curve_data,
            x='recall',
            y='precision',
            title='Precision-Recall Curve',
            labels={'precision': 'Precision', 'recall': 'Recall'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detect fraud for a customer
        st.write("### Detect Fraud for a Customer")
        client_num = st.number_input("Enter Client Number", min_value=1, step=1)
        
        if st.button("Check Fraud Risk"):
            customer = data_processing.get_customer_profile(st.session_state.data, client_num)
            
            # Get fraud prediction (the model handles None values)
            prediction = st.session_state.fraud_model.predict(customer)
            
            # Check if there was an error in prediction
            if 'error' in prediction:
                st.error(prediction['error'])
            elif customer is None:
                st.error(f"Customer with client number {client_num} not found.")
            else:
                # Display customer information
                st.write("#### Customer Information")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Customer Age", customer['Customer_Age'])
                    st.metric("Income", f"${customer['Income']:,}")
                
                with col2:
                    st.metric("Credit Limit", f"${customer['Credit_Limit']:,.2f}")
                    st.metric("Revolving Balance", f"${customer['Total_Revolving_Bal']:,}")
                
                with col3:
                    st.metric("Transaction Amount", f"${customer['Total_Trans_Amt']:,}")
                    st.metric("Transaction Count", customer['Total_Trans_Ct'])
                
                # Display fraud risk
                st.write("#### Fraud Risk Assessment")
                
                risk_color = "red" if prediction['risk_level'] == 'High' else "orange" if prediction['risk_level'] == 'Medium' else "green"
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Fraud Risk Level", prediction['risk_level'])
                
                with col2:
                    st.metric("Fraud Probability", f"{prediction['fraud_probability']:.2%}")
                
                # Display gauge chart for fraud probability
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction['fraud_probability'] * 100,
                    title={'text': "Fraud Risk"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': risk_color},
                        'steps': [
                            {'range': [0, 30], 'color': "green"},
                            {'range': [30, 70], 'color': "orange"},
                            {'range': [70, 100], 'color': "red"}
                        ]
                    }
                ))
                
                fig.update_layout(height=300)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display recent transactions
                st.write("#### Recent Transactions")
                recent_transactions = data_processing.get_recent_transactions(st.session_state.data, client_num)
                
                if recent_transactions is not None:
                    st.dataframe(recent_transactions[['Week_Start_Date', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Exp_Type']])

# Customer Analysis page
elif page == "Customer Analysis":
    st.write("## Customer Analysis")
    
    if st.session_state.data is None:
        st.warning("Please load the data first.")
        if st.button("Load Data"):
            load_data()
    else:
        # Customer search
        st.write("### Customer Search")
        search_method = st.radio("Search by", ["Client Number", "Filters"])
        
        if search_method == "Client Number":
            client_num = st.number_input("Enter Client Number", min_value=1, step=1)
            
            if st.button("Search"):
                customer = data_processing.get_customer_profile(st.session_state.data, client_num)
                
                if customer is not None:
                    # Get predictions
                    credit_pred = st.session_state.credit_model.predict(customer)
                    fraud_pred = st.session_state.fraud_model.predict(customer)
                    
                    # Display customer profile
                    st.write("#### Customer Profile")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Client Number", customer['Client_Num'])
                        st.metric("Age", customer['Customer_Age'])
                        st.metric("Gender", customer['Gender'])
                        st.metric("Dependents", customer['Dependent_Count'])
                    
                    with col2:
                        st.metric("Education", customer['Education_Level'])
                        st.metric("Marital Status", customer['Marital_Status'])
                        st.metric("Income", f"${customer['Income']:,}")
                        st.metric("Job", customer['Customer_Job'])
                    
                    with col3:
                        st.metric("State", customer['state_cd'])
                        st.metric("Car Owner", customer['Car_Owner'])
                        st.metric("House Owner", customer['House_Owner'])
                        st.metric("Satisfaction Score", f"{customer['Cust_Satisfaction_Score']}/5")
                    
                    # Display credit and fraud profile
                    st.write("#### Credit and Fraud Profile")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Credit Score Band", credit_pred['credit_score_band'])
                        st.metric("Credit Score Confidence", f"{credit_pred['probability']:.2%}")
                    
                    with col2:
                        st.metric("Fraud Risk Level", fraud_pred['risk_level'])
                        st.metric("Fraud Probability", f"{fraud_pred['fraud_probability']:.2%}")
                    
                    # Display visualized profile
                    st.plotly_chart(visualization.plot_customer_credit_profile(customer, credit_pred, fraud_pred), use_container_width=True)
                    
                    # Display financial information
                    st.write("#### Financial Information")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Credit Limit", f"${customer['Credit_Limit']:,.2f}")
                        st.metric("Revolving Balance", f"${customer['Total_Revolving_Bal']:,}")
                    
                    with col2:
                        st.metric("Transaction Amount", f"${customer['Total_Trans_Amt']:,}")
                        st.metric("Transaction Count", customer['Total_Trans_Ct'])
                    
                    with col3:
                        st.metric("Utilization Ratio", f"{customer['Avg_Utilization_Ratio']:.2%}")
                        st.metric("Interest Earned", f"${customer['Interest_Earned']:,.2f}")
                    
                    # Display recent transactions
                    st.write("#### Recent Transactions")
                    recent_transactions = data_processing.get_recent_transactions(st.session_state.data, client_num)
                    
                    if recent_transactions is not None:
                        st.dataframe(recent_transactions[['Week_Start_Date', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Exp_Type']])
                else:
                    st.error("Customer not found. Please enter a valid client number.")
        
        else:  # Filters
            col1, col2 = st.columns(2)
            
            with col1:
                age_range = st.slider("Age Range", min_value=18, max_value=100, value=(30, 60))
                income_range = st.slider("Income Range ($)", min_value=0, max_value=250000, value=(20000, 100000), step=5000)
            
            with col2:
                gender = st.selectbox("Gender", ["All", "M", "F"])
                credit_score = st.selectbox("Credit Score Band", ["All", "Poor", "Fair", "Good", "Excellent"])
            
            if st.button("Search"):
                # Filter data
                filtered_data = st.session_state.data.copy()
                
                # Apply filters
                filtered_data = filtered_data[(filtered_data['Customer_Age'] >= age_range[0]) & (filtered_data['Customer_Age'] <= age_range[1])]
                filtered_data = filtered_data[(filtered_data['Income'] >= income_range[0]) & (filtered_data['Income'] <= income_range[1])]
                
                if gender != "All":
                    filtered_data = filtered_data[filtered_data['Gender'] == gender]
                
                if credit_score != "All":
                    filtered_data = filtered_data[filtered_data['credit_score_band'] == credit_score]
                
                # Display results
                st.write(f"#### Found {len(filtered_data)} records")
                
                if len(filtered_data) > 0:
                    st.dataframe(filtered_data[['Client_Num', 'Customer_Age', 'Gender', 'Income', 'Credit_Limit', 'credit_score_band', 'fraud_flag']])
                    
                    # Display aggregate statistics
                    st.write("#### Aggregate Statistics for Filtered Customers")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Average Age", f"{filtered_data['Customer_Age'].mean():.1f}")
                        st.metric("Average Income", f"${filtered_data['Income'].mean():,.2f}")
                    
                    with col2:
                        st.metric("Average Credit Limit", f"${filtered_data['Credit_Limit'].mean():,.2f}")
                        st.metric("Average Revolving Balance", f"${filtered_data['Total_Revolving_Bal'].mean():,.2f}")
                    
                    with col3:
                        st.metric("Fraud Rate", f"{filtered_data['fraud_flag'].mean():.2%}")
                        st.metric("Average Satisfaction Score", f"{filtered_data['Cust_Satisfaction_Score'].mean():.2f}/5")
                else:
                    st.info("No customers found with the selected filters.")

# Power BI Integration page
elif page == "Power BI Integration":
    st.write("## Power BI Integration")
    
    if st.session_state.data is None:
        st.warning("Please load the data first.")
        if st.button("Load Data"):
            load_data()
    else:
        # Start API
        if not st.session_state.api_running:
            if st.button("Start API Server"):
                start_api_thread()
        else:
            st.success("API server is running!")
        
        # API Information
        st.write("### API Endpoints for Power BI")
        st.write("""
        The following API endpoints can be used to integrate with Power BI for visualization:
        """)
        
        st.code("""
        GET http://localhost:8000/api/load_data - Load and preprocess data
        GET http://localhost:8000/api/customer/<client_num> - Get customer information
        GET http://localhost:8000/api/credit_score/<client_num> - Get credit score for a customer
        GET http://localhost:8000/api/fraud_detection/<client_num> - Get fraud detection results for a customer
        GET http://localhost:8000/api/model_metrics - Get model metrics for Power BI
        GET http://localhost:8000/api/export_data - Export processed data for Power BI
        """)
        
        # Export data for Power BI
        st.write("### Export Data for Power BI")
        if st.button("Export Data"):
            # Create directory for exports
            if not os.path.exists('exports'):
                os.makedirs('exports')
            
            # Export data
            export_cols = [
                'Client_Num', 'Customer_Age', 'Gender', 'Income', 'Credit_Limit',
                'Total_Revolving_Bal', 'Total_Trans_Amt', 'Total_Trans_Ct',
                'Avg_Utilization_Ratio', 'Delinquent_Acc_x', 'credit_score',
                'credit_score_band', 'fraud_flag'
            ]
            
            st.session_state.data[export_cols].to_csv('exports/credit_card_data_for_powerbi.csv', index=False)
            
            st.success("Data exported successfully to 'exports/credit_card_data_for_powerbi.csv'!")
        
        # Power BI Dashboard Examples
        st.write("### Power BI Dashboard Examples")
        
        st.write("""
        #### Sample KPIs for Power BI Dashboards:
        
        1. **Credit Scoring KPIs:**
           - Credit score distribution
           - Average credit score by demographic
           - Credit limit utilization by credit score band
           - Income to credit ratio trends
        
        2. **Fraud Detection KPIs:**
           - Fraud rate over time
           - Fraud by transaction type
           - High-risk customer segments
           - Fraud detection accuracy metrics
        
        3. **Customer Insights:**
           - Customer segmentation by credit behavior
           - Transaction patterns by customer segment
           - Customer satisfaction correlation with credit score
           - Regional credit risk analysis
        """)
        
        # Example visualizations
        st.write("### Example Visualizations for Power BI")
        
        tab1, tab2, tab3 = st.tabs(["Credit Score Distribution", "Fraud Analysis", "Customer Segments"])
        
        with tab1:
            st.plotly_chart(visualization.plot_credit_score_distribution(st.session_state.data), use_container_width=True)
        
        with tab2:
            st.plotly_chart(visualization.plot_fraud_distribution(st.session_state.data), use_container_width=True)
        
        with tab3:
            st.plotly_chart(visualization.plot_credit_score_by_age(st.session_state.data), use_container_width=True)
        
        # Power BI connection instructions
        st.write("### Power BI Connection Instructions")
        
        st.write("""
        To connect Power BI to this API:
        
        1. In Power BI Desktop, click on **Get Data** > **Web**
        2. Enter the API URL (e.g., http://localhost:8000/api/export_data)
        3. If the API returns JSON, Power BI will automatically parse it
        4. Transform the data as needed using Power Query Editor
        5. Create visualizations and dashboards using the imported data
        
        You can also use the exported CSV file for direct import into Power BI.
        """)

if __name__ == "__main__":
    # Display footer
    st.markdown("---")
    st.markdown("Credit Scoring & Fraud Detection System | © 2023")
