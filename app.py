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
import base64
from datetime import datetime

# Import local modules
import data_processing
import credit_scoring
import fraud_detection
import visualization
import recommendation_system
import api

# Set page configuration first (must be at the top)
st.set_page_config(
    page_title="Credit Scoring & Fraud Detection System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load simple CSS 
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except Exception as e:
    st.warning(f"Error loading CSS: {str(e)}")

# Simple header function 
def add_simple_header():
    st.title("Credit Scoring & Fraud Detection System")
    st.markdown("**ADVANCED FINANCIAL ANALYTICS**")
    
    # Display current date and time
    now = datetime.now()
    st.text(f"Current date: {now.strftime('%B %d, %Y')} | Time: {now.strftime('%H:%M:%S')}")

# Add simple header
add_simple_header()

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
if 'recommender_model' not in st.session_state:
    st.session_state.recommender_model = recommendation_system.CreditCardRecommender()
if 'credit_evaluation' not in st.session_state:
    st.session_state.credit_evaluation = None
if 'fraud_evaluation' not in st.session_state:
    st.session_state.fraud_evaluation = None
if 'api_running' not in st.session_state:
    st.session_state.api_running = False
if 'api_thread' not in st.session_state:
    st.session_state.api_thread = None

# Function for safe plotting that handles errors gracefully
def safe_plot(plot_function, data, error_message="Error generating visualization"):
    """Safely execute a plotting function with error handling"""
    try:
        if data is None:
            st.warning("No data available for visualization")
            return None
        return plot_function(data)
    except Exception as e:
        st.warning(f"{error_message}: {str(e)}")
        return None

# Function to load and process data with robust error handling
def load_data():
    with st.spinner("Loading and processing data..."):
        try:
            # Load data
            data = data_processing.load_data()
            
            if data is not None and isinstance(data, pd.DataFrame) and not data.empty:
                # Preprocess data
                processed_data, credit_data, fraud_data = data_processing.preprocess_data(data)
                
                # Validate the processed data
                if (processed_data is None or 
                    not isinstance(processed_data, pd.DataFrame) or 
                    processed_data.empty):
                    st.error("Data preprocessing failed: Empty dataset returned")
                    return
                
                # Store in session state
                st.session_state.data = processed_data
                st.session_state.credit_data = credit_data
                st.session_state.fraud_data = fraud_data
                
                st.success("Data loaded and processed successfully!")
                
                # Train or load models with error handling
                with st.spinner("Loading and training models..."):
                    try:
                        # Credit model
                        credit_model_loaded = st.session_state.credit_model.load_model()
                        if not credit_model_loaded:
                            if (credit_data is not None and 
                                isinstance(credit_data, dict) and 
                                'X_train' in credit_data and 
                                'y_train' in credit_data and 
                                'features' in credit_data and 
                                'scaler' in credit_data):
                                
                                # Make sure to set model features and scaler
                                st.session_state.credit_model.features = credit_data['features']
                                st.session_state.credit_model.scaler = credit_data['scaler']
                                
                                # Allow selection of model type (Random Forest or XGBoost)
                                credit_model_type = st.selectbox(
                                    "Select Credit Scoring Model Type",
                                    options=["XGBoost (advanced)", "Random Forest (traditional)"],
                                    index=0
                                )
                                
                                # Train with selected model type
                                model_type = 'xgb' if 'XGBoost' in credit_model_type else 'rf'
                                st.session_state.credit_model.train(credit_data, model_type=model_type)
                            else:
                                st.warning("Cannot train credit model: Credit data is empty or invalid")
                        # Even if model loaded, still set features/scaler from current data if available
                        elif (credit_data is not None and isinstance(credit_data, dict) and 
                              'features' in credit_data and 'scaler' in credit_data and
                              st.session_state.credit_model.features is None):
                            st.session_state.credit_model.features = credit_data['features']
                            st.session_state.credit_model.scaler = credit_data['scaler']
                        
                        # Fraud model
                        fraud_model_loaded = st.session_state.fraud_model.load_model()
                        if not fraud_model_loaded:
                            if (fraud_data is not None and 
                                isinstance(fraud_data, dict) and
                                'X_train' in fraud_data and 
                                'y_train' in fraud_data and 
                                'features' in fraud_data and 
                                'scaler' in fraud_data):
                                
                                # Make sure to set model features and scaler
                                st.session_state.fraud_model.features = fraud_data['features']
                                st.session_state.fraud_model.scaler = fraud_data['scaler']
                                
                                # Allow selection of model type (Random Forest or XGBoost)
                                fraud_model_type = st.selectbox(
                                    "Select Fraud Detection Model Type",
                                    options=["XGBoost (advanced)", "Random Forest (traditional)"],
                                    index=0
                                )
                                
                                # Train with selected model type
                                model_type = 'xgb' if 'XGBoost' in fraud_model_type else 'rf'
                                st.session_state.fraud_model.train(fraud_data, model_type=model_type)
                            else:
                                st.warning("Cannot train fraud model: Fraud data is empty or invalid")
                        # Even if model loaded, still set features/scaler from current data if available
                        elif (fraud_data is not None and isinstance(fraud_data, dict) and 
                              'features' in fraud_data and 'scaler' in fraud_data and
                              st.session_state.fraud_model.features is None):
                            st.session_state.fraud_model.features = fraud_data['features']
                            st.session_state.fraud_model.scaler = fraud_data['scaler']
                        
                        # Evaluate models - with robust error handling
                        if st.session_state.credit_model.is_trained and st.session_state.credit_model.model is not None:
                            st.session_state.credit_evaluation = st.session_state.credit_model.evaluate(credit_data)
                        else:
                            st.warning("Credit model evaluation skipped: Model not properly trained")
                            
                        if st.session_state.fraud_model.is_trained and st.session_state.fraud_model.model is not None:
                            st.session_state.fraud_evaluation = st.session_state.fraud_model.evaluate(fraud_data)
                        else:
                            st.warning("Fraud model evaluation skipped: Model not properly trained")
                        
                        # Check model readiness
                        credit_ready = (st.session_state.credit_model.is_trained and 
                                       st.session_state.credit_model.model is not None and
                                       st.session_state.credit_model.features is not None and
                                       st.session_state.credit_model.scaler is not None)
                                       
                        fraud_ready = (st.session_state.fraud_model.is_trained and 
                                      st.session_state.fraud_model.model is not None and
                                      st.session_state.fraud_model.features is not None and
                                      st.session_state.fraud_model.scaler is not None)
                        
                        if credit_ready and fraud_ready:
                            st.success("Models loaded and evaluated successfully!")
                        else:
                            st.warning("Some models may not be fully initialized. Check system status.")
                            
                    except Exception as e:
                        st.error(f"Error during model training: {str(e)}")
            else:
                st.error("Failed to load data. Please check the file paths.")
        except Exception as e:
            st.error(f"Error during data loading: {str(e)}")

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

# Enhanced sidebar
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <div style="background: linear-gradient(45deg, #7792E3, #5F67EA); border-radius: 50%; width: 60px; height: 60px; display: flex; align-items: center; justify-content: center; margin: 0 auto 10px auto;">
        <span style="color: white; font-size: 2rem;">💳</span>
    </div>
    <div style="font-weight: 600; font-size: 1.2rem; margin-bottom: 5px;">FinAnalytica</div>
    <div style="font-size: 0.8rem; color: #7792E3;">v2.0 - AI Enhanced</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Home", "Data Exploration", "Credit Scoring", "Fraud Detection", "Customer Analysis", "Recommendation System"])

# Add user status to sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="display: flex; align-items: center; margin-top: 20px;">
    <div style="background: #7792E3; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; margin-right: 10px;">
        <span style="color: white; font-size: 0.9rem;">👤</span>
    </div>
    <div>
        <div style="font-size: 0.9rem; font-weight: 600;">Admin User</div>
        <div style="font-size: 0.7rem; color: #7792E3;">Full Access</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Header already displayed via add_simple_header() above
# Home page
if page == "Home":
    # Futuristic welcome banner with animation
    st.markdown("""
    <div style="background: linear-gradient(90deg, rgba(26, 31, 42, 0.8), rgba(26, 31, 42, 0.4), rgba(26, 31, 42, 0.8)); 
                padding: 20px; border-radius: 10px; margin-bottom: 30px; position: relative; overflow: hidden; 
                border: 1px solid rgba(119, 146, 227, 0.3); animation: glow 3s ease-in-out infinite;">
        <style>
            @keyframes glow {
                0% { box-shadow: 0 0 10px rgba(119, 146, 227, 0.1); }
                50% { box-shadow: 0 0 20px rgba(119, 146, 227, 0.3); }
                100% { box-shadow: 0 0 10px rgba(119, 146, 227, 0.1); }
            }
            @keyframes slide {
                0% { transform: translateX(-100%); }
                100% { transform: translateX(100%); }
            }
        </style>
        <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: linear-gradient(90deg, transparent, rgba(119, 146, 227, 0.1), transparent); 
                    animation: slide 3s ease-in-out infinite; z-index: 1;"></div>
        <div style="position: relative; z-index: 2;">
            <h2 style="color: white; margin-top: 0;">Welcome to the Future of Financial Analytics</h2>
            <p style="color: #B8C2E0; font-size: 1.1rem;">
                This AI-powered system helps financial institutions make data-driven decisions about credit applications and 
                detect potential fraudulent activities. Navigate using the sidebar to explore the advanced features.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # System overview cards in a grid
    st.markdown("""
    <h3 style="margin-bottom: 20px;">System Overview</h3>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: rgba(26, 31, 42, 0.6); padding: 20px; border-radius: 10px; border-left: 3px solid #7792E3; margin-bottom: 20px; height: 100%;">
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                <div style="background: linear-gradient(45deg, #7792E3, #5F67EA); border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; margin-right: 15px;">
                    <span style="color: white; font-size: 1.2rem;">📊</span>
                </div>
                <h4 style="margin: 0; color: white;">Credit Scoring Analytics</h4>
            </div>
            <ul style="color: #B8C2E0; margin-left: 15px; padding-left: 15px;">
                <li>Evaluate creditworthiness with ML algorithms</li>
                <li>Predict credit score bands with high precision</li>
                <li>Analyze key factors affecting credit scores</li>
                <li>Generate comprehensive risk profiles</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: rgba(26, 31, 42, 0.6); padding: 20px; border-radius: 10px; border-left: 3px solid #5F67EA; margin-bottom: 20px; height: 100%;">
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                <div style="background: linear-gradient(45deg, #5F67EA, #4752E9); border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; margin-right: 15px;">
                    <span style="color: white; font-size: 1.2rem;">🛡️</span>
                </div>
                <h4 style="margin: 0; color: white;">Fraud Detection System</h4>
            </div>
            <ul style="color: #B8C2E0; margin-left: 15px; padding-left: 15px;">
                <li>Identify suspicious transactions in real-time</li>
                <li>Calculate fraud risk probabilities with AI</li>
                <li>Monitor fraud patterns and anomalies</li>
                <li>Enhance security through pattern recognition</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Load data button with enhanced styling
    st.markdown("""
    <div style="background: rgba(26, 31, 42, 0.4); padding: 20px; border-radius: 10px; margin: 30px 0; 
               border: 1px dashed rgba(119, 146, 227, 0.3); text-align: center;">
        <h4 style="margin-top: 0; color: #7792E3;">System Initialization</h4>
        <p style="color: #B8C2E0;">Load data and train the machine learning models to begin your analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Center the button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("⚡ Load Data & Train Models", use_container_width=True):
            load_data()
    
    # Display system status with futuristic metrics
    st.markdown("""
    <h3 style="margin: 30px 0 20px 0;">System Status</h3>
    """, unsafe_allow_html=True)
    
    data_status = "Loaded" if st.session_state.data is not None else "Not Loaded"
    credit_model_status = "Trained" if st.session_state.credit_model.is_trained else "Not Trained" 
    fraud_model_status = "Trained" if st.session_state.fraud_model.is_trained else "Not Trained"
    api_status = "Running" if st.session_state.api_running else "Not Running"
    
    # Status indicators with icons and colors
    col1, col2, col3, col4 = st.columns(4)
    
    data_color = "#4CAF50" if st.session_state.data is not None else "#F44336"
    credit_color = "#4CAF50" if st.session_state.credit_model.is_trained else "#F44336"
    fraud_color = "#4CAF50" if st.session_state.fraud_model.is_trained else "#F44336"
    api_color = "#4CAF50" if st.session_state.api_running else "#F44336"
    
    col1.markdown(f"""
    <div style="background: rgba(26, 31, 42, 0.6); padding: 15px; border-radius: 10px; text-align: center;">
        <div style="color: {data_color}; font-size: 1.8rem; margin-bottom: 5px;">{'✓' if st.session_state.data is not None else '×'}</div>
        <div style="font-size: 0.9rem; color: #B8C2E0; text-transform: uppercase; letter-spacing: 0.05em;">Data</div>
        <div style="font-weight: 600; color: white;">{data_status}</div>
    </div>
    """, unsafe_allow_html=True)
    
    col2.markdown(f"""
    <div style="background: rgba(26, 31, 42, 0.6); padding: 15px; border-radius: 10px; text-align: center;">
        <div style="color: {credit_color}; font-size: 1.8rem; margin-bottom: 5px;">{'✓' if st.session_state.credit_model.is_trained else '×'}</div>
        <div style="font-size: 0.9rem; color: #B8C2E0; text-transform: uppercase; letter-spacing: 0.05em;">Credit Model</div>
        <div style="font-weight: 600; color: white;">{credit_model_status}</div>
    </div>
    """, unsafe_allow_html=True)
    
    col3.markdown(f"""
    <div style="background: rgba(26, 31, 42, 0.6); padding: 15px; border-radius: 10px; text-align: center;">
        <div style="color: {fraud_color}; font-size: 1.8rem; margin-bottom: 5px;">{'✓' if st.session_state.fraud_model.is_trained else '×'}</div>
        <div style="font-size: 0.9rem; color: #B8C2E0; text-transform: uppercase; letter-spacing: 0.05em;">Fraud Model</div>
        <div style="font-weight: 600; color: white;">{fraud_model_status}</div>
    </div>
    """, unsafe_allow_html=True)
    
    col4.markdown(f"""
    <div style="background: rgba(26, 31, 42, 0.6); padding: 15px; border-radius: 10px; text-align: center;">
        <div style="color: {api_color}; font-size: 1.8rem; margin-bottom: 5px;">{'✓' if st.session_state.api_running else '×'}</div>
        <div style="font-size: 0.9rem; color: #B8C2E0; text-transform: uppercase; letter-spacing: 0.05em;">API Status</div>
        <div style="font-weight: 600; color: white;">{api_status}</div>
    </div>
    """, unsafe_allow_html=True)
    
        
    # Add a footer with system info
    st.markdown("""
    <div style="margin-top: 60px; padding-top: 20px; border-top: 1px solid rgba(119, 146, 227, 0.2); text-align: center; color: #B8C2E0; font-size: 0.8rem;">
        <div>Credit Scoring & Fraud Detection System v2.0</div>
        <div style="margin-top: 5px; color: #7792E3;">Powered by Advanced Machine Learning Algorithms</div>
    </div>
    """, unsafe_allow_html=True)

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
        
        # Display visualizations with error handling
        st.write("### Visualizations")
        tab1, tab2, tab3, tab4 = st.tabs(["Credit Score Distribution", "Fraud Distribution", "Transaction Trends", "Demographics"])
        
        with tab1:
            plot1 = safe_plot(visualization.plot_credit_score_distribution, st.session_state.data, "Error generating credit score distribution")
            if plot1 is not None:
                st.plotly_chart(plot1, use_container_width=True)
                
            plot2 = safe_plot(visualization.plot_income_credit_limit_relation, st.session_state.data, "Error generating income-credit limit relation")
            if plot2 is not None:
                st.plotly_chart(plot2, use_container_width=True)
        
        with tab2:
            plot3 = safe_plot(visualization.plot_fraud_distribution, st.session_state.data, "Error generating fraud distribution")
            if plot3 is not None:
                st.plotly_chart(plot3, use_container_width=True)
        
        with tab3:
            plot4 = safe_plot(visualization.plot_transaction_trends, st.session_state.data, "Error generating transaction trends")
            if plot4 is not None:
                st.plotly_chart(plot4, use_container_width=True)
        
        with tab4:
            plot5 = safe_plot(visualization.plot_credit_score_by_age, st.session_state.data, "Error generating credit score by age")
            if plot5 is not None:
                st.plotly_chart(plot5, use_container_width=True)

# Credit Scoring page
elif page == "Credit Scoring":
    st.write("## Credit Scoring Model")
    
    # Check if data is loaded and model is trained with all required components
    model_ready = (st.session_state.data is not None and 
                  st.session_state.credit_model.is_trained and
                  st.session_state.credit_model.model is not None and
                  st.session_state.credit_model.features is not None and
                  st.session_state.credit_model.scaler is not None)
    
    if not model_ready:
        st.warning("Please load the data and train the models first.")
    else:
        # Display model info
        model_type = st.session_state.credit_model.model_type if hasattr(st.session_state.credit_model, 'model_type') else "Random Forest"
        model_name = "XGBoost (Advanced)" if model_type == 'xgb' else "Random Forest (Traditional)"
        
        st.markdown(f"""
        <div style="background: rgba(26, 31, 42, 0.6); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center;">
                <div style="background: linear-gradient(45deg, #7792E3, #5F67EA); border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; margin-right: 15px;">
                    <span style="color: white; font-size: 1.2rem;">🤖</span>
                </div>
                <div>
                    <div style="font-size: 1rem; color: white;">Active Model: <span style="color: #7792E3; font-weight: 600;">{model_name}</span></div>
                    <div style="font-size: 0.8rem; color: #B8C2E0;">Trained and optimized for credit risk assessment</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
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
    
    # Check if data is loaded and model is trained with all required components
    model_ready = (st.session_state.data is not None and 
                  st.session_state.fraud_model.is_trained and
                  st.session_state.fraud_model.model is not None and
                  st.session_state.fraud_model.features is not None and
                  st.session_state.fraud_model.scaler is not None)
    
    if not model_ready:
        st.warning("Please load the data and train the models first.")
        if st.button("Load Data and Train Models"):
            load_data()
    else:
        # Display model type info
        model_type = st.session_state.fraud_model.model_type if hasattr(st.session_state.fraud_model, 'model_type') else "Random Forest"
        model_name = "XGBoost (Advanced)" if model_type == 'xgb' else "Random Forest (Traditional)"
        
        st.markdown(f"""
        <div style="background: rgba(26, 31, 42, 0.6); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center;">
                <div style="background: linear-gradient(45deg, #5F67EA, #4752E9); border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; margin-right: 15px;">
                    <span style="color: white; font-size: 1.2rem;">🛡️</span>
                </div>
                <div>
                    <div style="font-size: 1rem; color: white;">Active Model: <span style="color: #5F67EA; font-weight: 600;">{model_name}</span></div>
                    <div style="font-size: 0.8rem; color: #B8C2E0;">Trained and optimized for fraud risk detection</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display model information
        st.write("### Model Information")
        st.write("The fraud detection model identifies potentially fraudulent transactions based on transaction patterns, customer behavior, and account characteristics.")
        
        # Display model evaluation metrics with robust error handling
        st.write("### Model Evaluation")
        
        # Check if fraud_evaluation is not None and has required keys
        if (st.session_state.fraud_evaluation is None or 
            not isinstance(st.session_state.fraud_evaluation, dict)):
            st.warning("Fraud model evaluation data is not available. Please retrain the model.")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'accuracy' in st.session_state.fraud_evaluation:
                    st.metric("Accuracy", f"{st.session_state.fraud_evaluation['accuracy']:.2%}")
                else:
                    st.metric("Accuracy", "N/A")
            
            with col2:
                if 'auc_score' in st.session_state.fraud_evaluation:
                    st.metric("AUC Score", f"{st.session_state.fraud_evaluation['auc_score']:.2%}")
                else:
                    st.metric("AUC Score", "N/A")
            
            with col3:
                try:
                    if ('classification_report' in st.session_state.fraud_evaluation and 
                        isinstance(st.session_state.fraud_evaluation['classification_report'], dict) and
                        '1' in st.session_state.fraud_evaluation['classification_report']):
                        
                        precision = st.session_state.fraud_evaluation['classification_report']['1']['precision']
                        st.metric("Fraud Precision", f"{precision:.2%}")
                    else:
                        st.metric("Fraud Precision", "N/A")
                except (KeyError, TypeError) as e:
                    st.metric("Fraud Precision", "N/A")
                    st.warning(f"Error calculating precision: {str(e)}")
            
            # Display feature importance with safe plotting
            st.write("### Feature Importance")
            feature_plot = safe_plot(
                visualization.plot_feature_importance, 
                st.session_state.fraud_evaluation, 
                "Error generating feature importance plot"
            )
            if feature_plot is not None:
                st.plotly_chart(feature_plot, use_container_width=True)
            
            # Display PR curve with safe error handling
            st.write("### Precision-Recall Curve")
            
            try:
                if ('pr_curve_data' in st.session_state.fraud_evaluation and 
                    st.session_state.fraud_evaluation['pr_curve_data'] is not None):
                    
                    pr_curve_data = st.session_state.fraud_evaluation['pr_curve_data']
                    
                    # Further validate the data
                    if isinstance(pr_curve_data, pd.DataFrame) and not pr_curve_data.empty:
                        if all(col in pr_curve_data.columns for col in ['recall', 'precision']):
                            fig = px.line(
                                pr_curve_data,
                                x='recall',
                                y='precision',
                                title='Precision-Recall Curve',
                                labels={'precision': 'Precision', 'recall': 'Recall'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("PR curve data is missing required columns.")
                    else:
                        st.warning("PR curve data is not in the expected format.")
                else:
                    st.warning("Precision-Recall curve data is not available.")
            except Exception as e:
                st.error(f"Error displaying PR curve: {str(e)}")
        
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
                    # Check which columns are available
                    available_columns = ['Week_Start_Date', 'Total_Trans_Amt', 'Total_Trans_Ct']
                    # Add Exp_Type if it exists
                    if 'Exp_Type' in recent_transactions.columns:
                        available_columns.append('Exp_Type')
                    st.dataframe(recent_transactions[available_columns])

# Customer Analysis page
elif page == "Customer Analysis":
    st.write("## Customer Analysis")
    
    # Check if data is loaded and models are trained with all required components
    models_ready = (st.session_state.data is not None and 
                   st.session_state.credit_model.is_trained and st.session_state.credit_model.model is not None and
                   st.session_state.fraud_model.is_trained and st.session_state.fraud_model.model is not None)
    
    if not models_ready:
        st.warning("Please load the data and train the models first.")
        if st.button("Load Data and Train Models"):
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
                        # Check which columns are available
                        available_columns = ['Week_Start_Date', 'Total_Trans_Amt', 'Total_Trans_Ct']
                        # Add Exp_Type if it exists
                        if 'Exp_Type' in recent_transactions.columns:
                            available_columns.append('Exp_Type')
                        st.dataframe(recent_transactions[available_columns])
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
                # Check if data is available and process if it exists
                data_available = st.session_state.data is not None
                filter_successful = False
                filtered_data = None  # Initialize outside the block to make it available in the entire scope
                
                if not data_available:
                    st.error("Data not available. Please load the data first.")
                else:
                    # Filter data
                    try:
                        filtered_data = st.session_state.data.copy()
                        
                        # Apply filters with error handling
                        # Age filter
                        filtered_data = filtered_data[(filtered_data['Customer_Age'] >= age_range[0]) & 
                                                    (filtered_data['Customer_Age'] <= age_range[1])]
                        
                        # Income filter
                        filtered_data = filtered_data[(filtered_data['Income'] >= income_range[0]) & 
                                                    (filtered_data['Income'] <= income_range[1])]
                        
                        # Gender filter
                        if gender != "All":
                            filtered_data = filtered_data[filtered_data['Gender'] == gender]
                        
                        # Credit score filter with null handling
                        if credit_score != "All":
                            # Handle potential missing values by filtering out nulls first
                            filtered_data = filtered_data[filtered_data['credit_score_band'].notna()]
                            filtered_data = filtered_data[filtered_data['credit_score_band'] == credit_score]
                            
                        filter_successful = True
                    except Exception as e:
                        st.error(f"Error filtering data: {str(e)}")
                
                # Display results only if filtering was successful and filtered_data exists
                if filter_successful and filtered_data is not None:
                    try:
                        st.write(f"#### Found {len(filtered_data)} records")
                        
                        if len(filtered_data) > 0:
                            # Select columns to display with error handling
                            display_cols = ['Client_Num', 'Customer_Age', 'Gender', 'Income', 'Credit_Limit']
                            
                            # Add credit_score_band and fraud_flag if they exist
                            if 'credit_score_band' in filtered_data.columns:
                                display_cols.append('credit_score_band')
                            if 'fraud_flag' in filtered_data.columns:
                                display_cols.append('fraud_flag')
                                
                            st.dataframe(filtered_data[display_cols])
                            
                            # Display aggregate statistics
                            st.write("#### Aggregate Statistics for Filtered Customers")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if 'Customer_Age' in filtered_data.columns:
                                    st.metric("Average Age", f"{filtered_data['Customer_Age'].mean():.1f}")
                                if 'Income' in filtered_data.columns:
                                    st.metric("Average Income", f"${filtered_data['Income'].mean():,.2f}")
                            
                            with col2:
                                if 'Credit_Limit' in filtered_data.columns:
                                    st.metric("Average Credit Limit", f"${filtered_data['Credit_Limit'].mean():,.2f}")
                                if 'Total_Revolving_Bal' in filtered_data.columns:
                                    st.metric("Average Revolving Balance", f"${filtered_data['Total_Revolving_Bal'].mean():,.2f}")
                            
                            with col3:
                                if 'fraud_flag' in filtered_data.columns:
                                    st.metric("Fraud Rate", f"{filtered_data['fraud_flag'].mean():.2%}")
                                if 'Cust_Satisfaction_Score' in filtered_data.columns:
                                    st.metric("Average Satisfaction Score", f"{filtered_data['Cust_Satisfaction_Score'].mean():.2f}/5")
                        else:
                            st.info("No customers found with the selected filters.")
                    except Exception as e:
                        st.error(f"Error displaying results: {str(e)}")

# Recommendation System page
elif page == "Recommendation System":
    st.write("## Credit Card Recommendation System")
    
    # Check if data is loaded
    if st.session_state.data is None:
        st.warning("Please load the data first.")
        if st.button("Load Data"):
            load_data()
    else:
        # Create or load the recommender model if not already loaded
        if not hasattr(st.session_state.recommender_model, 'credit_card_data') or st.session_state.recommender_model.credit_card_data is None:
            with st.spinner("Initializing recommendation system..."):
                if not st.session_state.recommender_model.load_model():
                    # Train the model with processed data
                    st.session_state.recommender_model.train({'processed_data': st.session_state.data})
                    st.success("Recommendation system initialized successfully!")
        
        # Introduction section with modern UI
        st.markdown("""
        <div style="background: linear-gradient(90deg, rgba(26, 31, 42, 0.8), rgba(26, 31, 42, 0.4), rgba(26, 31, 42, 0.8)); 
                    padding: 20px; border-radius: 10px; margin-bottom: 30px; position: relative; overflow: hidden; 
                    border: 1px solid rgba(119, 146, 227, 0.3);">
            <div style="position: relative; z-index: 2;">
                <h3 style="color: white; margin-top: 0;">Personalized Credit Card Recommendations</h3>
                <p style="color: #B8C2E0; font-size: 1rem;">
                    Our AI-powered recommendation system analyzes customer profiles to suggest the most suitable 
                    credit card offerings based on income, spending patterns, and credit history.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Two options: search by client number or manually enter profile
        tab1, tab2 = st.tabs(["Find by Client Number", "Custom Profile"])
        
        with tab1:
            st.write("### Get Recommendations for Existing Customer")
            client_num = st.number_input("Enter Client Number", min_value=1, step=1)
            
            if st.button("Get Card Recommendations", key="client_recommendations"):
                with st.spinner("Analyzing customer profile..."):
                    # Get customer data
                    customer = data_processing.get_customer_profile(st.session_state.data, client_num)
                    
                    if customer is not None:
                        # Get credit score if not already present
                        if 'credit_score' not in customer:
                            credit_prediction = st.session_state.credit_model.predict(customer)
                            customer['credit_score'] = credit_prediction.get('credit_score_numeric', 650)
                        
                        # Get recommendations
                        recommendations = st.session_state.recommender_model.recommend_cards(customer)
                        
                        # Get user segment
                        user_segment = st.session_state.recommender_model.get_user_segment(customer)
                        
                        # Display customer profile overview
                        st.markdown(f"""
                        <div style="background: rgba(26, 31, 42, 0.6); padding: 20px; border-radius: 10px; margin: 20px 0;">
                            <h4 style="color: white; margin-top: 0;">Customer Profile Overview</h4>
                            <div style="display: flex; flex-wrap: wrap; gap: 15px;">
                                <div style="flex: 1; min-width: 120px;">
                                    <div style="font-size: 0.8rem; color: #B8C2E0;">Client Number</div>
                                    <div style="font-size: 1.1rem; color: white; font-weight: 600;">{client_num}</div>
                                </div>
                                <div style="flex: 1; min-width: 120px;">
                                    <div style="font-size: 0.8rem; color: #B8C2E0;">User Segment</div>
                                    <div style="font-size: 1.1rem; color: white; font-weight: 600; text-transform: capitalize;">{user_segment}</div>
                                </div>
                                <div style="flex: 1; min-width: 120px;">
                                    <div style="font-size: 0.8rem; color: #B8C2E0;">Income</div>
                                    <div style="font-size: 1.1rem; color: white; font-weight: 600;">${customer['Income']:,.2f}</div>
                                </div>
                                <div style="flex: 1; min-width: 120px;">
                                    <div style="font-size: 0.8rem; color: #B8C2E0;">Credit Score</div>
                                    <div style="font-size: 1.1rem; color: white; font-weight: 600;">{customer['credit_score']}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display recommendations
                        st.markdown("### Recommended Credit Cards")
                        
                        for i, card in enumerate(recommendations):
                            match_score_color = "#4CAF50" if card['match_score'] >= 80 else "#FFC107" if card['match_score'] >= 60 else "#F44336"
                            
                            st.markdown(f"""
                            <div style="background: rgba(26, 31, 42, 0.6); padding: 20px; border-radius: 10px; margin-bottom: 15px; 
                                       border-left: 3px solid {match_score_color};">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                    <h4 style="color: white; margin: 0;">{card['name']}</h4>
                                    <div style="background: linear-gradient(45deg, #1A1F2A, {match_score_color}); padding: 5px 10px; border-radius: 15px;">
                                        <span style="color: white; font-weight: 600;">{card['match_score']:.0f}% Match</span>
                                    </div>
                                </div>
                                
                                <div style="display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 15px;">
                                    <div style="flex: 1; min-width: 120px;">
                                        <div style="font-size: 0.8rem; color: #B8C2E0;">Annual Fee</div>
                                        <div style="font-size: 1rem; color: white;">${card['annual_fee']}</div>
                                    </div>
                                    <div style="flex: 1; min-width: 120px;">
                                        <div style="font-size: 0.8rem; color: #B8C2E0;">Rewards Rate</div>
                                        <div style="font-size: 1rem; color: white;">{card['rewards_rate']*100:.1f}%</div>
                                    </div>
                                    <div style="flex: 1; min-width: 120px;">
                                        <div style="font-size: 0.8rem; color: #B8C2E0;">Intro APR</div>
                                        <div style="font-size: 1rem; color: white;">{card['intro_apr']}</div>
                                    </div>
                                    <div style="flex: 1; min-width: 120px;">
                                        <div style="font-size: 0.8rem; color: #B8C2E0;">Foreign Transaction Fee</div>
                                        <div style="font-size: 1rem; color: white;">{card['foreign_transaction_fee']*100:.1f}%</div>
                                    </div>
                                </div>
                                
                                <div style="margin-bottom: 15px;">
                                    <div style="font-size: 0.8rem; color: #B8C2E0; margin-bottom: 5px;">Benefits</div>
                                    <ul style="color: white; margin: 0; padding-left: 20px;">
                                        {' '.join(['<li>' + benefit + '</li>' for benefit in card['benefits']])}
                                    </ul>
                                </div>
                                
                                <div>
                                    <div style="font-size: 0.8rem; color: #B8C2E0; margin-bottom: 5px;">Ideal For</div>
                                    <div style="color: white; font-style: italic;">{card['ideal_for']}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Get similar customers
                        similar_customers = st.session_state.recommender_model.get_similar_customers(customer)
                        if similar_customers is not None and not similar_customers.empty:
                            st.markdown("### Similar Customer Profiles")
                            st.markdown("""
                            <div style="font-size: 0.9rem; color: #B8C2E0; margin-bottom: 15px;">
                                The following customers have similar financial profiles and may have similar credit needs:
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Check available columns and select appropriate ones for display
                            available_cols = similar_customers.columns.tolist()
                            display_cols = []
                            
                            # Map potential column names to standardized display columns
                            col_mapping = {
                                'Client_No': ['Client_No', 'Client_Number', 'client_num', 'client_number'],
                                'Customer_Age': ['Customer_Age', 'Age', 'age'],
                                'Income': ['Income', 'Annual_Income', 'income'],
                                'Credit_Limit': ['Credit_Limit', 'credit_limit'],
                                'Total_Trans_Amt': ['Total_Trans_Amt', 'Transaction_Amount', 'total_transaction_amount'],
                                'Total_Trans_Ct': ['Total_Trans_Ct', 'Transaction_Count', 'total_transaction_count']
                            }
                            
                            # Debug output to see available columns
                            st.write("Debug: Available columns in similar customers data:", available_cols)
                            
                            # Just display all columns without trying to extract specific ones or set an index
                            # This avoids KeyError issues regardless of what columns are actually available
                            st.dataframe(similar_customers, use_container_width=True)
                    else:
                        st.error(f"Customer with client number {client_num} not found.")
        
        with tab2:
            st.write("### Create Custom Profile for Recommendations")
            
            # Create form for custom profile input
            col1, col2 = st.columns(2)
            
            with col1:
                custom_age = st.slider("Age", 18, 80, 35)
                custom_income = st.number_input("Annual Income ($)", min_value=15000, max_value=250000, value=60000, step=5000)
                custom_credit_score = st.slider("Credit Score", 300, 850, 680)
            
            with col2:
                custom_revolving_balance = st.number_input("Revolving Balance ($)", min_value=0, max_value=50000, value=2000, step=500)
                custom_transaction_amount = st.number_input("Monthly Transaction Amount ($)", min_value=0, max_value=20000, value=3000, step=500)
                custom_transaction_count = st.slider("Monthly Transaction Count", 0, 200, 40)
            
            # Get credit limit based on income (rough estimate)
            custom_credit_limit = min(custom_income * 0.3, 30000)
            
            # Calculate utilization ratio
            if custom_credit_limit > 0:
                custom_utilization = custom_revolving_balance / custom_credit_limit
            else:
                custom_utilization = 0
            
            # Display estimated credit limit and utilization
            st.info(f"Estimated Credit Limit: ${custom_credit_limit:,.2f} | Utilization Ratio: {custom_utilization:.2%}")
            
            if st.button("Get Card Recommendations", key="custom_recommendations"):
                with st.spinner("Generating personalized recommendations..."):
                    # Create custom customer profile
                    custom_profile = {
                        'Customer_Age': custom_age,
                        'Income': custom_income,
                        'Credit_Limit': custom_credit_limit,
                        'Total_Revolving_Bal': custom_revolving_balance,
                        'Total_Trans_Amt': custom_transaction_amount,
                        'Total_Trans_Ct': custom_transaction_count,
                        'Avg_Utilization_Ratio': custom_utilization,
                        'credit_score': custom_credit_score
                    }
                    
                    # Get recommendations
                    recommendations = st.session_state.recommender_model.recommend_cards(custom_profile)
                    
                    # Get user segment
                    user_segment = st.session_state.recommender_model.get_user_segment(custom_profile)
                    
                    # Display user segment with nice UI
                    segment_color = "#4CAF50" if user_segment == "premium" else "#FFC107" if user_segment == "standard" else "#F44336"
                    st.markdown(f"""
                    <div style="background: rgba(26, 31, 42, 0.6); padding: 20px; border-radius: 10px; margin: 20px 0;">
                        <h4 style="color: white; margin-top: 0;">Custom Profile Analysis</h4>
                        <div style="margin: 15px 0;">
                            <div style="font-size: 0.8rem; color: #B8C2E0; margin-bottom: 5px;">User Segment</div>
                            <div style="display: inline-block; background: linear-gradient(45deg, #1A1F2A, {segment_color}); 
                                     padding: 8px 15px; border-radius: 20px; text-transform: capitalize;">
                                <span style="color: white; font-weight: 600; font-size: 1.1rem;">{user_segment}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display recommendations
                    st.markdown("### Recommended Credit Cards")
                    
                    for i, card in enumerate(recommendations):
                        match_score_color = "#4CAF50" if card['match_score'] >= 80 else "#FFC107" if card['match_score'] >= 60 else "#F44336"
                        
                        st.markdown(f"""
                        <div style="background: rgba(26, 31, 42, 0.6); padding: 20px; border-radius: 10px; margin-bottom: 15px; 
                                   border-left: 3px solid {match_score_color};">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                <h4 style="color: white; margin: 0;">{card['name']}</h4>
                                <div style="background: linear-gradient(45deg, #1A1F2A, {match_score_color}); padding: 5px 10px; border-radius: 15px;">
                                    <span style="color: white; font-weight: 600;">{card['match_score']:.0f}% Match</span>
                                </div>
                            </div>
                            
                            <div style="display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 15px;">
                                <div style="flex: 1; min-width: 120px;">
                                    <div style="font-size: 0.8rem; color: #B8C2E0;">Annual Fee</div>
                                    <div style="font-size: 1rem; color: white;">${card['annual_fee']}</div>
                                </div>
                                <div style="flex: 1; min-width: 120px;">
                                    <div style="font-size: 0.8rem; color: #B8C2E0;">Rewards Rate</div>
                                    <div style="font-size: 1rem; color: white;">{card['rewards_rate']*100:.1f}%</div>
                                </div>
                                <div style="flex: 1; min-width: 120px;">
                                    <div style="font-size: 0.8rem; color: #B8C2E0;">Intro APR</div>
                                    <div style="font-size: 1rem; color: white;">{card['intro_apr']}</div>
                                </div>
                                <div style="flex: 1; min-width: 120px;">
                                    <div style="font-size: 0.8rem; color: #B8C2E0;">Foreign Transaction Fee</div>
                                    <div style="font-size: 1rem; color: white;">{card['foreign_transaction_fee']*100:.1f}%</div>
                                </div>
                            </div>
                            
                            <div style="margin-bottom: 15px;">
                                <div style="font-size: 0.8rem; color: #B8C2E0; margin-bottom: 5px;">Benefits</div>
                                <ul style="color: white; margin: 0; padding-left: 20px;">
                                    {' '.join(['<li>' + benefit + '</li>' for benefit in card['benefits']])}
                                </ul>
                            </div>
                            
                            <div>
                                <div style="font-size: 0.8rem; color: #B8C2E0; margin-bottom: 5px;">Ideal For</div>
                                <div style="color: white; font-style: italic;">{card['ideal_for']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Display footer
    st.markdown("---")
    st.markdown("Credit Scoring & Fraud Detection System | © 2023")
