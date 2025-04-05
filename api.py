from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import json
import data_processing
import credit_scoring
import fraud_detection

app = Flask(__name__)

# Initialize models and data
data = None
credit_model = credit_scoring.CreditScoreModel()
fraud_model = fraud_detection.FraudDetectionModel()

@app.route('/api/load_data', methods=['GET'])
def api_load_data():
    """
    Load and preprocess data
    """
    global data
    try:
        data = data_processing.load_data()
        processed_data, _, _ = data_processing.preprocess_data(data)
        
        if processed_data is not None:
            # Get basic stats
            stats = data_processing.get_aggregate_stats(processed_data)
            return jsonify({
                'status': 'success',
                'message': 'Data loaded successfully',
                'stats': stats
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to process data'
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error loading data: {str(e)}'
        }), 500

@app.route('/api/customer/<int:client_num>', methods=['GET'])
def api_get_customer(client_num):
    """
    Get customer information
    """
    global data
    try:
        if data is None:
            data = data_processing.load_data()
            data, _, _ = data_processing.preprocess_data(data)
        
        customer = data_processing.get_customer_profile(data, client_num)
        
        if customer is not None:
            # Convert customer information to JSON-serializable format
            customer_info = customer.to_dict()
            
            # Convert numpy types to Python native types
            for key, value in customer_info.items():
                if isinstance(value, np.int64):
                    customer_info[key] = int(value)
                elif isinstance(value, np.float64):
                    customer_info[key] = float(value)
                elif pd.isna(value):
                    customer_info[key] = None
            
            return jsonify({
                'status': 'success',
                'customer': customer_info
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Customer not found'
            }), 404
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error retrieving customer: {str(e)}'
        }), 500

@app.route('/api/credit_score/<int:client_num>', methods=['GET'])
def api_get_credit_score(client_num):
    """
    Get credit score for a customer
    """
    global data, credit_model
    try:
        if data is None:
            data = data_processing.load_data()
            data, credit_data, _ = data_processing.preprocess_data(data)
            
            # Load or train the credit model
            if not credit_model.load_model():
                credit_model.train(credit_data)
        
        customer = data_processing.get_customer_profile(data, client_num)
        
        if customer is not None:
            # Get credit score prediction
            prediction = credit_model.predict(customer)
            
            return jsonify({
                'status': 'success',
                'client_num': client_num,
                'credit_score': prediction
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Customer not found'
            }), 404
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error calculating credit score: {str(e)}'
        }), 500

@app.route('/api/fraud_detection/<int:client_num>', methods=['GET'])
def api_fraud_detection(client_num):
    """
    Get fraud detection results for a customer
    """
    global data, fraud_model
    try:
        if data is None:
            data = data_processing.load_data()
            data, _, fraud_data = data_processing.preprocess_data(data)
            
            # Load or train the fraud model
            if not fraud_model.load_model():
                fraud_model.train(fraud_data)
        
        customer = data_processing.get_customer_profile(data, client_num)
        
        if customer is not None:
            # Get fraud prediction
            prediction = fraud_model.predict(customer)
            
            return jsonify({
                'status': 'success',
                'client_num': client_num,
                'fraud_detection': prediction
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Customer not found'
            }), 404
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error detecting fraud: {str(e)}'
        }), 500

@app.route('/api/model_metrics', methods=['GET'])
def api_model_metrics():
    """
    Get model metrics for Power BI
    """
    global data, credit_model, fraud_model
    try:
        if data is None:
            data = data_processing.load_data()
            data, credit_data, fraud_data = data_processing.preprocess_data(data)
            
            # Load or train the models
            if not credit_model.load_model():
                credit_model.train(credit_data)
            
            if not fraud_model.load_model():
                fraud_model.train(fraud_data)
        
        # Get model evaluations
        credit_eval = credit_model.evaluate(credit_data)
        fraud_eval = fraud_model.evaluate(fraud_data)
        
        # Convert evaluation results to JSON-serializable format
        credit_eval_json = {}
        fraud_eval_json = {}
        
        if credit_eval:
            credit_eval_json = {
                'accuracy': float(credit_eval['accuracy']),
                'classification_report': credit_eval['classification_report'],
                'feature_importance': credit_eval['feature_importance'].to_dict(orient='records')
            }
        
        if fraud_eval:
            fraud_eval_json = {
                'accuracy': float(fraud_eval['accuracy']),
                'auc_score': float(fraud_eval['auc_score']),
                'classification_report': fraud_eval['classification_report'],
                'feature_importance': fraud_eval['feature_importance'].to_dict(orient='records'),
                'pr_curve_data': fraud_eval['pr_curve_data'].to_dict(orient='records')
            }
        
        return jsonify({
            'status': 'success',
            'credit_model_metrics': credit_eval_json,
            'fraud_model_metrics': fraud_eval_json
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error retrieving model metrics: {str(e)}'
        }), 500

@app.route('/api/export_data', methods=['GET'])
def api_export_data():
    """
    Export processed data for Power BI
    """
    global data
    try:
        if data is None:
            data = data_processing.load_data()
            data, _, _ = data_processing.preprocess_data(data)
        
        # Create a directory for exports
        if not os.path.exists('exports'):
            os.makedirs('exports')
        
        # Export CSV file for Power BI
        export_cols = [
            'Client_Num', 'Customer_Age', 'Gender', 'Income', 'Credit_Limit',
            'Total_Revolving_Bal', 'Total_Trans_Amt', 'Total_Trans_Ct',
            'Avg_Utilization_Ratio', 'Delinquent_Acc', 'credit_score',
            'credit_score_band', 'fraud_flag'
        ]
        
        data[export_cols].to_csv('exports/credit_card_data_for_powerbi.csv', index=False)
        
        return jsonify({
            'status': 'success',
            'message': 'Data exported successfully',
            'file_path': 'exports/credit_card_data_for_powerbi.csv'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error exporting data: {str(e)}'
        }), 500

def start_api():
    """
    Start the Flask API server
    """
    app.run(host='0.0.0.0', port=8000, debug=False)

if __name__ == '__main__':
    start_api()
