from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import json
import data_processing
import credit_scoring
import fraud_detection
import recommendation_system

app = Flask(__name__)

# Initialize models and data
data = None
credit_model = credit_scoring.CreditScoreModel()
fraud_model = fraud_detection.FraudDetectionModel()
recommender = recommendation_system.CreditCardRecommender()

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

@app.route('/api/recommend_cards/<int:client_num>', methods=['GET'])
def api_recommend_cards(client_num):
    """
    Get credit card recommendations for a customer
    """
    global data, credit_model, recommender
    try:
        if data is None:
            data = data_processing.load_data()
            data, credit_data, _ = data_processing.preprocess_data(data)
            
            # Load or train the credit model (for credit scores)
            if not credit_model.load_model():
                credit_model.train(credit_data)
                
            # Load or train the recommender model
            if not recommender.load_model():
                recommender.train({'processed_data': data})
        
        customer = data_processing.get_customer_profile(data, client_num)
        
        if customer is not None:
            # First get credit score if it doesn't exist in customer data
            if 'credit_score' not in customer:
                credit_prediction = credit_model.predict(customer)
                customer['credit_score'] = credit_prediction.get('credit_score_numeric', 650)
            
            # Get recommendations
            recommendations = recommender.recommend_cards(customer)
            
            # Get similar customers
            similar_customers = recommender.get_similar_customers(customer)
            similar_client_nums = []
            if similar_customers is not None:
                similar_client_nums = similar_customers['Client_No'].tolist()
            
            # Get user segment
            user_segment = recommender.get_user_segment(customer)
            
            return jsonify({
                'status': 'success',
                'client_num': client_num,
                'user_segment': user_segment,
                'recommendations': recommendations,
                'similar_client_nums': similar_client_nums
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Customer not found'
            }), 404
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error getting recommendations: {str(e)}'
        }), 500

# Power BI integration removed

# Power BI export data endpoint removed

def start_api():
    """
    Start the Flask API server
    """
    app.run(host='0.0.0.0', port=8000, debug=False)

if __name__ == '__main__':
    start_api()
