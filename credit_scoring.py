import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

class CreditScoreModel:
    def __init__(self):
        self.model = None
        self.features = None
        self.scaler = None
        self.is_trained = False
        
    def train(self, credit_data):
        """
        Train the credit scoring model
        """
        X_train = credit_data['X_train']
        y_train = credit_data['y_train']
        self.features = credit_data['features']
        self.scaler = credit_data['scaler']
        
        # Using Random Forest for credit scoring
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Save the model
        self._save_model()
        
        return self.model
    
    def _save_model(self):
        """
        Save the model to file
        """
        if not os.path.exists('models'):
            os.makedirs('models')
        
        if self.model:
            # Save the model, features, and scaler together as a dictionary
            model_data = {
                'model': self.model,
                'features': self.features,
                'scaler': self.scaler
            }
            joblib.dump(model_data, 'models/credit_score_model.joblib')
    
    def load_model(self):
        """
        Load the saved model from file
        """
        if os.path.exists('models/credit_score_model.joblib'):
            try:
                # Load the dictionary containing model, features, and scaler
                model_data = joblib.load('models/credit_score_model.joblib')
                
                # Check if it's the old format (just the model) or new format (dictionary)
                if isinstance(model_data, dict) and 'model' in model_data:
                    self.model = model_data['model']
                    self.features = model_data['features']
                    self.scaler = model_data['scaler']
                else:
                    # Handle legacy format
                    self.model = model_data
                    # Features and scaler will need to be set separately
                
                self.is_trained = True if self.model is not None else False
                return True
            except Exception as e:
                print(f"Error loading credit score model: {str(e)}")
                return False
        return False
    
    def evaluate(self, credit_data):
        """
        Evaluate the model performance
        """
        if not self.is_trained:
            return None
        
        X_test = credit_data['X_test']
        y_test = credit_data['y_test']
        
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        evaluation = {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'feature_importance': feature_importance
        }
        
        return evaluation
    
    def predict(self, customer_data):
        """
        Predict credit score for a customer
        """
        if not self.is_trained or self.model is None or self.features is None or self.scaler is None:
            return {
                'error': 'Model not ready',
                'credit_score_band': 'Unknown',
                'probability': 0,
                'all_probabilities': {}
            }
            
        # Check if customer_data is None (customer not found)
        if customer_data is None:
            return {
                'error': 'Customer not found',
                'credit_score_band': 'Unknown',
                'probability': 0,
                'all_probabilities': {}
            }
        
        try:
            # Initialize empty list for missing features
            missing_features = []
            
            # Check if all required features are present
            for f in self.features:
                if f not in customer_data:
                    missing_features.append(f)
            
            if missing_features:
                return {
                    'error': f'Missing features: {missing_features}',
                    'credit_score_band': 'Unknown',
                    'probability': 0,
                    'all_probabilities': {}
                }
                
            # Extract features
            features = customer_data[self.features].values.reshape(1, -1)
            
            # Scale the features
            scaled_features = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(scaled_features)[0]
            
            # Get prediction probabilities
            proba = self.model.predict_proba(scaled_features)[0]
            
            # Create class-probability dictionary, ensuring classes_ is available
            if hasattr(self.model, 'classes_') and self.model.classes_ is not None:
                all_probabilities = dict(zip(self.model.classes_, proba))
            else:
                all_probabilities = {}
            
            # Create response
            result = {
                'credit_score_band': prediction,
                'probability': max(proba) if len(proba) > 0 else 0,
                'all_probabilities': all_probabilities
            }
            
            return result
        except Exception as e:
            return {
                'error': f'Prediction error: {str(e)}',
                'credit_score_band': 'Unknown',
                'probability': 0,
                'all_probabilities': {}
            }
