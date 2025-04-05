import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_recall_curve
import joblib
import os

class FraudDetectionModel:
    def __init__(self):
        self.model = None
        self.features = None
        self.scaler = None
        self.is_trained = False
        
    def train(self, fraud_data):
        """
        Train the fraud detection model
        """
        X_train = fraud_data['X_train']
        y_train = fraud_data['y_train']
        self.features = fraud_data['features']
        self.scaler = fraud_data['scaler']
        
        # Using Random Forest for fraud detection
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
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
            joblib.dump(self.model, 'models/fraud_detection_model.joblib')
    
    def load_model(self):
        """
        Load the saved model from file
        """
        if os.path.exists('models/fraud_detection_model.joblib'):
            self.model = joblib.load('models/fraud_detection_model.joblib')
            self.is_trained = True
            return True
        return False
    
    def evaluate(self, fraud_data):
        """
        Evaluate the model performance
        """
        if not self.is_trained:
            return None
        
        X_test = fraud_data['X_test']
        y_test = fraud_data['y_test']
        
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_proba)
        
        # Get feature importances
        feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calculate precision-recall points for curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        pr_curve_data = pd.DataFrame({
            'precision': precision,
            'recall': recall,
            'threshold': np.append(thresholds, 0)  # Add 0 for the last point
        })
        
        evaluation = {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'auc_score': auc_score,
            'feature_importance': feature_importance,
            'pr_curve_data': pr_curve_data
        }
        
        return evaluation
    
    def predict(self, customer_data):
        """
        Predict fraud probability for a customer
        """
        if not self.is_trained or self.model is None or self.features is None or self.scaler is None:
            return {
                'error': 'Model not ready',
                'fraud_flag': 0,
                'fraud_probability': 0,
                'risk_level': 'Unknown'
            }
        
        # Check if customer_data is None (customer not found)
        if customer_data is None:
            return {
                'error': 'Customer not found',
                'fraud_flag': 0,
                'fraud_probability': 0,
                'risk_level': 'Unknown'
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
                    'fraud_flag': 0,
                    'fraud_probability': 0,
                    'risk_level': 'Unknown'
                }
                
            # Extract features
            features = customer_data[self.features].values.reshape(1, -1)
            
            # Scale the features
            scaled_features = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(scaled_features)[0]
            
            # Get fraud probability
            proba_result = self.model.predict_proba(scaled_features)[0]
            
            # Check if we have at least 2 classes and get the fraud probability (class 1)
            if len(proba_result) >= 2:
                fraud_probability = proba_result[1]
            else:
                fraud_probability = 0
            
            # Create response
            result = {
                'fraud_flag': int(prediction),
                'fraud_probability': fraud_probability,
                'risk_level': 'High' if fraud_probability > 0.7 else 'Medium' if fraud_probability > 0.3 else 'Low'
            }
            
            return result
        except Exception as e:
            return {
                'error': f'Prediction error: {str(e)}',
                'fraud_flag': 0,
                'fraud_probability': 0,
                'risk_level': 'Unknown'
            }
