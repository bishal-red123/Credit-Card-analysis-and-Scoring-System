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
            joblib.dump(self.model, 'models/credit_score_model.joblib')
    
    def load_model(self):
        """
        Load the saved model from file
        """
        if os.path.exists('models/credit_score_model.joblib'):
            self.model = joblib.load('models/credit_score_model.joblib')
            self.is_trained = True
            return True
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
        if not self.is_trained:
            return None
        
        # Extract features
        features = customer_data[self.features].values.reshape(1, -1)
        
        # Scale the features
        scaled_features = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(scaled_features)[0]
        
        # Get prediction probabilities
        proba = self.model.predict_proba(scaled_features)[0]
        
        # Create response
        result = {
            'credit_score_band': prediction,
            'probability': max(proba),
            'all_probabilities': dict(zip(self.model.classes_, proba))
        }
        
        return result
