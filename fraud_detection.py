import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_recall_curve
import xgboost as xgb
import joblib
import os

class FraudDetectionModel:
    def __init__(self):
        self.model = None
        self.features = None
        self.scaler = None
        self.is_trained = False
        self.model_type = 'rf'  # Default model type: 'rf' for Random Forest, 'xgb' for XGBoost
        
    def train(self, fraud_data, model_type='xgb'):
        """
        Train the fraud detection model
        
        Parameters:
        -----------
        fraud_data : dict
            Dictionary containing training data and features
        model_type : str, optional
            Type of model to train: 'rf' for Random Forest, 'xgb' for XGBoost
            Default is 'xgb'
        """
        X_train = fraud_data['X_train']
        y_train = fraud_data['y_train']
        self.features = fraud_data['features']
        self.scaler = fraud_data['scaler']
        self.model_type = model_type
        
        if model_type == 'rf':
            # Using Random Forest for fraud detection
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42
            )
            
            # Train the model
            self.model.fit(X_train, y_train)
            
        elif model_type == 'xgb':
            # Using XGBoost for fraud detection
            # XGBoost works best with binary classes as 0 and 1
            # No need to convert if they are already numeric
            
            # Create DMatrix for XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train)
            
            # XGBoost parameters optimized for imbalanced binary classification
            params = {
                'objective': 'binary:logistic',  # Binary classification
                'eval_metric': 'auc',            # AUC for imbalanced data
                'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),  # Balance class weights
                'eta': 0.05,                     # Lower learning rate for better convergence
                'max_depth': 5,                  # Maximum tree depth
                'subsample': 0.8,                # Subsample ratio
                'colsample_bytree': 0.8,         # Feature sampling per tree
                'min_child_weight': 1,           # Controls complexity
                'gamma': 1,                      # Minimum loss reduction for split
                'seed': 42                       # Random seed
            }
            
            # Train XGBoost model with early stopping
            self.model = xgb.train(
                params, 
                dtrain, 
                num_boost_round=200
            )
            
            # Store class labels for prediction
            self.model.classes_ = np.array([0, 1])  # Binary classification
            
        else:
            raise ValueError(f"Unknown model type: {model_type}. Choose 'rf' or 'xgb'.")
        
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
            # Save the model, features, scaler and model_type together as a dictionary
            model_data = {
                'model': self.model,
                'features': self.features,
                'scaler': self.scaler,
                'model_type': self.model_type
            }
            joblib.dump(model_data, 'models/fraud_detection_model.joblib')
    
    def load_model(self):
        """
        Load the saved model from file
        """
        if os.path.exists('models/fraud_detection_model.joblib'):
            try:
                # Load the dictionary containing model, features, and scaler
                model_data = joblib.load('models/fraud_detection_model.joblib')
                
                # Check if it's the old format (just the model) or new format (dictionary)
                if isinstance(model_data, dict) and 'model' in model_data:
                    self.model = model_data['model']
                    self.features = model_data['features']
                    self.scaler = model_data['scaler']
                    
                    # Set model_type if available, otherwise default to 'rf'
                    if 'model_type' in model_data:
                        self.model_type = model_data['model_type']
                    else:
                        self.model_type = 'rf'  # Default to Random Forest for backward compatibility
                else:
                    # Handle legacy format
                    self.model = model_data
                    self.model_type = 'rf'  # Default to Random Forest for legacy models
                    # Features and scaler will need to be set separately
                
                self.is_trained = True if self.model is not None else False
                return True
            except Exception as e:
                print(f"Error loading fraud detection model: {str(e)}")
                return False
        return False
    
    def evaluate(self, fraud_data):
        """
        Evaluate the model performance
        """
        if not self.is_trained:
            return None
        
        X_test = fraud_data['X_test']
        y_test = fraud_data['y_test']
        
        if self.model_type == 'rf':
            # Random Forest evaluation
            y_pred = self.model.predict(X_test)
            y_proba = self.model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_proba)
            
            # Get feature importances for Random Forest
            feature_importance = pd.DataFrame({
                'feature': self.features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
        elif self.model_type == 'xgb':
            # XGBoost evaluation
            # Convert test data to DMatrix
            dtest = xgb.DMatrix(X_test)
            
            # Get predictions (probabilities)
            y_proba = self.model.predict(dtest)
            
            # Binary classification - threshold at 0.5 for class prediction
            y_pred = (y_proba > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_proba)
            
            # Get feature importance from XGBoost model
            importance_scores = self.model.get_score(importance_type='weight')
            
            # Create a DataFrame with all features and set unused features to 0
            feature_importance = pd.DataFrame({
                'feature': self.features,
                'importance': [importance_scores.get(f'f{i}', 0) for i in range(len(self.features))]
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
            'pr_curve_data': pr_curve_data,
            'model_type': self.model_type
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
                'risk_level': 'Unknown',
                'model_type': self.model_type if hasattr(self, 'model_type') else 'unknown'
            }
        
        # Check if customer_data is None (customer not found)
        if customer_data is None:
            return {
                'error': 'Customer not found',
                'fraud_flag': 0,
                'fraud_probability': 0,
                'risk_level': 'Unknown',
                'model_type': self.model_type
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
                    'risk_level': 'Unknown',
                    'model_type': self.model_type
                }
                
            # Extract features
            features = customer_data[self.features].values.reshape(1, -1)
            
            # Scale the features
            scaled_features = self.scaler.transform(features)
            
            if self.model_type == 'rf':
                # Random Forest prediction
                prediction = self.model.predict(scaled_features)[0]
                
                # Get fraud probability
                proba_result = self.model.predict_proba(scaled_features)[0]
                
                # Check if we have at least 2 classes and get the fraud probability (class 1)
                if len(proba_result) >= 2:
                    fraud_probability = proba_result[1]
                else:
                    fraud_probability = 0
                    
            elif self.model_type == 'xgb':
                # XGBoost prediction
                # Convert to DMatrix for XGBoost
                dtest = xgb.DMatrix(scaled_features)
                
                # For binary classification, XGBoost returns the probability of the positive class
                fraud_probability = float(self.model.predict(dtest)[0])
                
                # Determine class based on threshold
                prediction = 1 if fraud_probability > 0.5 else 0
            
            # Create response
            result = {
                'fraud_flag': int(prediction),
                'fraud_probability': fraud_probability,
                'risk_level': 'High' if fraud_probability > 0.7 else 'Medium' if fraud_probability > 0.3 else 'Low',
                'model_type': self.model_type
            }
            
            return result
        except Exception as e:
            return {
                'error': f'Prediction error: {str(e)}',
                'fraud_flag': 0,
                'fraud_probability': 0,
                'risk_level': 'Unknown',
                'model_type': self.model_type
            }
