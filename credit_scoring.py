import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import joblib
import os

class CreditScoreModel:
    def __init__(self):
        self.model = None
        self.features = None
        self.scaler = None
        self.is_trained = False
        self.model_type = 'rf'  # Default model type: 'rf' for Random Forest, 'xgb' for XGBoost
        
    def train(self, credit_data, model_type='xgb'):
        """
        Train the credit scoring model
        
        Parameters:
        -----------
        credit_data : dict
            Dictionary containing training data and features
        model_type : str, optional
            Type of model to train: 'rf' for Random Forest, 'xgb' for XGBoost
            Default is 'xgb'
        """
        X_train = credit_data['X_train']
        y_train = credit_data['y_train']
        self.features = credit_data['features']
        self.scaler = credit_data['scaler']
        self.model_type = model_type
        
        if model_type == 'rf':
            # Using Random Forest for credit scoring
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Train the model
            self.model.fit(X_train, y_train)
            
        elif model_type == 'xgb':
            # Using XGBoost for credit scoring
            # Convert string labels to integers for XGBoost
            label_map = {label: i for i, label in enumerate(y_train.unique())}
            y_train_encoded = y_train.map(label_map)
            
            # Create DMatrix for XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train_encoded)
            
            # XGBoost parameters
            params = {
                'objective': 'multi:softprob',  # Multiclass classification with probability outputs
                'num_class': len(label_map),    # Number of classes
                'eval_metric': 'mlogloss',      # Multiclass logloss
                'eta': 0.1,                     # Learning rate
                'max_depth': 6,                 # Maximum tree depth
                'subsample': 0.8,               # Subsample ratio of training instances
                'colsample_bytree': 0.8,        # Subsample ratio of columns
                'min_child_weight': 1,          # Minimum sum of instance weight needed in a child
                'seed': 42                      # Random seed
            }
            
            # Train XGBoost model
            self.model = xgb.train(params, dtrain, num_boost_round=100)
            
            # Store label mapping for prediction
            self.model.label_map = label_map
            self.model.label_map_inverse = {v: k for k, v in label_map.items()}
        
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
        
        if self.model_type == 'rf':
            # Random Forest evaluation
            y_pred = self.model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            feature_importance = pd.DataFrame({
                'feature': self.features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
        elif self.model_type == 'xgb':
            # XGBoost evaluation
            # Convert test data to DMatrix
            dtest = xgb.DMatrix(X_test)
            
            # Get predictions
            y_pred_probs = self.model.predict(dtest)
            
            # Convert from one-hot encoded probabilities to class indices
            if len(y_pred_probs.shape) > 1:  # Multi-class
                y_pred_indices = np.argmax(y_pred_probs, axis=1)
            else:  # Binary classification
                y_pred_indices = (y_pred_probs > 0.5).astype(int)
            
            # Map class indices back to original labels
            y_pred = [self.model.label_map_inverse[idx] for idx in y_pred_indices]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Get feature importance from XGBoost model
            importance_scores = self.model.get_score(importance_type='weight')
            
            # Some features might not be in the importance scores if they weren't used
            # Create a DataFrame with all features and set unused features to 0
            feature_importance = pd.DataFrame({
                'feature': self.features,
                'importance': [importance_scores.get(f'f{i}', 0) for i in range(len(self.features))]
            }).sort_values('importance', ascending=False)
        
        evaluation = {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'feature_importance': feature_importance,
            'model_type': self.model_type
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
                'all_probabilities': {},
                'model_type': self.model_type if hasattr(self, 'model_type') else 'unknown'
            }
            
        # Check if customer_data is None (customer not found)
        if customer_data is None:
            return {
                'error': 'Customer not found',
                'credit_score_band': 'Unknown',
                'probability': 0,
                'all_probabilities': {},
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
                    'credit_score_band': 'Unknown',
                    'probability': 0,
                    'all_probabilities': {},
                    'model_type': self.model_type
                }
                
            # Extract features
            features = customer_data[self.features].values.reshape(1, -1)
            
            # Scale the features
            scaled_features = self.scaler.transform(features)
            
            if self.model_type == 'rf':
                # Random Forest prediction
                prediction = self.model.predict(scaled_features)[0]
                
                # Get prediction probabilities
                proba = self.model.predict_proba(scaled_features)[0]
                
                # Create class-probability dictionary
                if hasattr(self.model, 'classes_') and self.model.classes_ is not None:
                    all_probabilities = dict(zip(self.model.classes_, proba))
                else:
                    all_probabilities = {}
                
                max_prob = max(proba) if len(proba) > 0 else 0
                
            elif self.model_type == 'xgb':
                # XGBoost prediction
                # Convert to DMatrix for XGBoost
                dtest = xgb.DMatrix(scaled_features)
                
                # Predict probabilities
                y_pred_probs = self.model.predict(dtest)
                
                # For multi-class, convert from probability array to class and probability
                if len(y_pred_probs.shape) > 1:  # Multi-class
                    y_pred_indices = np.argmax(y_pred_probs[0])
                    prediction = self.model.label_map_inverse[y_pred_indices]
                    max_prob = y_pred_probs[0][y_pred_indices]
                    
                    # Create probabilities dictionary for all classes
                    all_probabilities = {
                        self.model.label_map_inverse[i]: float(prob) 
                        for i, prob in enumerate(y_pred_probs[0])
                    }
                else:  # Binary classification
                    y_pred_idx = 1 if y_pred_probs[0] > 0.5 else 0
                    prediction = self.model.label_map_inverse[y_pred_idx]
                    max_prob = float(y_pred_probs[0]) if y_pred_idx == 1 else 1 - float(y_pred_probs[0])
                    
                    all_probabilities = {
                        self.model.label_map_inverse[0]: 1 - float(y_pred_probs[0]),
                        self.model.label_map_inverse[1]: float(y_pred_probs[0])
                    }
            
            # Create response
            result = {
                'credit_score_band': prediction,
                'probability': max_prob,
                'all_probabilities': all_probabilities,
                'model_type': self.model_type
            }
            
            return result
        except Exception as e:
            return {
                'error': f'Prediction error: {str(e)}',
                'credit_score_band': 'Unknown',
                'probability': 0,
                'all_probabilities': {},
                'model_type': self.model_type
            }
