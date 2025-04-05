import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

class CreditCardRecommender:
    def __init__(self):
        self.credit_card_data = None
        self.scaler = StandardScaler()
        self.features = [
            'Customer_Age', 'Income', 'Credit_Limit', 
            'Total_Revolving_Bal', 'Total_Trans_Amt', 
            'Total_Trans_Ct', 'Avg_Utilization_Ratio'
        ]
        
        # Credit card offerings with their features/benefits
        self.credit_card_offerings = {
            'Premium Rewards Card': {
                'annual_fee': 95,
                'rewards_rate': 0.02,  # 2% cashback
                'points_multiplier': 3,
                'intro_apr': '0% for 12 months',
                'foreign_transaction_fee': 0,
                'min_credit_score': 720,
                'min_income': 60000,
                'benefits': [
                    'Travel insurance',
                    'Airport lounge access',
                    '3x points on dining and travel',
                    'No foreign transaction fees'
                ],
                'ideal_for': 'High income customers who travel frequently'
            },
            'Cash Back Plus': {
                'annual_fee': 0,
                'rewards_rate': 0.015,  # 1.5% cashback
                'points_multiplier': 2,
                'intro_apr': '0% for 15 months',
                'foreign_transaction_fee': 0.03,
                'min_credit_score': 680,
                'min_income': 40000,
                'benefits': [
                    '2x points on groceries and gas',
                    'Cell phone protection',
                    'Extended warranty'
                ],
                'ideal_for': 'Everyday spenders looking for cash back on regular purchases'
            },
            'Secured Builder Card': {
                'annual_fee': 0,
                'rewards_rate': 0.01,  # 1% cashback
                'points_multiplier': 1,
                'intro_apr': 'N/A',
                'foreign_transaction_fee': 0.03,
                'min_credit_score': 580,
                'min_income': 25000,
                'benefits': [
                    'Credit building tools',
                    'Free credit score monitoring',
                    'Automatic credit line reviews'
                ],
                'ideal_for': 'Customers looking to build or repair credit'
            },
            'Balance Transfer Elite': {
                'annual_fee': 0,
                'rewards_rate': 0.01,  # 1% cashback
                'points_multiplier': 1,
                'intro_apr': '0% for 18 months',
                'foreign_transaction_fee': 0.03,
                'min_credit_score': 690,
                'min_income': 45000,
                'benefits': [
                    'No balance transfer fee for first 60 days',
                    'Long 0% intro APR period',
                    'Free credit score access'
                ],
                'ideal_for': 'Customers with existing credit card debt looking to save on interest'
            },
            'Business Rewards Card': {
                'annual_fee': 125,
                'rewards_rate': 0.018,  # 1.8% cashback
                'points_multiplier': 3,
                'intro_apr': '0% for 9 months',
                'foreign_transaction_fee': 0,
                'min_credit_score': 700,
                'min_income': 75000,
                'benefits': [
                    '3x points on business expenses',
                    'Employee cards at no additional cost',
                    'Detailed spending reports',
                    'Travel and purchase protections'
                ],
                'ideal_for': 'Small business owners and entrepreneurs'
            }
        }
    
    def _save_model(self):
        """
        Save the model to file
        """
        # Create directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Save the scaler for later use
        try:
            joblib.dump(self.scaler, 'models/credit_card_recommender_scaler.joblib')
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self):
        """
        Load the saved model from file
        """
        try:
            # Check if model file exists
            if os.path.exists('models/credit_card_recommender_scaler.joblib'):
                self.scaler = joblib.load('models/credit_card_recommender_scaler.joblib')
                return True
            return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def train(self, credit_data):
        """
        Train the recommendation system
        
        Parameters:
        -----------
        credit_data : dict
            Dictionary containing training data and features
        """
        try:
            # Get the processed data
            self.credit_card_data = credit_data['processed_data']
            
            # Extract features and scale them
            feature_data = self.credit_card_data[self.features].copy()
            self.scaler.fit(feature_data)
            
            # Save the model
            self._save_model()
            return True
        except Exception as e:
            print(f"Error training recommendation system: {str(e)}")
            return False
    
    def get_user_segment(self, customer_data):
        """
        Determine the user segment based on their profile
        
        Parameters:
        -----------
        customer_data : pandas.Series or dict
            Customer profile data
        
        Returns:
        --------
        str
            User segment ('premium', 'standard', 'starter')
        """
        # Convert to pandas Series if dict
        if isinstance(customer_data, dict):
            customer_data = pd.Series(customer_data)
        
        # Determine segment based on income, credit score, and spending
        income = customer_data.get('Income', 0)
        credit_score = customer_data.get('credit_score', 650)
        credit_limit = customer_data.get('Credit_Limit', 0)
        
        if income >= 75000 and credit_score >= 720:
            return 'premium'
        elif income >= 45000 and credit_score >= 680:
            return 'standard'
        else:
            return 'starter'
    
    def get_similar_customers(self, customer_data, top_n=5):
        """
        Find similar customers to the given customer
        
        Parameters:
        -----------
        customer_data : pandas.Series or dict
            Customer profile data
        top_n : int
            Number of similar customers to return
        
        Returns:
        --------
        pandas.DataFrame
            Top N similar customers
        """
        if self.credit_card_data is None:
            return None
        
        # Convert to pandas Series if dict
        if isinstance(customer_data, dict):
            customer_data = pd.Series(customer_data)
        
        # Extract features that we have in our model
        customer_features = []
        for feature in self.features:
            if feature in customer_data:
                customer_features.append(customer_data[feature])
            else:
                # Use mean value from training data if feature is missing
                customer_features.append(self.credit_card_data[feature].mean())
        
        try:
            # Scale customer features
            customer_features_scaled = self.scaler.transform(np.array(customer_features).reshape(1, -1))
            
            # Scale all customer data
            all_customers_scaled = self.scaler.transform(self.credit_card_data[self.features])
            
            # Calculate similarity
            similarities = cosine_similarity(customer_features_scaled, all_customers_scaled)
            
            # Get indices of top N similar customers
            similar_indices = similarities[0].argsort()[::-1][:top_n]
            
            # Get similar customers
            similar_customers = self.credit_card_data.iloc[similar_indices].copy()
            
            # Make sure the client number column exists and is standardized
            # Add Client_No column if it doesn't exist
            if 'Client_No' not in similar_customers.columns and 'Client_Number' in similar_customers.columns:
                similar_customers['Client_No'] = similar_customers['Client_Number']
            elif 'Client_No' not in similar_customers.columns and 'client_num' in similar_customers.columns:
                similar_customers['Client_No'] = similar_customers['client_num']
            elif 'Client_No' not in similar_customers.columns:
                # Create a synthetic client number if none exists
                similar_customers['Client_No'] = range(100001, 100001 + len(similar_customers))
            
            # Return similar customers with standardized columns
            return similar_customers
            
        except Exception as e:
            print(f"Error finding similar customers: {str(e)}")
            # Create a sample dataframe with the most essential columns
            columns = ['Client_No', 'Customer_Age', 'Income', 'Credit_Limit', 
                       'Total_Trans_Amt', 'Total_Trans_Ct']
            return pd.DataFrame(columns=columns)
    
    def recommend_cards(self, customer_data, top_n=3):
        """
        Recommend credit cards for a customer
        
        Parameters:
        -----------
        customer_data : pandas.Series or dict
            Customer profile data
        top_n : int
            Number of card recommendations to return
        
        Returns:
        --------
        list
            List of recommended card dictionaries with card details and match scores
        """
        # Convert to pandas Series if dict
        if isinstance(customer_data, dict):
            customer_data = pd.Series(customer_data)
            
        # Get key customer attributes
        income = customer_data.get('Income', 0)
        credit_score = customer_data.get('credit_score', 650)
        revolving_balance = customer_data.get('Total_Revolving_Bal', 0)
        utilization = customer_data.get('Avg_Utilization_Ratio', 0)
        transaction_amount = customer_data.get('Total_Trans_Amt', 0)
        transaction_count = customer_data.get('Total_Trans_Ct', 0)
        
        # Calculate scores for each card based on customer attributes
        card_scores = {}
        for card_name, card_info in self.credit_card_offerings.items():
            score = 0
            
            # Check if customer meets minimum requirements
            if credit_score < card_info['min_credit_score'] or income < card_info['min_income']:
                score -= 50  # Major penalty for not meeting minimums
            
            # Score based on income match
            income_match = min(income / card_info['min_income'], 2.0)  # Cap at 2x minimum
            score += 20 * income_match
            
            # Score based on credit score match
            credit_score_match = (credit_score - card_info['min_credit_score']) / 150  # Normalize
            score += 20 * max(0, min(credit_score_match, 1.0))  # Between 0 and 20
            
            # High utilization customers benefit from balance transfer cards
            if utilization > 0.5 and 'Balance Transfer' in card_name:
                score += 30
            
            # High transaction count customers benefit from rewards cards
            if transaction_count > 50 and card_info['rewards_rate'] > 0.015:
                score += 25
            
            # High spending customers benefit from points multipliers
            if transaction_amount > 5000 and card_info['points_multiplier'] > 1:
                score += 20
            
            # Account for annual fee (lower score for high fee)
            if card_info['annual_fee'] > 0:
                fee_impact = min(20, card_info['annual_fee'] / 10)
                score -= fee_impact
            
            # Store the score
            card_scores[card_name] = min(100, max(0, score))  # Ensure score is between 0 and 100
        
        # Sort cards by score
        sorted_cards = sorted(card_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Prepare recommendations
        recommendations = []
        for card_name, score in sorted_cards[:top_n]:
            card_info = self.credit_card_offerings[card_name].copy()
            card_info['name'] = card_name
            card_info['match_score'] = score
            recommendations.append(card_info)
        
        return recommendations