import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from django.conf import settings

class LoanPrediction:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.load_model()

    def load_model(self):
        """Load the trained model and scaler"""
        model_path = os.path.join(settings.BASE_DIR, 'loan_manager', 'ml', 'model.joblib')
        self.model = joblib.load(model_path)
        self.scaler = self.model['scaler']
        self.label_encoders = self.model['label_encoders']

    def preprocess_input(self, data):
        """Preprocess the input data"""
        # Convert to DataFrame if it's not already
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame([data])

        # Handle categorical variables
        categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 
                          'Self_Employed', 'Property_Area']
        
        for col in categorical_cols:
            if col in data.columns:
                le = self.label_encoders.get(col)
                if le is not None:
                    data[col] = le.transform(data[col])

        # Scale numerical features
        numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                         'Loan_Amount_Term', 'Credit_History']
        
        if self.scaler is not None:
            data[numerical_cols] = self.scaler.transform(data[numerical_cols])

        return data

    def predict(self, data):
        """Make loan prediction"""
        # Preprocess the input data
        processed_data = self.preprocess_input(data)
        
        # Make prediction
        prediction = self.model['model'].predict(processed_data)
        probability = self.model['model'].predict_proba(processed_data)
        
        return {
            'prediction': prediction[0],
            'probability': probability[0].tolist()
        }

    def get_feature_importance(self):
        """Get feature importance from the model"""
        if hasattr(self.model['model'], 'feature_importances_'):
            return dict(zip(self.model['feature_names'], 
                          self.model['model'].feature_importances_))
        return None 