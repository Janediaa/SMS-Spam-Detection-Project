import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV

class SpamPredictor:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.models = {}
        self.vectorizer = None
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        models_to_load = ['naive_bayes', 'svm', 'random_forest', 'xgboost']
        
        for model_name in models_to_load:
            model_path = f'{self.model_dir}/{model_name}_model.pkl'
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
        
        vectorizer_path = f'{self.model_dir}/vectorizer.pkl'
        if os.path.exists(vectorizer_path):
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
    
    def preprocess_text(self, text):
        """Preprocess input text"""
        text = text.lower()
        text = ''.join(c for c in text if c.isalnum() or c.isspace())
        return text
    
    def predict(self, message):
        """Predict if message is spam or ham using all models"""
        if not self.models or self.vectorizer is None:
            return None
        
        processed_message = self.preprocess_text(message)
        vectorized_message = self.vectorizer.transform([processed_message])
        
        results = {}
        for model_name, model in self.models.items():
            prediction = model.predict(vectorized_message)[0]
            
            # Handle SVM separately since it doesn't have predict_proba
            if model_name == 'svm':
                # Use decision function for SVM
                decision = model.decision_function(vectorized_message)[0]
                # Convert to probability-like scores
                spam_prob = 1 / (1 + np.exp(-decision))
                ham_prob = 1 - spam_prob
            else:
                # Other models have predict_proba
                confidence = model.predict_proba(vectorized_message)[0]
                spam_prob = float(confidence[1])
                ham_prob = float(confidence[0])
            
            results[model_name] = {
                'prediction': 'SPAM' if prediction == 1 else 'HAM',
                'spam_probability': float(spam_prob),
                'ham_probability': float(ham_prob)
            }
        
        return results
    
    def get_ensemble_prediction(self, message):
        """Get ensemble prediction using voting"""
        predictions = self.predict(message)
        
        if predictions is None:
            return None
        
        spam_votes = sum(1 for p in predictions.values() if p['prediction'] == 'SPAM')
        total_votes = len(predictions)
        
        ensemble_result = {
            'prediction': 'SPAM' if spam_votes > total_votes / 2 else 'HAM',
            'confidence': spam_votes / total_votes,
            'individual_predictions': predictions
        }
        
        return ensemble_result