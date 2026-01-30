import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os

class SpamDetectionModel:
    def __init__(self, data_path='../data/spam.csv'):
        self.data_path = data_path
        self.vectorizer = None
        self.models = {}
        self.metrics = {}
        
    def load_data(self):
        """Load and preprocess the spam dataset"""
        df = pd.read_csv(self.data_path, encoding='latin-1')
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        df = df.dropna()
        return df
    
    def preprocess_data(self, df):
        """Preprocess text data"""
        df['message'] = df['message'].str.lower()
        df['message'] = df['message'].str.replace(r'[^\w\s]', '', regex=True)
        return df
    
    def vectorize_text(self, X_train, X_test):
        """Convert text to TF-IDF vectors"""
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        return X_train_vec, X_test_vec
    
    def train_naive_bayes(self, X_train, y_train, X_test, y_test):
        """Train Naive Bayes model"""
        model = MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        self.models['naive_bayes'] = model
        self.metrics['naive_bayes'] = self._calculate_metrics(y_test, y_pred)
        
        return model, self.metrics['naive_bayes']
    
    def train_svm(self, X_train, y_train, X_test, y_test):
        """Train SVM model"""
        model = LinearSVC(random_state=42, max_iter=2000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        self.models['svm'] = model
        self.metrics['svm'] = self._calculate_metrics(y_test, y_pred)
        
        return model, self.metrics['svm']
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        self.models['random_forest'] = model
        self.metrics['random_forest'] = self._calculate_metrics(y_test, y_pred)
        
        return model, self.metrics['random_forest']
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        self.models['xgboost'] = model
        self.metrics['xgboost'] = self._calculate_metrics(y_test, y_pred)
        
        return model, self.metrics['xgboost']
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred)),
            'recall': float(recall_score(y_true, y_pred)),
            'f1': float(f1_score(y_true, y_pred))
        }
    
    def save_models(self, model_dir='../models'):
        """Save all trained models"""
        os.makedirs(model_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            with open(f'{model_dir}/{model_name}_model.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        with open(f'{model_dir}/vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def train_all(self):
        """Train all models"""
        print("Loading data...")
        df = self.load_data()
        
        print("Preprocessing data...")
        df = self.preprocess_data(df)
        
        X = df['message']
        y = df['label']
        
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Vectorizing text...")
        X_train_vec, X_test_vec = self.vectorize_text(X_train, X_test)
        
        print("\nTraining models...")
        
        print("Training Naive Bayes...")
        self.train_naive_bayes(X_train_vec, y_train, X_test_vec, y_test)
        
        print("Training SVM...")
        self.train_svm(X_train_vec, y_train, X_test_vec, y_test)
        
        print("Training Random Forest...")
        self.train_random_forest(X_train_vec, y_train, X_test_vec, y_test)
        
        print("Training XGBoost...")
        self.train_xgboost(X_train_vec, y_train, X_test_vec, y_test)
        
        print("\nModel Performance:")
        for model_name, metrics in self.metrics.items():
            print(f"\n{model_name.upper()}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
        
        print("\nSaving models...")
        self.save_models()
        print("Models saved successfully!")

if __name__ == "__main__":
    trainer = SpamDetectionModel('../data/spam.csv')
    trainer.train_all()
