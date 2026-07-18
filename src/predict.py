import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

class PhishingDetector:
    def __init__(self, model_path='models/phishing_cnn_model.h5', scaler_path='models/scaler.pkl'):
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        
    def predict_single(self, features):
        feat_df = pd.DataFrame([features])
        feat_scaled = self.scaler.transform(feat_df)
        proba = self.model.predict(np.expand_dims(feat_scaled, axis=-1), verbose=0)[0][0]
        prediction = 1 if proba > 0.5 else 0
        return {
            'prediction': prediction,
            'probability': float(proba),
            'confidence': float(proba if prediction == 1 else 1 - proba),
            'status': 'Phishing' if prediction == 1 else 'Legitimate'
        }
    
    def predict_batch(self, features_list):
        return [self.predict_single(f) for f in features_list]
    
    def get_feature_importance(self):
        raw_w = np.abs(self.model.layers[0].get_weights()[0])
        return np.mean(raw_w, axis=(0, 1))[:30]

detector = PhishingDetector()

def predict_phishing(features):
    return detector.predict_single(features)
