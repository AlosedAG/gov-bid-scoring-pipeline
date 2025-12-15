import joblib
import os
import pandas as pd
from utils.cache import get_path

class ModelManager:
    def __init__(self, model_filename="bid_software_model.pkl"):
        self.model = None
        self.path = get_path(model_filename)
        self.load_model()

    def load_model(self):
        if os.path.exists(self.path):
            try:
                self.model = joblib.load(self.path)
                print(f"Model loaded successfully from {self.path}")
            except Exception as e:
                print(f"Failed to load model: {e}")
        else:
            print(f"Warning: Model file not found at {self.path}. Using fallback logic.")

    def get_prob(self, text):
        if pd.isna(text) or text == '':
            return 0.0
        
        if self.model:
            try:
                # Expecting model to be a pipeline with predict_proba
                preds = self.model.predict_proba([str(text)])[0]
                return preds[1] # Probability of class 1
            except Exception as e:
                # Fallback if model structure differs
                return 0.0
        
        return 0.0