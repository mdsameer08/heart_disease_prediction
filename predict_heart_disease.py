
import numpy as np
import pandas as pd
import pickle

def predict_heart_disease(data, model_file='heart-disease-prediction-model.pkl'):
    """
    Make heart disease predictions on new data.
    
    Parameters:
    -----------
    data : pandas DataFrame
        New data to make predictions on. Should contain the same features as the training data.
    model_file : str
        Path to the saved model file.
        
    Returns:
    --------
    predictions : numpy array
        Binary predictions (0 or 1) for heart disease.
    probabilities : numpy array
        Probability of heart disease for each sample.
    """
    # Load the model
    model_data = pickle.load(open(model_file, 'rb'))
    
    # Extract required components
    if model_data['type'] == 'single-model':
        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['features']
        
        # Select and scale features
        X = data[features]
        X_scaled = scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else None
        
    else:  # meta-ensemble
        models = model_data['models']
        meta_clf = model_data['meta_classifier']
        scaler = model_data['scaler']
        features = model_data['features']
        
        # Select and scale features
        X = data[features]
        X_scaled = scaler.transform(X)
        
        # Get predictions from all models
        preds = np.column_stack([model.predict(X_scaled) for model in models.values()])
        
        # Make final predictions
        predictions = meta_clf.predict(preds)
        probabilities = meta_clf.predict_proba(preds)[:, 1] if hasattr(meta_clf, "predict_proba") else None
    
    return predictions, probabilities
