# Importing essential libraries
from flask import Flask, render_template, request, jsonify, redirect, url_for
import pickle
import numpy as np
import pandas as pd
import json

# Load the improved Neural Network model
filename = 'heart-disease-prediction-neural-network-model.pkl'
model_data = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

# Low risk sample data
low_risk_data = {
    'age': 22,
    'sex': 0,  # Female
    'cp': 1,   # Atypical Angina
    'trestbps': 100,
    'chol': 120,
    'fbs': 0,  # Less than 120 mg/dl
    'restecg': 0,  # Normal
    'thalach': 210,
    'exang': 0,  # No
    'oldpeak': 0.0,
    'slope': 2,  # Upsloping
    'ca': 0,
    'thal': 0  # Normal
}

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/low_risk_test')
def low_risk_test():
    """
    Automatically run a prediction with low-risk data
    """
    # Create a DataFrame with the low-risk input data
    input_data = pd.DataFrame({
        'age': [low_risk_data['age']],
        'sex': [low_risk_data['sex']],
        'cp': [low_risk_data['cp']],
        'trestbps': [low_risk_data['trestbps']],
        'chol': [low_risk_data['chol']],
        'fbs': [low_risk_data['fbs']],
        'restecg': [low_risk_data['restecg']],
        'thalach': [low_risk_data['thalach']],
        'exang': [low_risk_data['exang']],
        'oldpeak': [low_risk_data['oldpeak']],
        'slope': [low_risk_data['slope']],
        'ca': [low_risk_data['ca']],
        'thal': [low_risk_data['thal']]
    })
    
    # Feature engineering for the input data
    # Create age groups
    input_data['age_group'] = pd.cut(input_data['age'], bins=[0, 40, 50, 60, 70, 100], labels=[0, 1, 2, 3, 4])
    input_data['age_group'] = input_data['age_group'].astype(int)
    
    # Create interaction features
    input_data['trestbps_chol_ratio'] = input_data['trestbps'] / input_data['chol']
    input_data['thalach_age_ratio'] = input_data['thalach'] / input_data['age']
    input_data['oldpeak_thalach_ratio'] = (input_data['oldpeak'] + 1) / input_data['thalach']
    input_data['trestbps_thalach_ratio'] = input_data['trestbps'] / input_data['thalach']
    
    # Create polynomial features
    input_data['trestbps_squared'] = input_data['trestbps'] ** 2
    input_data['chol_squared'] = input_data['chol'] ** 2
    input_data['oldpeak_squared'] = input_data['oldpeak'] ** 2
    input_data['thalach_squared'] = input_data['thalach'] ** 2
    
    # Create domain-specific features
    input_data['has_high_bp'] = (input_data['trestbps'] > 140).astype(int)
    input_data['has_high_chol'] = (input_data['chol'] > 240).astype(int)
    input_data['has_tachycardia'] = (input_data['thalach'] > 100).astype(int)
    
    # Create categorical interactions
    for cat1 in ['cp', 'restecg', 'slope', 'ca', 'thal']:
        for cat2 in ['cp', 'restecg', 'slope', 'ca', 'thal']:
            if cat1 != cat2:
                col_name = f'{cat1}_{cat2}_interaction'
                input_data[col_name] = input_data[cat1].astype(str) + '_' + input_data[cat2].astype(str)
                # Since we don't have the original LabelEncoder, we'll use a simple hash function
                input_data[col_name] = input_data[col_name].apply(lambda x: hash(x) % 10)
    
    # Extract model components
    if model_data['type'] == 'single-model':
        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['features']
        
        # Select the required features
        X = input_data[features]
        
        # Scale the features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        my_prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1] * 100  # Get probability as percentage
    else:  # meta-ensemble
        models = model_data['models']
        meta_clf = model_data['meta_classifier']
        scaler = model_data['scaler']
        features = model_data['features']
        
        # Select the required features
        X = input_data[features]
        
        # Scale the features
        X_scaled = scaler.transform(X)
        
        # Get predictions from all models
        preds = np.column_stack([model.predict(X_scaled) for model in models.values()])
        
        # Make final prediction
        my_prediction = meta_clf.predict(preds)[0]
        probability = meta_clf.predict_proba(preds)[0][1] * 100  # Get probability as percentage
    
    # Create human-readable derived features for display
    # Age group
    age = low_risk_data['age']
    trestbps = low_risk_data['trestbps']
    chol = low_risk_data['chol']
    thalach = low_risk_data['thalach']
    oldpeak = low_risk_data['oldpeak']
    
    # Age group
    if age < 40:
        age_group = "Under 40"
    elif age < 50:
        age_group = "40-49"
    elif age < 60:
        age_group = "50-59"
    elif age < 70:
        age_group = "60-69"
    else:
        age_group = "70+"
    
    # Blood pressure status
    if trestbps > 140:
        bp_status = "High (Hypertension)"
    elif trestbps > 120:
        bp_status = "Elevated"
    else:
        bp_status = "Normal"
    
    # Cholesterol status
    if chol > 240:
        chol_status = "High"
    elif chol > 200:
        chol_status = "Borderline High"
    else:
        chol_status = "Normal"
    
    # Heart rate status
    if thalach > 100:
        hr_status = "Tachycardia (Elevated)"
    elif thalach < 60:
        hr_status = "Bradycardia (Low)"
    else:
        hr_status = "Normal"
    
    # Calculate ratios for display
    trestbps_chol_ratio = round(trestbps / chol, 3)
    thalach_age_ratio = round(thalach / age, 3)
    oldpeak_thalach_ratio = round((oldpeak + 1) / thalach, 3)
    trestbps_thalach_ratio = round(trestbps / thalach, 3)
    
    # Get top 5 most important features used by the model
    if model_data['type'] == 'single-model':
        top_features = model_data['features'][:5]
    else:
        top_features = model_data['features'][:5]
    
    # Create a dictionary of derived features
    derived_features = {
        'age_group': age_group,
        'bp_status': bp_status,
        'chol_status': chol_status,
        'hr_status': hr_status,
        'trestbps_chol_ratio': trestbps_chol_ratio,
        'thalach_age_ratio': thalach_age_ratio,
        'oldpeak_thalach_ratio': oldpeak_thalach_ratio,
        'trestbps_thalach_ratio': trestbps_thalach_ratio
    }
    
    return render_template(
        'result.html', 
        prediction=my_prediction, 
        probability=round(probability, 2),
        derived_features=derived_features,
        top_features=top_features,
        input_data=low_risk_data
    )

@app.route('/calculate_features', methods=['POST'])
def calculate_features():
    # Get form data
    age = int(request.form['age'])
    sex = request.form.get('sex')
    cp = request.form.get('cp')
    trestbps = int(request.form['trestbps'])
    chol = int(request.form['chol'])
    fbs = request.form.get('fbs')
    restecg = int(request.form['restecg'])
    thalach = int(request.form['thalach'])
    exang = request.form.get('exang')
    oldpeak = float(request.form['oldpeak'])
    slope = request.form.get('slope')
    ca = int(request.form['ca'])
    thal = request.form.get('thal')
    
    # Create derived features
    # Age group
    if age < 40:
        age_group = "Under 40"
    elif age < 50:
        age_group = "40-49"
    elif age < 60:
        age_group = "50-59"
    elif age < 70:
        age_group = "60-69"
    else:
        age_group = "70+"
    
    # Blood pressure status
    if trestbps > 140:
        bp_status = "High (Hypertension)"
    elif trestbps > 120:
        bp_status = "Elevated"
    else:
        bp_status = "Normal"
    
    # Cholesterol status
    if chol > 240:
        chol_status = "High"
    elif chol > 200:
        chol_status = "Borderline High"
    else:
        chol_status = "Normal"
    
    # Heart rate status
    if thalach > 100:
        hr_status = "Tachycardia (Elevated)"
    elif thalach < 60:
        hr_status = "Bradycardia (Low)"
    else:
        hr_status = "Normal"
    
    # Calculate ratios
    trestbps_chol_ratio = round(trestbps / chol, 3)
    thalach_age_ratio = round(thalach / age, 3)
    oldpeak_thalach_ratio = round((oldpeak + 1) / thalach, 3)
    trestbps_thalach_ratio = round(trestbps / thalach, 3)
    
    # Return the calculated features
    return jsonify({
        'age_group': age_group,
        'bp_status': bp_status,
        'chol_status': chol_status,
        'hr_status': hr_status,
        'trestbps_chol_ratio': trestbps_chol_ratio,
        'thalach_age_ratio': thalach_age_ratio,
        'oldpeak_thalach_ratio': oldpeak_thalach_ratio,
        'trestbps_thalach_ratio': trestbps_thalach_ratio
    })

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        age = int(request.form['age'])
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = request.form.get('fbs')
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = request.form.get('exang')
        oldpeak = float(request.form['oldpeak'])
        slope = request.form.get('slope')
        ca = int(request.form['ca'])
        thal = request.form.get('thal')
        
        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'cp': [cp],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs],
            'restecg': [restecg],
            'thalach': [thalach],
            'exang': [exang],
            'oldpeak': [oldpeak],
            'slope': [slope],
            'ca': [ca],
            'thal': [thal]
        })
        
        # Feature engineering for the input data
        # Create age groups
        input_data['age_group'] = pd.cut(input_data['age'], bins=[0, 40, 50, 60, 70, 100], labels=[0, 1, 2, 3, 4])
        input_data['age_group'] = input_data['age_group'].astype(int)
        
        # Create interaction features
        input_data['trestbps_chol_ratio'] = input_data['trestbps'] / input_data['chol']
        input_data['thalach_age_ratio'] = input_data['thalach'] / input_data['age']
        input_data['oldpeak_thalach_ratio'] = (input_data['oldpeak'] + 1) / input_data['thalach']
        input_data['trestbps_thalach_ratio'] = input_data['trestbps'] / input_data['thalach']
        
        # Create polynomial features
        input_data['trestbps_squared'] = input_data['trestbps'] ** 2
        input_data['chol_squared'] = input_data['chol'] ** 2
        input_data['oldpeak_squared'] = input_data['oldpeak'] ** 2
        input_data['thalach_squared'] = input_data['thalach'] ** 2
        
        # Create domain-specific features
        input_data['has_high_bp'] = (input_data['trestbps'] > 140).astype(int)
        input_data['has_high_chol'] = (input_data['chol'] > 240).astype(int)
        input_data['has_tachycardia'] = (input_data['thalach'] > 100).astype(int)
        
        # Create categorical interactions
        for cat1 in ['cp', 'restecg', 'slope', 'ca', 'thal']:
            for cat2 in ['cp', 'restecg', 'slope', 'ca', 'thal']:
                if cat1 != cat2:
                    col_name = f'{cat1}_{cat2}_interaction'
                    input_data[col_name] = input_data[cat1].astype(str) + '_' + input_data[cat2].astype(str)
                    # Since we don't have the original LabelEncoder, we'll use a simple hash function
                    input_data[col_name] = input_data[col_name].apply(lambda x: hash(x) % 10)
        
        # Extract model components
        if model_data['type'] == 'single-model':
            model = model_data['model']
            scaler = model_data['scaler']
            features = model_data['features']
            
            # Select the required features
            X = input_data[features]
            
            # Scale the features
            X_scaled = scaler.transform(X)
            
            # Make prediction
            my_prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0][1] * 100  # Get probability as percentage
        else:  # meta-ensemble
            models = model_data['models']
            meta_clf = model_data['meta_classifier']
            scaler = model_data['scaler']
            features = model_data['features']
            
            # Select the required features
            X = input_data[features]
            
            # Scale the features
            X_scaled = scaler.transform(X)
            
            # Get predictions from all models
            preds = np.column_stack([model.predict(X_scaled) for model in models.values()])
            
            # Make final prediction
            my_prediction = meta_clf.predict(preds)[0]
            probability = meta_clf.predict_proba(preds)[0][1] * 100  # Get probability as percentage
        
        # Create human-readable derived features for display
        # Age group
        if age < 40:
            age_group = "Under 40"
        elif age < 50:
            age_group = "40-49"
        elif age < 60:
            age_group = "50-59"
        elif age < 70:
            age_group = "60-69"
        else:
            age_group = "70+"
        
        # Blood pressure status
        if trestbps > 140:
            bp_status = "High (Hypertension)"
        elif trestbps > 120:
            bp_status = "Elevated"
        else:
            bp_status = "Normal"
        
        # Cholesterol status
        if chol > 240:
            chol_status = "High"
        elif chol > 200:
            chol_status = "Borderline High"
        else:
            chol_status = "Normal"
        
        # Heart rate status
        if thalach > 100:
            hr_status = "Tachycardia (Elevated)"
        elif thalach < 60:
            hr_status = "Bradycardia (Low)"
        else:
            hr_status = "Normal"
        
        # Calculate ratios for display
        trestbps_chol_ratio = round(trestbps / chol, 3)
        thalach_age_ratio = round(thalach / age, 3)
        oldpeak_thalach_ratio = round((oldpeak + 1) / thalach, 3)
        trestbps_thalach_ratio = round(trestbps / thalach, 3)
        
        # Get top 5 most important features used by the model
        if model_data['type'] == 'single-model':
            top_features = model_data['features'][:5]
        else:
            top_features = model_data['features'][:5]
        
        # Create a dictionary of derived features
        derived_features = {
            'age_group': age_group,
            'bp_status': bp_status,
            'chol_status': chol_status,
            'hr_status': hr_status,
            'trestbps_chol_ratio': trestbps_chol_ratio,
            'thalach_age_ratio': thalach_age_ratio,
            'oldpeak_thalach_ratio': oldpeak_thalach_ratio,
            'trestbps_thalach_ratio': trestbps_thalach_ratio
        }
        
        return render_template(
            'result.html', 
            prediction=my_prediction, 
            probability=round(probability, 2),
            derived_features=derived_features,
            top_features=top_features
        )
        
        

if __name__ == '__main__':
	app.run(debug=True)

