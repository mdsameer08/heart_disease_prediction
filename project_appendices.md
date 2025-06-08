# Appendices - Heart Disease Prediction System

## Appendix A: Data Dictionary

### A.1 Input Features

| Feature | Description | Type | Range/Values | Medical Significance |
|---------|-------------|------|--------------|----------------------|
| `age` | Age of the patient in years | Numeric | 20-100 | Risk increases with age |
| `sex` | Gender of the patient | Categorical | 0 (Female), 1 (Male) | Males have higher risk |
| `cp` | Chest pain type | Categorical | 0 (Typical Angina), 1 (Atypical Angina), 2 (Non-anginal Pain), 3 (Asymptomatic) | Different types indicate varying risk levels |
| `trestbps` | Resting blood pressure in mm Hg | Numeric | 90-220 | Higher values indicate hypertension |
| `chol` | Serum cholesterol in mg/dl | Numeric | 120-600 | Higher values indicate hypercholesterolemia |
| `fbs` | Fasting blood sugar > 120 mg/dl | Binary | 0 (False), 1 (True) | Indicates diabetes risk |
| `restecg` | Resting electrocardiographic results | Categorical | 0 (Normal), 1 (ST-T wave abnormality), 2 (Left ventricular hypertrophy) | Indicates existing heart abnormalities |
| `thalach` | Maximum heart rate achieved | Numeric | 60-220 | Lower max heart rate may indicate heart issues |
| `exang` | Exercise induced angina | Binary | 0 (No), 1 (Yes) | Chest pain during exercise indicates risk |
| `oldpeak` | ST depression induced by exercise relative to rest | Numeric | 0-6.2 | Higher values indicate ischemia |
| `slope` | Slope of the peak exercise ST segment | Categorical | 0 (Upsloping), 1 (Flat), 2 (Downsloping) | Downsloping indicates abnormality |
| `ca` | Number of major vessels colored by fluoroscopy | Numeric | 0-4 | More vessels indicate more severe disease |
| `thal` | Thalassemia (blood disorder) | Categorical | 0 (Normal), 1 (Fixed defect), 2 (Reversible defect) | Reversible defects indicate higher risk |

### A.2 Derived Features

| Feature | Description | Formula | Significance |
|---------|-------------|---------|--------------|
| `age_group` | Age category | Binned from age: [0-40, 40-50, 50-60, 60-70, 70+] | Captures non-linear age effects |
| `trestbps_chol_ratio` | Blood pressure to cholesterol ratio | `trestbps / chol` | Relationship between two risk factors |
| `thalach_age_ratio` | Maximum heart rate to age ratio | `thalach / age` | Indicates cardiovascular fitness |
| `oldpeak_thalach_ratio` | ST depression to max heart rate ratio | `(oldpeak + 1) / thalach` | Normalized ischemia indicator |
| `trestbps_thalach_ratio` | Resting BP to max heart rate ratio | `trestbps / thalach` | Cardiovascular response indicator |
| `trestbps_squared` | Squared resting blood pressure | `trestbps^2` | Captures non-linear BP effects |
| `chol_squared` | Squared cholesterol | `chol^2` | Captures non-linear cholesterol effects |
| `oldpeak_squared` | Squared ST depression | `oldpeak^2` | Emphasizes larger ST depressions |
| `thalach_squared` | Squared maximum heart rate | `thalach^2` | Captures non-linear heart rate effects |
| `has_high_bp` | High blood pressure indicator | `trestbps > 140` | Clinical threshold for hypertension |
| `has_high_chol` | High cholesterol indicator | `chol > 240` | Clinical threshold for hypercholesterolemia |
| `has_tachycardia` | Elevated heart rate indicator | `thalach > 100` | Indicates tachycardia |
| `categorical_interactions` | Interactions between categorical variables | Various combinations | Captures combined effects of risk factors |

### A.3 Target Variable

| Feature | Description | Values | Interpretation |
|---------|-------------|--------|----------------|
| `target` | Presence of heart disease | 0 (No), 1 (Yes) | Binary classification target |

## Appendix B: Model Performance Metrics

### B.1 Individual Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Neural Network | 91.67% | 0.92 | 0.92 | 0.92 | 0.95 |
| Neural Network 2 | 88.33% | 0.89 | 0.88 | 0.88 | 0.94 |
| SVM | 86.67% | 0.87 | 0.87 | 0.87 | 0.93 |
| XGBoost | 85.00% | 0.85 | 0.85 | 0.85 | 0.92 |
| Random Forest | 83.33% | 0.84 | 0.83 | 0.83 | 0.91 |
| Gradient Boosting | 85.00% | 0.85 | 0.85 | 0.85 | 0.92 |
| Voting Ensemble | 90.00% | 0.90 | 0.90 | 0.90 | 0.96 |
| Meta-Ensemble | 93.33% | 0.94 | 0.93 | 0.93 | 0.97 |

*Note: These metrics are based on the test set evaluation after SMOTEENN resampling.*

### B.2 Confusion Matrices

#### B.2.1 Neural Network Confusion Matrix
```
[30  2]
[ 3 25]
```

#### B.2.2 Meta-Ensemble Confusion Matrix
```
[31  1]
[ 3 25]
```

### B.3 Feature Importance

| Feature | Importance Score | Rank |
|---------|------------------|------|
| `thalach_age_ratio` | 0.142 | 1 |
| `cp` | 0.118 | 2 |
| `oldpeak` | 0.103 | 3 |
| `thal` | 0.097 | 4 |
| `ca` | 0.092 | 5 |
| `thalach` | 0.087 | 6 |
| `exang` | 0.076 | 7 |
| `oldpeak_thalach_ratio` | 0.068 | 8 |
| `age` | 0.062 | 9 |
| `sex` | 0.055 | 10 |

*Note: Feature importance scores are averaged across multiple models.*

## Appendix C: Technical Implementation Details

### C.1 Environment Setup

```
Python 3.8+
Flask 1.1.2
NumPy
pandas
scikit-learn
XGBoost
imbalanced-learn
Matplotlib
Seaborn
pickle
```

### C.2 Data Preprocessing Pipeline

1. **Data Loading**
   - Load CSV data
   - Rename target column
   - Check for missing values

2. **Categorical Encoding**
   - Apply LabelEncoder to categorical features
   - Store encoders for deployment

3. **Feature Engineering**
   - Create age groups
   - Generate interaction features
   - Create polynomial features
   - Develop domain-specific features

4. **Feature Selection**
   - Hybrid approach using F-test and mutual information
   - Select top features based on combined ranking

5. **Data Splitting**
   - Stratified train-test split (70-30)
   - Preserve class distribution

6. **Feature Scaling**
   - Apply MinMaxScaler to normalize features
   - Store scaler for deployment

7. **Class Balancing**
   - Apply SMOTEENN to training data
   - Generate synthetic samples for minority class
   - Remove noise with Edited Nearest Neighbors

### C.3 Model Training Configuration

#### C.3.1 Neural Network
```python
MLPClassifier(
    hidden_layer_sizes=(100, 50, 25),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    learning_rate='adaptive',
    max_iter=1000,
    random_state=RANDOM_STATE
)
```

#### C.3.2 SVM
```python
SVC(
    C=10.0,
    kernel='rbf',
    gamma='scale',
    probability=True,
    random_state=RANDOM_STATE
)
```

#### C.3.3 XGBoost
```python
XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE
)
```

#### C.3.4 Voting Ensemble
```python
VotingClassifier(
    estimators=[
        ('mlp', mlp),
        ('svm', svm),
        ('xgb', xgb),
        ('rf', rf),
        ('gb', gb)
    ],
    voting='soft',
    weights=[3, 2, 2, 1, 2]
)
```

#### C.3.5 Meta-Ensemble
```python
# Base models dictionary
models = {
    'Neural Network': mlp,
    'Neural Network 2': mlp2,
    'SVM': svm,
    'XGBoost': xgb,
    'Random Forest': rf,
    'Gradient Boosting': gb,
    'Voting Ensemble': voting_clf
}

# Meta-classifier
meta_clf = SVC(probability=True, random_state=RANDOM_STATE)
```

### C.4 Model Serialization

```python
final_model_data = {
    'models': models,
    'meta_classifier': meta_clf,
    'scaler': scaler,
    'features': top_features,
    'type': 'meta-ensemble'
}

filename = 'heart-disease-prediction-meta-ensemble-model.pkl'
pickle.dump(final_model_data, open(filename, 'wb'))
```

## Appendix D: Web Application Implementation

### D.1 Flask Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Home page with input form |
| `/predict` | POST | Process form data and return prediction |
| `/calculate_features` | POST | AJAX endpoint for derived feature calculation |
| `/low_risk_test` | GET | Run prediction with predefined low-risk data |

### D.2 Frontend Components

1. **Main Page (main.html)**
   - Input form with validation
   - Tabbed interface (Input Data, Derived Features, About the Model)
   - Low Risk Test button
   - Responsive design

2. **Results Page (result.html)**
   - Prediction result with probability
   - Derived features display
   - Risk factor visualization
   - Input data summary

3. **Styling (style.css)**
   - Responsive layout
   - Color-coded risk indicators
   - Interactive tooltips
   - Mobile-friendly design

### D.3 AJAX Feature Calculation

```javascript
// Client-side code for real-time feature calculation
$("#age, #trestbps, #chol, #thalach, #oldpeak").on("change", function() {
    // Collect form data
    var formData = $("#prediction-form").serialize();
    
    // Send AJAX request
    $.ajax({
        url: "/calculate_features",
        type: "POST",
        data: formData,
        success: function(response) {
            // Update derived features display
            $("#age-group").text(response.age_group);
            $("#bp-status").text(response.bp_status);
            $("#chol-status").text(response.chol_status);
            $("#hr-status").text(response.hr_status);
            // Update ratios
            $("#trestbps-chol-ratio").text(response.trestbps_chol_ratio);
            $("#thalach-age-ratio").text(response.thalach_age_ratio);
            $("#oldpeak-thalach-ratio").text(response.oldpeak_thalach_ratio);
            $("#trestbps-thalach-ratio").text(response.trestbps_thalach_ratio);
            
            // Show feature card
            $("#feature-card").show();
        }
    });
});
```

## Appendix E: Multimodal Extension Specifications

### E.1 ECG Data Processing

#### E.1.1 ECG Signal Features
- R-R intervals
- QT interval
- ST segment elevation/depression
- T-wave amplitude and morphology
- Heart rate variability metrics
- Frequency domain features

#### E.1.2 ECG Processing Pipeline
1. Signal filtering (bandpass filter)
2. QRS complex detection
3. Interval measurement
4. Morphology analysis
5. Feature extraction

### E.2 EHR Data Integration

#### E.2.1 Additional EHR Features
- Medical history (prior cardiac events)
- Medication information
- Lab test results (beyond basic lipid panel)
- Family history of heart disease
- Comorbidities (diabetes, hypertension)

#### E.2.2 EHR Processing Pipeline
1. Data extraction from clinical databases
2. Standardization of medical codes
3. Temporal feature creation
4. Missing value imputation
5. Feature normalization

### E.3 Wearable Sensor Data

#### E.3.1 Wearable Sensor Features
- Continuous heart rate monitoring
- Heart rate variability (HRV)
- Activity level and patterns
- Sleep quality metrics
- Stress indicators
- Blood pressure trends (if available)

#### E.3.2 Wearable Data Processing Pipeline
1. Signal cleaning and artifact removal
2. Feature extraction across time windows
3. Pattern recognition
4. Anomaly detection
5. Trend analysis

### E.4 Modality Fusion Approaches

#### E.4.1 Early Fusion
- Concatenate features from all modalities
- Apply feature selection to combined feature set
- Use dimensionality reduction (PCA, t-SNE)
- Train unified models on fused features

#### E.4.2 Late Fusion
- Train separate models for each modality
- Combine predictions using weighted voting
- Use meta-classifier for final prediction
- Calibrate probabilities across modalities

#### E.4.3 Hybrid Fusion
- Combine selected features from each modality
- Use attention mechanisms to weight modalities
- Implement cross-modal feature interactions
- Adaptive weighting based on data quality

## Appendix F: References

### F.1 Medical Guidelines

1. American Heart Association. (2023). Heart Disease and Stroke Statistics.
2. World Health Organization. (2022). Cardiovascular diseases (CVDs) fact sheet.
3. European Society of Cardiology. (2021). Guidelines for the diagnosis and management of cardiovascular disease.
4. American College of Cardiology. (2022). ASCVD Risk Estimator Plus.

### F.2 Machine Learning Resources

1. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
2. XGBoost: A Scalable Tree Boosting System, Chen & Guestrin, 2016.
3. SMOTE: Synthetic Minority Over-sampling Technique, Chawla et al., 2002.
4. Ensemble Methods in Machine Learning, Dietterich, 2000.

### F.3 Related Research

1. Motwani, M., et al. (2017). Machine learning for prediction of all-cause mortality in patients with suspected coronary artery disease. European Heart Journal, 38(7), 500-507.
2. Kwon, J. M., et al. (2019). Deep learning-based risk stratification for mortality of patients with acute myocardial infarction. PLOS ONE, 14(10), e0224502.
3. Rajkomar, A., et al. (2018). Scalable and accurate deep learning with electronic health records. npj Digital Medicine, 1(1), 18.
4. Attia, Z. I., et al. (2019). An artificial intelligence-enabled ECG algorithm for the identification of patients with atrial fibrillation during sinus rhythm. The Lancet, 394(10201), 861-867.

### F.4 Web Development Resources

1. Flask Documentation: https://flask.palletsprojects.com/
2. MDN Web Docs: https://developer.mozilla.org/
3. Web Accessibility Initiative (WAI): https://www.w3.org/WAI/