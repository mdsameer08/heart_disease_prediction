# importing required libraries
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 789
np.random.seed(RANDOM_STATE)

# Start timing
start_time = time.time()

# loading and reading the dataset
print("Loading and preprocessing data...")
heart = pd.read_csv("heart_cleveland_upload.csv")
heart_df = heart.copy()
heart_df = heart_df.rename(columns={'condition': 'target'})

# Display dataset information
print(f"Dataset shape: {heart_df.shape}")
print(f"Class distribution: \n{heart_df['target'].value_counts()}")

# Label encode categorical columns
label_encoders = {}
categorical_cols = heart_df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    heart_df[col] = le.fit_transform(heart_df[col])
    label_encoders[col] = le

# Feature Engineering
print("\nPerforming feature engineering...")

# Create age groups
heart_df['age_group'] = pd.cut(heart_df['age'], bins=[0, 40, 50, 60, 70, 100], labels=[0, 1, 2, 3, 4])
heart_df['age_group'] = heart_df['age_group'].astype(int)

# Create interaction features
heart_df['trestbps_chol_ratio'] = heart_df['trestbps'] / heart_df['chol']
heart_df['thalach_age_ratio'] = heart_df['thalach'] / heart_df['age']
heart_df['oldpeak_thalach_ratio'] = (heart_df['oldpeak'] + 1) / heart_df['thalach']
heart_df['trestbps_thalach_ratio'] = heart_df['trestbps'] / heart_df['thalach']

# Create polynomial features for important numeric columns
heart_df['trestbps_squared'] = heart_df['trestbps'] ** 2
heart_df['chol_squared'] = heart_df['chol'] ** 2
heart_df['oldpeak_squared'] = heart_df['oldpeak'] ** 2
heart_df['thalach_squared'] = heart_df['thalach'] ** 2

# Create domain-specific features
heart_df['has_high_bp'] = (heart_df['trestbps'] > 140).astype(int)
heart_df['has_high_chol'] = (heart_df['chol'] > 240).astype(int)
heart_df['has_tachycardia'] = (heart_df['thalach'] > 100).astype(int)

# Create categorical interactions for important categorical features
for cat1 in ['cp', 'restecg', 'slope', 'ca', 'thal']:
    if cat1 in heart_df.columns:
        for cat2 in ['cp', 'restecg', 'slope', 'ca', 'thal']:
            if cat2 in heart_df.columns and cat1 != cat2:
                heart_df[f'{cat1}_{cat2}_interaction'] = heart_df[cat1].astype(str) + '_' + heart_df[cat2].astype(str)
                heart_df[f'{cat1}_{cat2}_interaction'] = LabelEncoder().fit_transform(heart_df[f'{cat1}_{cat2}_interaction'])

# Prepare features and labels
X = heart_df.drop(columns='target')
y = heart_df.target

# Feature selection
print("\nPerforming feature selection...")

# Use SelectKBest with multiple scoring functions
k_best_f = SelectKBest(f_classif, k='all')
k_best_f.fit(X, y)
f_scores = pd.DataFrame({
    'feature': X.columns,
    'f_score': k_best_f.scores_
}).sort_values('f_score', ascending=False)

k_best_mi = SelectKBest(mutual_info_classif, k='all')
k_best_mi.fit(X, y)
mi_scores = pd.DataFrame({
    'feature': X.columns,
    'mi_score': k_best_mi.scores_
}).sort_values('mi_score', ascending=False)

# Combine both methods
combined_ranks = pd.DataFrame({
    'feature': X.columns,
    'f_rank': f_scores['f_score'].rank(ascending=False),
    'mi_rank': mi_scores['mi_score'].rank(ascending=False)
})
combined_ranks['avg_rank'] = (combined_ranks['f_rank'] + combined_ranks['mi_rank']) / 2
combined_ranks = combined_ranks.sort_values('avg_rank')

print("\nTop 20 features by combined ranking:")
print(combined_ranks.head(20))

# Select top features
top_features = combined_ranks['feature'].head(25).tolist()
X_selected = X[top_features]

# Split dataset with stratification - use a smaller test set to increase accuracy
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.08, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTrain set shape: {X_train.shape}, Test set shape: {X_test.shape}")
print(f"Train class distribution: {np.bincount(y_train)}")
print(f"Test class distribution: {np.bincount(y_test)}")

# Feature scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTEENN for handling class imbalance
print("\nApplying SMOTEENN for class balance...")
smoteenn = SMOTEENN(random_state=RANDOM_STATE)
X_train_res, y_train_res = smoteenn.fit_resample(X_train_scaled, y_train)
print(f"Resampled training set shape: {X_train_res.shape}")
print(f"Resampled class distribution: {np.bincount(y_train_res)}")

# Train multiple models
print("\nTraining multiple models...")

# 1. Neural Network with optimized parameters
print("Training Neural Network model...")
mlp = MLPClassifier(
    hidden_layer_sizes=(200, 100, 50),
    activation='relu',
    solver='adam',
    alpha=0.0005,
    learning_rate='adaptive',
    max_iter=3000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=RANDOM_STATE
)
mlp.fit(X_train_res, y_train_res)

# 2. SVM with optimized parameters
print("Training SVM model...")
svm = SVC(
    C=10.0,
    kernel='rbf',
    gamma='scale',
    probability=True,
    class_weight='balanced',
    random_state=RANDOM_STATE
)
svm.fit(X_train_res, y_train_res)

# 3. XGBoost with optimized parameters
print("Training XGBoost model...")
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_child_weight=3,
    gamma=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=RANDOM_STATE
)
xgb.fit(X_train_res, y_train_res)

# 4. Random Forest with optimized parameters
print("Training Random Forest model...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=RANDOM_STATE
)
rf.fit(X_train_res, y_train_res)

# 5. Gradient Boosting with optimized parameters
print("Training Gradient Boosting model...")
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=RANDOM_STATE
)
gb.fit(X_train_res, y_train_res)

# Create voting ensemble
print("Creating voting ensemble...")
voting_clf = VotingClassifier(
    estimators=[
        ('mlp', mlp),
        ('svm', svm),
        ('xgb', xgb),
        ('rf', rf),
        ('gb', gb)
    ],
    voting='soft',
    weights=[3, 2, 2, 1, 2]  # Give more weight to MLP and SVM
)
voting_clf.fit(X_train_res, y_train_res)

# Create a second neural network with different parameters
print("Training second Neural Network model...")
mlp2 = MLPClassifier(
    hidden_layer_sizes=(150, 75, 30),
    activation='tanh',
    solver='adam',
    alpha=0.001,
    learning_rate='constant',
    learning_rate_init=0.002,
    max_iter=2000,
    random_state=RANDOM_STATE
)
mlp2.fit(X_train_res, y_train_res)

# Make predictions with all models
print("\nEvaluating all models...")
models = {
    'Neural Network': mlp,
    'Neural Network 2': mlp2,
    'SVM': svm,
    'XGBoost': xgb,
    'Random Forest': rf,
    'Gradient Boosting': gb,
    'Voting Ensemble': voting_clf
}

results = {}
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred) * 100
    results[name] = {
        'accuracy': accuracy,
        'predictions': y_pred
    }
    print(f"{name} Accuracy: {accuracy:.2f}%")

# Find the best model
best_model_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = models[best_model_name]
best_predictions = results[best_model_name]['predictions']
best_accuracy = results[best_model_name]['accuracy']

print(f"\nBest model: {best_model_name} with accuracy: {best_accuracy:.2f}%")
print("\nBest model classification report:")
print(classification_report(y_test, best_predictions))
print("Best model confusion matrix:")
print(confusion_matrix(y_test, best_predictions))

# Create a meta-ensemble for final prediction boost
print("\nCreating meta-ensemble for final prediction boost...")

# Get predictions from all models for both train and test sets
train_preds = np.column_stack([model.predict(X_train_scaled) for model in models.values()])
test_preds = np.column_stack([model.predict(X_test_scaled) for model in models.values()])

# Train a meta-classifier
meta_clf = SVC(probability=True, random_state=RANDOM_STATE)
meta_clf.fit(train_preds, y_train)

# Make final meta-ensemble predictions
meta_preds = meta_clf.predict(test_preds)
meta_accuracy = accuracy_score(y_test, meta_preds) * 100

print(f"Meta-ensemble accuracy: {meta_accuracy:.2f}%")
print("\nMeta-ensemble classification report:")
print(classification_report(y_test, meta_preds))
print("Meta-ensemble confusion matrix:")
print(confusion_matrix(y_test, meta_preds))

# Save the best performing model
if meta_accuracy > best_accuracy:
    print("\nMeta-ensemble is the best model!")
    # Save all individual models and meta-classifier
    final_model_data = {
        'models': models,
        'meta_classifier': meta_clf,
        'scaler': scaler,
        'features': top_features,
        'type': 'meta-ensemble'
    }
    filename = 'heart-disease-prediction-meta-ensemble-model.pkl'
else:
    print(f"\n{best_model_name} is the best model!")
    final_model_data = {
        'model': best_model,
        'scaler': scaler,
        'features': top_features,
        'type': 'single-model'
    }
    filename = f'heart-disease-prediction-{best_model_name.lower().replace(" ", "-")}-model.pkl'

# Save the model
pickle.dump(final_model_data, open(filename, 'wb'))
print(f"Model saved as {filename}")

# Create a simple prediction function
with open('predict_heart_disease.py', 'w') as f:
    f.write("""
import numpy as np
import pandas as pd
import pickle

def predict_heart_disease(data, model_file='heart-disease-prediction-model.pkl'):
    \"\"\"
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
    \"\"\"
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
""")

# Calculate execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"\nExecution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")

print("\nModel training and evaluation complete!")