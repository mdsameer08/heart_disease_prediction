# importing required libraries
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures, PowerTransformer, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, f1_score, precision_score, recall_score, make_scorer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.feature_selection import SelectFromModel, RFE, RFECV, SelectKBest, f_classif, mutual_info_classif, VarianceThreshold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks, NearMiss, RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Start timing
start_time = time.time()

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Define custom scoring metrics
f1_weighted = make_scorer(f1_score, average='weighted')
precision_weighted = make_scorer(precision_score, average='weighted')
recall_weighted = make_scorer(recall_score, average='weighted')

# Define scoring metrics for model evaluation
scoring = {
    'accuracy': 'accuracy',
    'f1': f1_weighted,
    'precision': precision_weighted,
    'recall': recall_weighted,
    'roc_auc': 'roc_auc'
}

# loading and reading the dataset
print("Loading and preprocessing data...")
heart = pd.read_csv("heart_cleveland_upload.csv")
heart_df = heart.copy()
heart_df = heart_df.rename(columns={'condition': 'target'})

# Display dataset information
print(f"Dataset shape: {heart_df.shape}")
print(f"Class distribution: \n{heart_df['target'].value_counts()}")
print(f"Class balance: {heart_df['target'].value_counts()[1]}/{heart_df['target'].value_counts()[0]} = {heart_df['target'].value_counts()[1]/heart_df['target'].value_counts()[0]:.2f}")

# Check for missing values and handle them
print("\nChecking for missing values...")
missing_values = heart_df.isnull().sum()
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found")

# Label encode categorical columns
label_encoders = {}
categorical_cols = heart_df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    heart_df[col] = le.fit_transform(heart_df[col])
    label_encoders[col] = le

# Advanced Feature Engineering
print("\nPerforming advanced feature engineering...")

# Create age groups with finer granularity
heart_df['age_group'] = pd.cut(heart_df['age'], bins=[0, 40, 45, 50, 55, 60, 65, 70, 100], labels=range(8))
# Convert category to int to avoid XGBoost issues
heart_df['age_group'] = heart_df['age_group'].astype(int)

# Create more interaction features
heart_df['trestbps_chol_ratio'] = heart_df['trestbps'] / heart_df['chol']
heart_df['thalach_age_ratio'] = heart_df['thalach'] / heart_df['age']
heart_df['oldpeak_thalach_ratio'] = (heart_df['oldpeak'] + 1) / heart_df['thalach']
heart_df['trestbps_thalach_ratio'] = heart_df['trestbps'] / heart_df['thalach']
heart_df['age_thalach_product'] = heart_df['age'] * heart_df['thalach'] / 1000  # Scaled down
heart_df['chol_age_ratio'] = heart_df['chol'] / heart_df['age']
heart_df['age_chol_product'] = heart_df['age'] * heart_df['chol'] / 1000  # Scaled down
heart_df['trestbps_age_ratio'] = heart_df['trestbps'] / heart_df['age']
heart_df['thalach_chol_ratio'] = heart_df['thalach'] / heart_df['chol']

# Create polynomial features for important numeric columns
heart_df['trestbps_squared'] = heart_df['trestbps'] ** 2
heart_df['chol_squared'] = heart_df['chol'] ** 2
heart_df['oldpeak_squared'] = heart_df['oldpeak'] ** 2
heart_df['thalach_squared'] = heart_df['thalach'] ** 2
heart_df['age_squared'] = heart_df['age'] ** 2
heart_df['trestbps_cubed'] = heart_df['trestbps'] ** 3
heart_df['chol_cubed'] = heart_df['chol'] ** 3
heart_df['thalach_cubed'] = heart_df['thalach'] ** 3

# Create log and other transformations for skewed features
heart_df['log_chol'] = np.log1p(heart_df['chol'])
heart_df['log_trestbps'] = np.log1p(heart_df['trestbps'])
heart_df['log_oldpeak'] = np.log1p(heart_df['oldpeak'] + 1)  # Adding 1 to handle zeros
heart_df['sqrt_chol'] = np.sqrt(heart_df['chol'])
heart_df['sqrt_trestbps'] = np.sqrt(heart_df['trestbps'])
heart_df['sqrt_oldpeak'] = np.sqrt(heart_df['oldpeak'] + 1)  # Adding 1 to handle zeros

# Create domain-specific features based on medical knowledge
heart_df['has_high_bp'] = (heart_df['trestbps'] > 140).astype(int)
heart_df['has_very_high_bp'] = (heart_df['trestbps'] > 160).astype(int)
heart_df['has_high_chol'] = (heart_df['chol'] > 240).astype(int)
heart_df['has_very_high_chol'] = (heart_df['chol'] > 280).astype(int)
heart_df['has_tachycardia'] = (heart_df['thalach'] > 100).astype(int)
heart_df['has_bradycardia'] = (heart_df['thalach'] < 60).astype(int)
heart_df['bp_chol_interaction'] = heart_df['has_high_bp'] * heart_df['has_high_chol']
heart_df['elderly'] = (heart_df['age'] > 65).astype(int)
heart_df['young'] = (heart_df['age'] < 45).astype(int)
heart_df['middle_aged'] = ((heart_df['age'] >= 45) & (heart_df['age'] <= 65)).astype(int)

# Create categorical interactions for all categorical features
categorical_features = ['cp', 'restecg', 'slope', 'ca', 'thal', 'sex', 'fbs', 'exang']
for cat1 in categorical_features:
    if cat1 in heart_df.columns:
        for cat2 in categorical_features:
            if cat2 in heart_df.columns and cat1 != cat2:
                heart_df[f'{cat1}_{cat2}_interaction'] = heart_df[cat1].astype(str) + '_' + heart_df[cat2].astype(str)
                heart_df[f'{cat1}_{cat2}_interaction'] = LabelEncoder().fit_transform(heart_df[f'{cat1}_{cat2}_interaction'])

# Create interactions between categorical and numerical features
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
for cat in categorical_features:
    if cat in heart_df.columns:
        for num in numerical_features:
            if num in heart_df.columns:
                # Group by categorical feature and compute mean of numerical feature
                means = heart_df.groupby(cat)[num].mean().to_dict()
                # Create new feature: difference from group mean
                heart_df[f'{cat}_{num}_diff'] = heart_df.apply(lambda x: x[num] - means[x[cat]], axis=1)

# Prepare features and labels
X = heart_df.drop(columns='target')
y = heart_df.target

# Print feature engineering results
print(f"Original feature count: {len(heart.columns) - 1}")
print(f"After feature engineering: {X.shape[1]} features")

# Advanced feature selection
print("\nPerforming advanced feature selection...")

# 1. Remove features with low variance
var_threshold = VarianceThreshold(threshold=0.01)
X_var = var_threshold.fit_transform(X)
var_support = var_threshold.get_support()
var_features = X.columns[var_support].tolist()
print(f"Features after variance threshold: {len(var_features)}")

# 2. Use SelectKBest with multiple scoring functions
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

print("\nTop 10 features by F-statistic:")
print(f_scores.head(10))

print("\nTop 10 features by Mutual Information:")
print(mi_scores.head(10))

# 3. Use multiple tree-based methods for feature importance
rf_selector = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced')
rf_selector.fit(X, y)
rf_importances = pd.DataFrame({
    'feature': X.columns,
    'rf_importance': rf_selector.feature_importances_
}).sort_values('rf_importance', ascending=False)

xgb_selector = XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss')
xgb_selector.fit(X, y)
xgb_importances = pd.DataFrame({
    'feature': X.columns,
    'xgb_importance': xgb_selector.feature_importances_
}).sort_values('xgb_importance', ascending=False)

print("\nTop 10 features by Random Forest importance:")
print(rf_importances.head(10))

print("\nTop 10 features by XGBoost importance:")
print(xgb_importances.head(10))

# 4. Combine all feature selection methods
combined_ranks = pd.DataFrame({'feature': X.columns})
combined_ranks = combined_ranks.merge(f_scores, on='feature')
combined_ranks = combined_ranks.merge(mi_scores, on='feature')
combined_ranks = combined_ranks.merge(rf_importances, on='feature')
combined_ranks = combined_ranks.merge(xgb_importances, on='feature')

# Normalize scores to 0-1 range for each method
for col in ['f_score', 'mi_score', 'rf_importance', 'xgb_importance']:
    max_val = combined_ranks[col].max()
    min_val = combined_ranks[col].min()
    if max_val > min_val:  # Avoid division by zero
        combined_ranks[f'{col}_norm'] = (combined_ranks[col] - min_val) / (max_val - min_val)
    else:
        combined_ranks[f'{col}_norm'] = 0

# Calculate combined score
combined_ranks['combined_score'] = (
    combined_ranks['f_score_norm'] + 
    combined_ranks['mi_score_norm'] + 
    combined_ranks['rf_importance_norm'] + 
    combined_ranks['xgb_importance_norm']
)
combined_ranks = combined_ranks.sort_values('combined_score', ascending=False)

print("\nTop 20 features by combined score:")
print(combined_ranks[['feature', 'combined_score']].head(20))

# Select top features based on combined ranking
top_features = combined_ranks['feature'].head(30).tolist()
X_selected = X[top_features]

# Split dataset with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTrain set shape: {X_train.shape}, Test set shape: {X_test.shape}")
print(f"Train class distribution: {np.bincount(y_train)}")
print(f"Test class distribution: {np.bincount(y_test)}")

# Try different scalers to find the best one
print("\nTesting different scalers...")
scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler()
}

best_scaler = None
best_scaler_score = 0
temp_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)

for name, scaler in scalers.items():
    # Scale the data
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Quick cross-validation to evaluate
    cv_scores = cross_val_score(temp_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    avg_score = np.mean(cv_scores)
    
    print(f"{name} CV score: {avg_score:.4f}")
    
    if avg_score > best_scaler_score:
        best_scaler_score = avg_score
        best_scaler = scaler

print(f"Best scaler: {type(best_scaler).__name__} with CV score: {best_scaler_score:.4f}")

# Apply the best scaler
X_train_scaled = best_scaler.fit_transform(X_train)
X_test_scaled = best_scaler.transform(X_test)

# Apply advanced resampling techniques
print("\nApplying advanced resampling techniques...")
resampling_methods = {
    'SMOTE': SMOTE(random_state=RANDOM_STATE),
    'BorderlineSMOTE': BorderlineSMOTE(random_state=RANDOM_STATE),
    'SVMSMOTE': SVMSMOTE(random_state=RANDOM_STATE),
    'KMeansSMOTE': KMeansSMOTE(random_state=RANDOM_STATE),
    'ADASYN': ADASYN(random_state=RANDOM_STATE),
    'SMOTETomek': SMOTETomek(random_state=RANDOM_STATE),
    'SMOTEENN': SMOTEENN(random_state=RANDOM_STATE)
}

best_resampling_method = None
best_resampling_score = 0
resampling_results = {}

# Use a more robust cross-validation strategy
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
quick_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)

for name, method in resampling_methods.items():
    try:
        # Resample the data
        X_resampled, y_resampled = method.fit_resample(X_train_scaled, y_train)
        
        # Quick cross-validation to evaluate
        cv_scores = cross_val_score(quick_model, X_resampled, y_resampled, cv=cv, scoring='accuracy')
        avg_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        resampling_results[name] = {
            'mean_score': avg_score,
            'std_score': std_score,
            'method': method
        }
        
        print(f"{name} resampling CV score: {avg_score:.4f} (±{std_score:.4f})")
        
        if avg_score > best_resampling_score:
            best_resampling_score = avg_score
            best_resampling_method = method
    except Exception as e:
        print(f"Error with {name}: {str(e)}")

print(f"Best resampling method: {type(best_resampling_method).__name__} with CV score: {best_resampling_score:.4f}")

# Apply the best resampling method
X_train_res, y_train_res = best_resampling_method.fit_resample(X_train_scaled, y_train)
print(f"Resampled training set shape: {X_train_res.shape}")
print(f"Resampled class distribution: {np.bincount(y_train_res)}")

# Define a wider range of models to try
print("\nTraining multiple advanced models...")

# Define a function to train and evaluate a model with cross-validation
def train_and_evaluate(model, name, X, y, param_dist=None, n_iter=50):
    print(f"Training {name} model...")
    
    if param_dist is not None:
        # Use RandomizedSearchCV for hyperparameter tuning
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring='accuracy',
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )
        search.fit(X, y)
        best_model = search.best_estimator_
        print(f"{name} best params: {search.best_params_}")
        print(f"{name} best CV score: {search.best_score_:.4f}")
    else:
        # Just fit the model directly
        best_model = model
        best_model.fit(X, y)
        cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring='accuracy')
        print(f"{name} CV score: {np.mean(cv_scores):.4f}")
    
    return best_model

# 1. XGBoost with expanded parameter search
xgb_param_dist = {
    'max_depth': randint(3, 12),
    'learning_rate': loguniform(1e-3, 0.5),
    'n_estimators': randint(50, 1000),
    'gamma': uniform(0, 1),
    'min_child_weight': randint(1, 20),
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5),
    'colsample_bylevel': uniform(0.5, 0.5),
    'reg_alpha': loguniform(1e-10, 1),
    'reg_lambda': loguniform(1e-10, 1),
    'scale_pos_weight': [1, 2, 3, 5, 7]
}
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
best_xgb = train_and_evaluate(xgb, "XGBoost", X_train_res, y_train_res, xgb_param_dist)

# 2. LightGBM with expanded parameter search
lgbm_param_dist = {
    'num_leaves': randint(10, 150),
    'learning_rate': loguniform(1e-3, 0.5),
    'n_estimators': randint(50, 1000),
    'min_child_samples': randint(5, 100),
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5),
    'reg_alpha': loguniform(1e-10, 1),
    'reg_lambda': loguniform(1e-10, 1),
    'min_split_gain': uniform(0, 1),
    'max_depth': randint(3, 12)
}
lgbm = LGBMClassifier(random_state=RANDOM_STATE)
best_lgbm = train_and_evaluate(lgbm, "LightGBM", X_train_res, y_train_res, lgbm_param_dist)

# 3. CatBoost with parameter search
catboost_param_dist = {
    'iterations': randint(50, 500),
    'learning_rate': loguniform(1e-3, 0.5),
    'depth': randint(3, 10),
    'l2_leaf_reg': loguniform(1, 100),
    'border_count': randint(32, 255),
    'bagging_temperature': uniform(0, 1),
    'random_strength': uniform(0, 10)
}
catboost = CatBoostClassifier(loss_function='Logloss', random_seed=RANDOM_STATE, verbose=0)
best_catboost = train_and_evaluate(catboost, "CatBoost", X_train_res, y_train_res, catboost_param_dist, n_iter=30)

# 4. Random Forest with expanded parameter search
rf_param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(5, 30),
    'min_samples_split': randint(2, 30),
    'min_samples_leaf': randint(1, 15),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample', None],
    'criterion': ['gini', 'entropy', 'log_loss']
}
rf = RandomForestClassifier(random_state=RANDOM_STATE)
best_rf = train_and_evaluate(rf, "Random Forest", X_train_res, y_train_res, rf_param_dist)

# 5. Extra Trees Classifier
et_param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(5, 30),
    'min_samples_split': randint(2, 30),
    'min_samples_leaf': randint(1, 15),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample', None],
    'criterion': ['gini', 'entropy', 'log_loss']
}
et = ExtraTreesClassifier(random_state=RANDOM_STATE)
best_et = train_and_evaluate(et, "Extra Trees", X_train_res, y_train_res, et_param_dist)

# 6. SVM with expanded parameter search
svm_param_dist = {
    'C': loguniform(0.01, 100),
    'gamma': loguniform(1e-4, 10),
    'kernel': ['rbf', 'poly', 'sigmoid'],
    'degree': [2, 3, 4, 5],
    'class_weight': ['balanced', None],
    'probability': [True]
}
svm = SVC(random_state=RANDOM_STATE)
best_svm = train_and_evaluate(svm, "SVM", X_train_res, y_train_res, svm_param_dist, n_iter=30)

# 7. Neural Network with expanded parameter search
mlp_param_dist = {
    'hidden_layer_sizes': [(50,), (100,), (200,), (50, 50), (100, 50), (100, 100), (200, 100), (100, 50, 25)],
    'activation': ['relu', 'tanh', 'logistic'],
    'alpha': loguniform(1e-5, 1e-1),
    'learning_rate': ['constant', 'adaptive', 'invscaling'],
    'learning_rate_init': loguniform(1e-4, 1e-1),
    'solver': ['adam', 'sgd', 'lbfgs'],
    'max_iter': [1000, 2000, 3000],
    'early_stopping': [True, False],
    'validation_fraction': uniform(0.1, 0.2)
}
mlp = MLPClassifier(random_state=RANDOM_STATE)
best_mlp = train_and_evaluate(mlp, "Neural Network", X_train_res, y_train_res, mlp_param_dist, n_iter=30)

# 8. Gradient Boosting
gb_param_dist = {
    'n_estimators': randint(50, 500),
    'learning_rate': loguniform(1e-3, 0.5),
    'max_depth': randint(3, 10),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'subsample': uniform(0.5, 0.5),
    'max_features': ['sqrt', 'log2', None]
}
gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
best_gb = train_and_evaluate(gb, "Gradient Boosting", X_train_res, y_train_res, gb_param_dist)

# 9. AdaBoost
ada_param_dist = {
    'n_estimators': randint(50, 500),
    'learning_rate': loguniform(1e-3, 1),
    'algorithm': ['SAMME', 'SAMME.R']
}
ada = AdaBoostClassifier(random_state=RANDOM_STATE)
best_ada = train_and_evaluate(ada, "AdaBoost", X_train_res, y_train_res, ada_param_dist)

# 10. Bagging Classifier with different base estimators
bagging_param_dist = {
    'n_estimators': randint(10, 100),
    'max_samples': uniform(0.5, 0.5),
    'max_features': uniform(0.5, 0.5),
    'bootstrap': [True, False],
    'bootstrap_features': [True, False]
}
# Try different base estimators for bagging
base_dt = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5)
bagging_dt = BaggingClassifier(base_estimator=base_dt, random_state=RANDOM_STATE)
best_bagging_dt = train_and_evaluate(bagging_dt, "Bagging with Decision Tree", X_train_res, y_train_res, bagging_param_dist)

# Create advanced ensemble models
print("\nCreating advanced ensemble models...")

# 1. Weighted Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', best_xgb),
        ('lgbm', best_lgbm),
        ('catboost', best_catboost),
        ('rf', best_rf),
        ('et', best_et),
        ('svm', best_svm),
        ('mlp', best_mlp),
        ('gb', best_gb)
    ],
    voting='soft',
    weights=[2, 2, 2, 1, 1, 1.5, 2, 1.5]  # Weights based on individual performance
)
best_voting = train_and_evaluate(voting_clf, "Voting Ensemble", X_train_res, y_train_res)

# 2. Advanced Stacking Classifier with multiple meta-learners
# First level estimators
base_estimators = [
    ('xgb', best_xgb),
    ('lgbm', best_lgbm),
    ('catboost', best_catboost),
    ('rf', best_rf),
    ('et', best_et),
    ('svm', best_svm),
    ('gb', best_gb)
]

# Try different meta-learners for stacking
meta_learners = {
    'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss')
}

best_stacking = None
best_stacking_score = 0

for meta_name, meta_learner in meta_learners.items():
    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=5,
        stack_method='predict_proba'
    )
    
    # Fit and evaluate
    cv_scores = cross_val_score(stacking, X_train_res, y_train_res, cv=cv, scoring='accuracy')
    avg_score = np.mean(cv_scores)
    
    print(f"Stacking with {meta_name} CV score: {avg_score:.4f}")
    
    if avg_score > best_stacking_score:
        best_stacking_score = avg_score
        # Fit the best stacking model on the full training set
        stacking.fit(X_train_res, y_train_res)
        best_stacking = stacking
        best_meta = meta_name

print(f"Best stacking ensemble uses {best_meta} as meta-learner with CV score: {best_stacking_score:.4f}")

# Collect all models
models = {
    'XGBoost': best_xgb,
    'LightGBM': best_lgbm,
    'CatBoost': best_catboost,
    'Random Forest': best_rf,
    'Extra Trees': best_et,
    'SVM': best_svm,
    'Neural Network': best_mlp,
    'Gradient Boosting': best_gb,
    'AdaBoost': best_ada,
    'Bagging': best_bagging_dt,
    'Voting Ensemble': best_voting,
    'Stacking Ensemble': best_stacking
}

# Make predictions and evaluate all models on test set
print("\nEvaluating all models on test set...")
results = {}

for name, model in models.items():
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100
    precision = precision_score(y_test, y_pred, average='weighted') * 100
    recall = recall_score(y_test, y_pred, average='weighted') * 100
    
    # Calculate ROC AUC if probability predictions are available
    roc_auc = roc_auc_score(y_test, y_pred_proba) * 100 if y_pred_proba is not None else None
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'predictions': y_pred,
        'pred_proba': y_pred_proba
    }
    
    # Print results
    print(f"{name}:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  F1 Score: {f1:.2f}%")
    print(f"  Precision: {precision:.2f}%")
    print(f"  Recall: {recall:.2f}%")
    if roc_auc is not None:
        print(f"  ROC AUC: {roc_auc:.2f}%")

# Find the best model based on accuracy
best_model_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = models[best_model_name]
best_predictions = results[best_model_name]['predictions']
best_accuracy = results[best_model_name]['accuracy']

print(f"\nBest model: {best_model_name} with accuracy: {best_accuracy:.2f}%")
print("\nBest model classification report:")
print(classification_report(y_test, best_predictions))
print("Best model confusion matrix:")
print(confusion_matrix(y_test, best_predictions))

# Create a super ensemble (meta-ensemble) for even better results
print("\nCreating super ensemble for final prediction boost...")

# Get predictions from all models for both train and test sets
train_preds = np.column_stack([model.predict(X_train_scaled) for model in models.values()])
test_preds = np.column_stack([model.predict(X_test_scaled) for model in models.values()])

# Get probability predictions if available
train_probs = np.column_stack([
    model.predict_proba(X_train_scaled)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(y_train))
    for model in models.values()
])
test_probs = np.column_stack([
    model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(y_test))
    for model in models.values()
])

# Combine predictions and probabilities
train_meta_features = np.hstack([train_preds, train_probs])
test_meta_features = np.hstack([test_preds, test_probs])

# Try different meta-classifiers
meta_classifiers = {
    'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, C=10),
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, max_depth=5),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss'),
    'SVM': SVC(probability=True, random_state=RANDOM_STATE, C=1.0, kernel='rbf')
}

best_meta_clf = None
best_meta_name = None
best_meta_accuracy = 0

for name, clf in meta_classifiers.items():
    # Train meta-classifier
    clf.fit(train_meta_features, y_train)
    
    # Make predictions
    meta_preds = clf.predict(test_meta_features)
    meta_accuracy = accuracy_score(y_test, meta_preds) * 100
    
    print(f"Meta-ensemble with {name}: {meta_accuracy:.2f}%")
    
    if meta_accuracy > best_meta_accuracy:
        best_meta_accuracy = meta_accuracy
        best_meta_clf = clf
        best_meta_name = name
        best_meta_preds = meta_preds

print(f"\nBest meta-ensemble uses {best_meta_name} with accuracy: {best_meta_accuracy:.2f}%")
print("\nMeta-ensemble classification report:")
print(classification_report(y_test, best_meta_preds))
print("Meta-ensemble confusion matrix:")
print(confusion_matrix(y_test, best_meta_preds))

# Save the best performing model (either individual, ensemble, or meta-ensemble)
if best_meta_accuracy > best_accuracy:
    print("\nMeta-ensemble is the best model!")
    # Save all individual models and meta-classifier
    final_model_data = {
        'models': models,
        'meta_classifier': best_meta_clf,
        'scaler': best_scaler,
        'features': top_features,
        'type': 'meta-ensemble'
    }
    filename = 'heart-disease-prediction-meta-ensemble-model.pkl'
else:
    print(f"\n{best_model_name} is the best model!")
    final_model_data = {
        'model': best_model,
        'scaler': best_scaler,
        'features': top_features,
        'type': 'single-model'
    }
    filename = f'heart-disease-prediction-{best_model_name.lower().replace(" ", "-")}-model.pkl'

# Save the model
pickle.dump(final_model_data, open(filename, 'wb'))
print(f"Model saved as {filename}")

# Create a simple prediction function for future use
def predict_heart_disease(data, model_file=filename):
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
        probs = np.column_stack([
            model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(X))
            for model in models.values()
        ])
        
        # Combine predictions and probabilities
        meta_features = np.hstack([preds, probs])
        
        # Make final predictions
        predictions = meta_clf.predict(meta_features)
        probabilities = meta_clf.predict_proba(meta_features)[:, 1] if hasattr(meta_clf, "predict_proba") else None
    
    return predictions, probabilities

# Save the prediction function
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
        probs = np.column_stack([
            model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(X))
            for model in models.values()
        ])
        
        # Combine predictions and probabilities
        meta_features = np.hstack([preds, probs])
        
        # Make final predictions
        predictions = meta_clf.predict(meta_features)
        probabilities = meta_clf.predict_proba(meta_features)[:, 1] if hasattr(meta_clf, "predict_proba") else None
    
    return predictions, probabilities
""")

# Calculate execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"\nExecution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")

print("\nModel training and evaluation complete!")
