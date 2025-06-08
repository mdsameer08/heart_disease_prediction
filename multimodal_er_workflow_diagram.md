# Multimodal Heart Disease Prediction System: Entity Relationship Workflow Diagram

## Diagram Description

The following is a textual representation of an entity relationship workflow diagram for a multimodal heart disease prediction system that integrates ECGs, Electronic Health Records (EHRs), and wearable sensor data. This diagram is based on the ensemble machine learning approach used in the final_prediction.py file.

```
+---------------------+     +---------------------+     +---------------------+
|                     |     |                     |     |                     |
|  ECG Data Sources   |     |  EHR Data Sources   |     | Wearable Sensor     |
|                     |     |                     |     | Data Sources        |
+----------+----------+     +----------+----------+     +----------+----------+
           |                           |                           |
           v                           v                           v
+----------+----------+     +----------+----------+     +----------+----------+
|                     |     |                     |     |                     |
| ECG Data Processing |     | EHR Data Processing |     | Wearable Data      |
| - Signal filtering  |     | - Data extraction   |     | Processing         |
| - Feature extraction|     | - Normalization     |     | - Noise reduction  |
| - Waveform analysis |     | - Missing data      |     | - Feature          |
|                     |     |   imputation        |     |   extraction       |
+----------+----------+     +----------+----------+     +----------+----------+
           |                           |                           |
           v                           v                           v
+----------+----------+     +----------+----------+     +----------+----------+
|                     |     |                     |     |                     |
| ECG Feature         |     | EHR Feature         |     | Wearable Feature   |
| Engineering         |     | Engineering         |     | Engineering        |
| - Interval analysis |     | - Age grouping      |     | - Activity patterns|
| - Morphology        |     | - Lab value ratios  |     | - Sleep metrics    |
|   features          |     | - Medical history   |     | - Heart rate       |
| - Rhythm analysis   |     |   categorization    |     |   variability      |
+----------+----------+     +----------+----------+     +----------+----------+
           |                           |                           |
           |                           |                           |
           +---------------------------+--------------------------+
                                       |
                                       v
                           +-----------+-----------+
                           |                       |
                           | Feature Fusion        |
                           | - Early fusion        |
                           | - Feature selection   |
                           | - Dimensionality      |
                           |   reduction           |
                           |                       |
                           +-----------+-----------+
                                       |
                                       v
                           +-----------+-----------+
                           |                       |
                           | Class Balancing       |
                           | - SMOTEENN            |
                           | - Stratified sampling |
                           |                       |
                           +-----------+-----------+
                                       |
                                       v
                           +-----------+-----------+
                           |                       |
                           | Model Training        |
                           |                       |
           +---------------+---+---------------+---+---------------+
           |                   |               |   |               |
           v                   v               v   v               v
+----------+------+   +--------+--------+ +----+---+----+   +-----+-------+
|                 |   |                 | |             |   |             |
| Neural Network  |   | SVM             | | XGBoost     |   | Random      |
| Models          |   | Models          | | Models      |   | Forest      |
| - MLP           |   | - RBF kernel    | |             |   | Models      |
| - CNN for ECG   |   | - Linear kernel | |             |   |             |
|                 |   |                 | |             |   |             |
+---------+-------+   +---------+-------+ +------+------+   +------+------+
          |                     |                |                 |
          |                     |                |                 |
          +---------------------+----------------+-----------------+
                                |
                                v
                    +-----------+-----------+
                    |                       |
                    | Level 1 Ensemble      |
                    | - Voting Classifier   |
                    | - Weighted voting     |
                    |                       |
                    +-----------+-----------+
                                |
                                v
                    +-----------+-----------+
                    |                       |
                    | Meta-Ensemble         |
                    | - Stacking            |
                    | - SVC meta-classifier |
                    |                       |
                    +-----------+-----------+
                                |
                                v
                    +-----------+-----------+
                    |                       |
                    | Model Evaluation      |
                    | - Accuracy            |
                    | - Precision/Recall    |
                    | - ROC-AUC             |
                    | - Confusion Matrix    |
                    |                       |
                    +-----------+-----------+
                                |
                                v
                    +-----------+-----------+
                    |                       |
                    | Best Model Selection  |
                    | - Compare single vs   |
                    |   ensemble models     |
                    | - Select highest      |
                    |   performing model    |
                    |                       |
                    +-----------+-----------+
                                |
                                v
                    +-----------+-----------+
                    |                       |
                    | Model Deployment      |
                    | - Serialization       |
                    | - API integration     |
                    | - Web interface       |
                    |                       |
                    +-----------+-----------+
                                |
                                v
                    +-----------+-----------+
                    |                       |
                    | Real-time Prediction  |
                    | - Patient risk score  |
                    | - Confidence interval |
                    | - Feature importance  |
                    |                       |
                    +-----------+-----------+
```

## Entity Descriptions

### Data Sources
1. **ECG Data Sources**
   - 12-lead ECG recordings
   - Holter monitor data
   - Event recorder data

2. **EHR Data Sources**
   - Demographic information (age, sex)
   - Medical history
   - Lab test results (cholesterol, blood sugar)
   - Medication information
   - Previous diagnoses

3. **Wearable Sensor Data Sources**
   - Continuous heart rate monitoring
   - Activity tracking
   - Sleep patterns
   - Blood pressure trends
   - Oxygen saturation levels

### Data Processing
1. **ECG Data Processing**
   - Signal filtering and noise reduction
   - Baseline wander correction
   - QRS complex detection
   - Interval measurement (PR, QT, QRS)
   - Waveform morphology analysis

2. **EHR Data Processing**
   - Data extraction from clinical databases
   - Standardization of medical codes
   - Handling of missing values
   - Temporal alignment of records
   - Normalization of lab values

3. **Wearable Data Processing**
   - Noise filtering
   - Motion artifact removal
   - Signal quality assessment
   - Data aggregation across time periods
   - Synchronization with other data sources

### Feature Engineering
1. **ECG Feature Engineering**
   - Heart rate variability metrics
   - ST segment deviation measurements
   - T-wave alternans
   - QT dispersion
   - Frequency domain features

2. **EHR Feature Engineering**
   - Age grouping
   - Lab value ratios (e.g., trestbps_chol_ratio)
   - Medication interaction features
   - Temporal patterns in vital signs
   - Risk factor combinations

3. **Wearable Feature Engineering**
   - Resting heart rate trends
   - Heart rate recovery after activity
   - Sleep quality metrics
   - Activity level categorization
   - Stress level estimation

### Feature Fusion and Model Training
1. **Feature Fusion**
   - Early fusion (combining features before modeling)
   - Feature selection using combined ranking methods
   - Dimensionality reduction techniques
   - Creation of cross-modal interaction features

2. **Class Balancing**
   - SMOTEENN for handling imbalanced classes
   - Stratified sampling techniques
   - Weighted loss functions

3. **Model Training**
   - Neural Networks (including specialized CNN for ECG)
   - Support Vector Machines
   - XGBoost and Gradient Boosting
   - Random Forest
   - Specialized models for each data modality

### Ensemble Methods
1. **Level 1 Ensemble**
   - Voting Classifier with weighted voting
   - Specialized weights for different data modalities
   - Soft voting for probability calibration

2. **Meta-Ensemble**
   - Stacking approach with SVC meta-classifier
   - Learning optimal combinations of model predictions
   - Handling of prediction conflicts between modalities

### Evaluation and Deployment
1. **Model Evaluation**
   - Comprehensive metrics (accuracy, precision, recall, F1)
   - ROC-AUC and precision-recall curves
   - Confusion matrix analysis
   - Cross-validation strategies

2. **Model Deployment**
   - Serialization for production use
   - API development for integration
   - Web interface for clinical use
   - Real-time prediction capabilities

## Implementation Notes

1. The multimodal approach extends the existing ensemble methodology by incorporating specialized models for each data type (ECG, EHR, wearable).

2. Feature fusion occurs at multiple levels:
   - Early fusion: combining features before modeling
   - Late fusion: ensemble methods combining model predictions

3. The two-level ensemble approach is particularly valuable for multimodal data:
   - First level: combines predictions from models specialized in each data modality
   - Meta-ensemble: learns optimal weighting across modalities

4. The system maintains the core strengths of the original implementation:
   - Robust feature engineering
   - Class imbalance handling
   - Comprehensive model evaluation
   - Flexible deployment options

5. Additional considerations for multimodal implementation:
   - Data synchronization across modalities
   - Handling missing modalities for some patients
   - Varying importance of modalities for different patient subgroups
   - Computational efficiency with increased feature dimensionality