import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load and preprocess data
df = pd.read_csv("C:\\Users\\Admin\\Desktop\\fraud checker\\cleaned_nhis_with_fraud_types (1).csv")

# Convert dates and handle missing values
df['DATE OF ENCOUNTER'] = pd.to_datetime(df['DATE OF ENCOUNTER'], errors='coerce')
df['DATE OF DISCHARGE'] = pd.to_datetime(df['DATE OF DISCHARGE'], errors='coerce')

# Enhanced feature engineering
df['LENGTH_OF_STAY'] = (df['DATE OF DISCHARGE'] - df['DATE OF ENCOUNTER']).dt.days
df['LENGTH_OF_STAY'] = df['LENGTH_OF_STAY'].apply(lambda x: x if x >= 0 else 0)  # Set negative to 0

# Create time-based features
df['ENCOUNTER_MONTH'] = df['DATE OF ENCOUNTER'].dt.month
df['ENCOUNTER_DAY_OF_WEEK'] = df['DATE OF ENCOUNTER'].dt.dayofweek
df['IS_WEEKEND'] = df['ENCOUNTER_DAY_OF_WEEK'].isin([5, 6]).astype(int)

# Create billing categories
df['BILLING_CATEGORY'] = pd.cut(df['Amount Billed'], 
                               bins=[-1, 0, 1000, 5000, 10000, 50000, float('inf')],
                               labels=['No Charge', 'Low', 'Medium', 'High', 'Very High', 'Extreme'])

# Age groups
df['AGE_GROUP'] = pd.cut(df['AGE'], 
                        bins=[0, 18, 35, 50, 65, 100],
                        labels=['Child', 'Young Adult', 'Adult', 'Senior', 'Elderly'])

# Diagnosis length (proxy for complexity)
df['DIAGNOSIS_LENGTH'] = df['DIAGNOSIS'].str.len().fillna(0)

# Encode categorical variables
df['GENDER'] = df['GENDER'].map({'M': 0, 'F': 1, 'm': 0, 'f': 1}).fillna(0)

# Target encoding
target_encoder = LabelEncoder()
df['FRAUD_LABEL'] = target_encoder.fit_transform(df['FRAUD_TYPE'])

# Print class distribution
print("Class Distribution:")
print(df['FRAUD_TYPE'].value_counts())
print(f"\nTotal records: {len(df)}")

# Select features for modeling
features = [
    'AGE', 'GENDER', 'Amount Billed', 'LENGTH_OF_STAY',
    'ENCOUNTER_MONTH', 'ENCOUNTER_DAY_OF_WEEK', 'IS_WEEKEND',
    'DIAGNOSIS_LENGTH'
]

# One-hot encoding for categorical features
billing_dummies = pd.get_dummies(df['BILLING_CATEGORY'], prefix='BILLING')
age_dummies = pd.get_dummies(df['AGE_GROUP'], prefix='AGE_GROUP')

# Combine all features
X = pd.concat([
    df[features],
    billing_dummies,
    age_dummies
], axis=1)

y = df['FRAUD_LABEL']

# Handle missing values
X = X.fillna(X.median())

print(f"Feature matrix shape: {X.shape}")
print(f"Features used: {list(X.columns)}")

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# FIXED: Multi-class imbalance handling with INTEGER values
# Get class distribution
class_counts = pd.Series(y_train).value_counts()
print(f"\nTraining set class distribution: {class_counts.to_dict()}")

# Define sampling strategy for multi-class - MUST USE INTEGERS
# Note: Class 1 = Wrong Diagnosis (26 samples), Class 2 = Phantom Billing (13 samples)
smote_strategy = {
    1: min(500, max(class_counts[1] * 5, 100)),  # Increase Wrong Diagnosis
    2: min(300, max(class_counts[2] * 10, 50))   # Increase Phantom Billing more aggressively
}

# For undersampling: reduce majority class - MUST USE INTEGER
under_strategy = {
    0: int(class_counts[0] * 0.7)  # Reduce No Fraud to 70% of original
}

print(f"SMOTE strategy: {smote_strategy}")
print(f"Under sampling strategy: {under_strategy}")

# Handle class imbalance with SMOTE + UnderSampling
smote = SMOTE(sampling_strategy=smote_strategy, random_state=42, k_neighbors=2)  # Reduced k_neighbors for small classes
under = RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)

# Enhanced Random Forest with hyperparameters
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# Create pipeline
pipeline = Pipeline([
    ('smote', smote),
    ('under', under),
    ('classifier', model)
])

# Train model
print("Training model with multi-class imbalance handling...")
pipeline.fit(X_train_scaled, y_train)

# Check resampled distribution
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
X_resampled, y_resampled = under.fit_resample(X_resampled, y_resampled)
print(f"Resampled class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train_scaled, y_train, cv=cv, scoring='f1_macro')
print(f"Cross-validation F1 scores: {cv_scores}")
print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# =============================================================================
# COMPREHENSIVE MODEL EVALUATION
# =============================================================================

# Predictions
y_pred = pipeline.predict(X_test_scaled)
y_pred_proba = pipeline.predict_proba(X_test_scaled)

print("\n" + "="*60)
print("COMPREHENSIVE MODEL EVALUATION")
print("="*60)

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# AUC-ROC
try:
    auc_roc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    print(f"\nAUC-ROC Score: {auc_roc:.4f}")
except Exception as e:
    print(f"\nAUC-ROC calculation failed: {e}")

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_encoder.classes_,
            yticklabels=target_encoder.classes_)
plt.title('Confusion Matrix\n', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('improved_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================================================

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': pipeline.named_steps['classifier'].feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
plt.title('Top 15 Feature Importances\n', fontsize=14, fontweight='bold')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# =============================================================================
# PRECISION-RECALL ANALYSIS
# =============================================================================

# Precision-Recall tradeoff for fraud classes
fraud_classes = [i for i, label in enumerate(target_encoder.classes_) if label != 'No Fraud']

if len(fraud_classes) > 0:
    plt.figure(figsize=(12, 8))
    for fraud_class in fraud_classes:
        precision, recall, thresholds = precision_recall_curve(
            (y_test == fraud_class).astype(int), 
            y_pred_proba[:, fraud_class]
        )
        
        plt.plot(recall, precision, label=f'{target_encoder.classes_[fraud_class]}', linewidth=2)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Fraud Classes\n', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

# Save the complete pipeline (including preprocessing)
model_artifacts = {
    'pipeline': pipeline,
    'scaler': scaler,
    'target_encoder': target_encoder,
    'feature_names': list(X.columns),
    'model_metadata': {
        'training_date': pd.Timestamp.now(),
        'features_used': list(X.columns),
        'cv_score': cv_scores.mean() if 'cv_scores' in locals() else None,
        'auc_roc': auc_roc if 'auc_roc' in locals() else None
    }
}

joblib.dump(model_artifacts, "enhanced_fraud_detector_pipeline.joblib")
print("✅ Enhanced model pipeline saved successfully!")

# Function for new predictions
def predict_fraud(new_data, model_path="enhanced_fraud_detector_pipeline.joblib"):
    """
    Predict fraud for new data
    """
    artifacts = joblib.load(model_path)
    
    # Ensure same features and order
    new_data = new_data[artifacts['feature_names']]
    new_data_scaled = artifacts['scaler'].transform(new_data)
    
    predictions = artifacts['pipeline'].predict(new_data_scaled)
    probabilities = artifacts['pipeline'].predict_proba(new_data_scaled)
    
    results = pd.DataFrame({
        'prediction': artifacts['target_encoder'].inverse_transform(predictions),
        'probability': np.max(probabilities, axis=1)
    })
    
    return results

print("\n" + "="*60)
print("MODEL TRAINING COMPLETE")
print("="*60)
print(f"Model saved as: enhanced_fraud_detector_pipeline.joblib")
if 'auc_roc' in locals():
    print(f"Final AUC-ROC: {auc_roc:.4f}")
if 'cv_scores' in locals():
    print(f"Final CV F1: {cv_scores.mean():.4f}")

# =============================================================================
# ADDITIONAL ANALYSIS: BINARY FRAUD DETECTION (RECOMMENDED)
# =============================================================================

print("\n" + "="*60)
print("ADDITIONAL: BINARY FRAUD DETECTION ANALYSIS")
print("="*60)

# Create binary target (Fraud vs No Fraud)
df['IS_FRAUD'] = (df['FRAUD_TYPE'] != 'No Fraud').astype(int)

# Use the same features
X_binary = X.copy()
y_binary = df['IS_FRAUD']

# Split with stratification
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_binary, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

# Scale features
X_train_bin_scaled = scaler.fit_transform(X_train_bin)
X_test_bin_scaled = scaler.transform(X_test_bin)

# Binary SMOTE
smote_binary = SMOTE(sampling_strategy=0.3, random_state=42)  # Balance fraud class

# Binary model
binary_pipeline = Pipeline([
    ('smote', smote_binary),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

print("Training binary fraud detection model...")
binary_pipeline.fit(X_train_bin_scaled, y_train_bin)

# Predictions
y_pred_bin = binary_pipeline.predict(X_test_bin_scaled)
y_proba_bin = binary_pipeline.predict_proba(X_test_bin_scaled)[:, 1]

print("\n--- BINARY FRAUD DETECTION RESULTS ---")
print(classification_report(y_test_bin, y_pred_bin, target_names=['No Fraud', 'Fraud']))

# Binary confusion matrix
plt.figure(figsize=(8, 6))
cm_bin = confusion_matrix(y_test_bin, y_pred_bin)
sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Fraud', 'Fraud'],
            yticklabels=['No Fraud', 'Fraud'])
plt.title('Binary Fraud Detection - Confusion Matrix\n', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('binary_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature importance for binary model
feature_importance_bin = pd.DataFrame({
    'feature': X_binary.columns,
    'importance': binary_pipeline.named_steps['classifier'].feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features for Binary Fraud Detection:")
print(feature_importance_bin.head(10))

# Save binary model
binary_model_artifacts = {
    'pipeline': binary_pipeline,
    'scaler': scaler,
    'feature_names': list(X_binary.columns),
    'model_metadata': {
        'training_date': pd.Timestamp.now(),
        'model_type': 'binary_fraud_detection'
    }
}

joblib.dump(binary_model_artifacts, "binary_fraud_detector.joblib")
print("✅ Binary fraud detection model saved successfully!")
