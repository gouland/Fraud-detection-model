# Healthcare Insurance Fraud Detection

## Overview

This machine learning model identifies fraudulent healthcare insurance claims by analyzing patient encounter data. It's trained to spot two main types of fraud: phantom billing (charging for services never provided) and wrong diagnosis claims (intentional misdiagnosis for higher reimbursements).

Healthcare fraud drains billions from the system annually. This tool helps flag suspicious claims by examining patient demographics, billing patterns, hospital stay duration, and diagnosis complexity.

## The Data

Working with 4,388 insurance claims:
- Clean claims: 4,332 (98.7%)
- Wrong diagnosis fraud: 38 (0.9%)
- Phantom billing: 18 (0.4%)

Yeah, the fraud cases are rare—which makes this a challenging problem.

## How It Works

### Features We Built

**Patient Info:**
- Age groups (Child, Young Adult, Adult, Senior, Elderly)
- Gender (Male=0, Female=1)

**Money Stuff:**
- Billed amounts grouped into categories (No Charge through Extreme)
- One-hot encoding for different billing types

**Timing Patterns:**
- Length of stay calculations
- Month and day of week tracking
- Weekend flag

**Clinical Data:**
- Diagnosis text length (more complex = longer descriptions)
- Basic text analysis of diagnosis codes

### The Model

We built two versions:

**Multi-class version** handles all three categories (no fraud, wrong diagnosis, phantom billing):
- Random Forest with 200 trees
- SMOTE + undersampling to handle the massive class imbalance
- Cross-validation F1 around 0.83

**Binary version** just answers "fraud or not":
- Simpler and often more reliable given how few fraud cases exist
- Better precision/recall trade-offs for production use

Random Forest settings we landed on:
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)
```

## Files You'll Find Here

```
fraud-detection/
├── fraud_detection_model.py          # Main code
├── cleaned_nhis_with_fraud_types.csv # The dataset
├── enhanced_fraud_detector_pipeline.joblib    # Multi-class model
├── binary_fraud_detector.joblib      # Binary model
├── improved_confusion_matrix.png     # Multi-class results
├── binary_confusion_matrix.png       # Binary results
├── feature_importance.png            # What matters most
├── precision_recall_curve.png        # Performance analysis
└── README.md
```

## Getting Started

You'll need Python 3.8 or newer, plus these packages:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib
```

Make sure your data file is at: `C:\Users\Admin\Desktop\fraud checker\cleaned_nhis_with_fraud_types (1).csv`

Then just run:
```bash
python fraud_detection_model.py
```

## Using the Model

To predict fraud on new claims:

```python
import joblib

# Load the model
artifacts = joblib.load("enhanced_fraud_detector_pipeline.joblib")

# Prepare your new claims data (needs same features as training data)
new_claims = prepare_new_claims()

# Get predictions
def predict_fraud(new_data, model_path="enhanced_fraud_detector_pipeline.joblib"):
    artifacts = joblib.load(model_path)
    new_data = new_data[artifacts['feature_names']]
    new_data_scaled = artifacts['scaler'].transform(new_data)
    
    predictions = artifacts['pipeline'].predict(new_data_scaled)
    probabilities = artifacts['pipeline'].predict_proba(new_data_scaled)
    
    return pd.DataFrame({
        'prediction': artifacts['target_encoder'].inverse_transform(predictions),
        'probability': np.max(probabilities, axis=1)
    })

results = predict_fraud(new_claims)
print(results)
```

## What We Learned

The extreme class imbalance (less than 2% fraud) makes this tricky. The binary model tends to perform better in practice since it doesn't try to distinguish between fraud types—just flags anything suspicious.

Feature importance analysis showed that billing amounts, length of stay, and certain temporal patterns are the strongest fraud indicators.

## Limitations

Be real about what this can and can't do:
- Training data is heavily skewed (very few fraud examples)
- Only as good as the fraud labels in historical data
- Limited to the features we could extract from claims
- Won't catch brand new fraud schemes it hasn't seen before

This isn't a magic bullet. Use it as part of a broader fraud detection strategy, not as the sole decision-maker.

## What's Next

Some ideas for improvement:
- Add provider history and network analysis
- Try ensemble methods combining multiple models
- Implement anomaly detection for unknown fraud types
- Build explainability features so investigators understand why something was flagged
- Set up real-time processing for live claims

## Charts and Metrics

The code generates several visualizations showing model performance, feature importance, and the precision/recall trade-offs. Check the PNG files after running the model.

## Contributing

Pull requests welcome. If you're adding features, include tests and update the documentation.

## License

Educational and research use. If you're deploying this in production, make sure you're compliant with HIPAA and other healthcare data regulations.

---

**Bottom line**: This model flags potentially fraudulent claims for human review. It's a tool to help fraud investigators, not replace them.
