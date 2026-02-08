import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def load_and_clean():
    print("Step 1: Fetching Clinical Data...")
    dataset = fetch_ucirepo(id=296)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    
    # 2026 Governance: Standardize missing values and drop low-quality columns
    df.replace('?', np.nan, inplace=True)
    df.drop(columns=['weight', 'payer_code', 'medical_specialty'], inplace=True)
    
    # Define High Risk Target (< 30 days)
    df['high_risk'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    df.drop(columns=['readmitted'], inplace=True)
    
    # Simple encoding for this baseline
    df = pd.get_dummies(df, drop_first=True)
    return df.dropna()

def train_governed_model(df):
    print("Step 2: Training Model with Class Weighting (Recall Focus)...")
    X = df.drop('high_risk', axis=1)
    y = df['high_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Governance: Using 'balanced' weights to ensure we don't ignore minority class
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluation
    probs = model.predict_proba(X_test)[:, 1]
    print(f"\nModel Performance Metrics:")
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.2f}")
    print(classification_report(y_test, model.predict(X_test)))
    
    return model, X_test

def run_explainability_audit(model, X_test):
    print("\nStep 3: Running SHAP Explainability Audit (Sampled for Speed)...")
    # Governance Move: Sample 500 patients for a fast but accurate audit
    X_sample = X_test.sample(500, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Plotting the 'Why'
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values[1], X_sample, show=False)
    plt.title("2026 AI Governance: Feature Impact on Readmission Risk")
    plt.show()

if __name__ == "__main__":
    data = load_and_clean()
    clinical_model, test_data = train_governed_model(data)
    run_explainability_audit(clinical_model, test_data)