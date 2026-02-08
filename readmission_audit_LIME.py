import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Import LIME
import lime
import lime.lime_tabular

def load_and_clean():
    print("Step 1: Fetching Clinical Data...")
    dataset = fetch_ucirepo(id=296)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    
    # Standard 2026 Data Cleaning
    df.replace('?', np.nan, inplace=True)
    df.drop(columns=['weight', 'payer_code', 'medical_specialty'], inplace=True)
    
    # Target: 1 for Readmitted <30 days (High Risk)
    df['high_risk'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    df.drop(columns=['readmitted'], inplace=True)
    
    # LIME works best with numeric data, so we'll encode our categories
    df = df.fillna("Unknown")
    df_encoded = pd.get_dummies(df, drop_first=True)
    return df_encoded

def train_governed_model(df):
    print("Step 2: Training Clinical Model (Recall Focus)...")
    X = df.drop('high_risk', axis=1)
    y = df['high_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 'balanced' weights help us catch the rare 'High Risk' cases
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_train, X_test

def run_lime_explanation(model, X_train, X_test):
    print("\nStep 3: Generating LIME Explanation for a Specific Patient...")
    
    # Initialize the LIME Explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['Low Risk', 'High Risk'],
        mode='classification'
    )
    
    # Let's pick a patient who was actually flagged as High Risk
    # Index 15 is often a good sample in this dataset
    patient_idx = 15 
    patient_to_explain = X_test.iloc[patient_idx]
    
    # Generate the local explanation
    exp = explainer.explain_instance(
        data_row=patient_to_explain.values, 
        predict_fn=model.predict_proba,
        num_features=8
    )
    
    # Visualize the results
    print(f"Showing 'Clinical Logic' for Patient at index {patient_idx}")
    exp.as_pyplot_figure()
    plt.title("LIME: Why is this patient flagged as High Risk?")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data = load_and_clean()
    model, X_train, X_test = train_governed_model(data)
    run_lime_explanation(model, X_train, X_test)