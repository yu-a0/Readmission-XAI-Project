import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import webbrowser
import os
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
import lime
import lime.lime_tabular

# --- 1. CLINICAL MAPPING ---
def map_icd9(code):
    if pd.isnull(code) or code == '?': return 'Other'
    try:
        if str(code).startswith(('V', 'E')): return 'Other'
        val = float(code)
        if 390 <= val <= 459 or val == 785: return 'Circulatory'
        elif 460 <= val <= 519 or val == 786: return 'Respiratory'
        elif 520 <= val <= 579 or val == 787: return 'Digestive'
        elif 250 <= val < 251: return 'Diabetes'
        else: return 'Other'
    except: return 'Other'

# --- 2. DATA LOAD & CLEAN ---
def load_data():
    print("Step 1: Fetching Clinical Data...")
    dataset = fetch_ucirepo(id=296)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    df.replace('?', np.nan, inplace=True)
    
    # Apply Clinical Mapping
    for col in ['diag_1', 'diag_2', 'diag_3']:
        df[col] = df[col].apply(map_icd9)
        
    df['high_risk'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    df.drop(columns=['weight', 'payer_code', 'medical_specialty', 'readmitted'], inplace=True, errors='ignore')
    
    df_encoded = pd.get_dummies(df, drop_first=True)
    return df_encoded.dropna()

# --- 3. SHAP (GLOBAL AUDIT) ---
def run_shap_audit(model, X_test):
    print("\nStep 3: Running Global SHAP Audit (Optimized)...")
    # Small sample ensures it doesn't freeze
    X_sample = X_test.sample(200, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample, check_additivity=False)
    
    # Handle list format for Random Forest classification
    target_values = shap_values[1] if isinstance(shap_values, list) else shap_values
    
    plt.figure(figsize=(10,6))
    shap.summary_plot(target_values, X_sample, show=False)
    plt.title("2026 AI Governance: Global Feature Impact")
    #Automatically adjusts plot area size
    plt.tight_layout()
    #Fine-tuning the size of the charts
    #plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.1)
    plt.savefig('global_risk_audit.png', bbox_inches='tight')
    plt.show()

# --- 4. LIME (LOCAL EXPLANATION) ---
def run_lime_explanation(model, X_train, X_test, patient_index=101):
    print(f"\nStep 4: Generating Patient-Specific Report (Index {patient_index})...")
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['Low Risk', 'High Risk'],
        mode='classification',
        random_state=42 # Locked for reproducibility
    )
    
    # Generate explanation
    exp = explainer.explain_instance(
        X_test.values[patient_index], 
        model.predict_proba, 
        num_features=10
    )
    
    # Save the HTML file
    filename = 'clinical_risk_report.html'
    exp.save_to_file(filename)
    print(f"✅ Success! Report saved as: {filename}")
    
    # Automatically open in browser
    webbrowser.open('file://' + os.path.realpath(filename))

# --- EXECUTION ---
if __name__ == "__main__":
    df = load_data()
    X = df.drop('high_risk', axis=1)
    y = df['high_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Step 2: Training Governed Random Forest...")
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    run_shap_audit(model, X_test)
    run_lime_explanation(model, X_train, X_test)