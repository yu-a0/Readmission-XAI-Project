import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lime
import lime.lime_tabular

# --- 1. HUMAN-READABLE MAPPING ---
def map_icd9(code):
    """Translates numeric codes into Clinical Categories for the Y-Axis labels."""
    if pd.isnull(code) or code == '?': return 'Other'
    try:
        if str(code).startswith(('V', 'E')): return 'Other'
        val = float(code)
        # Groups based on standard medical billing categories
        if 390 <= val <= 459 or val == 785: return 'Circulatory'
        elif 460 <= val <= 519 or val == 786: return 'Respiratory'
        elif 520 <= val <= 579 or val == 787: return 'Digestive'
        elif 250 <= val < 251: return 'Diabetes'
        elif 800 <= val <= 999: return 'Injury'
        elif 710 <= val <= 739: return 'Musculoskeletal'
        elif 580 <= val <= 629 or val == 788: return 'Genitourinary'
        else: return 'Other'
    except: return 'Other'

print("Step 1: Fetching and Grouping Clinical Data...")
dataset = fetch_ucirepo(id=296)
df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

# Apply mapping to ensure labels like 'diag_1_Circulatory' appear instead of numbers
for col in ['diag_1', 'diag_2', 'diag_3']:
    df[col] = df[col].apply(map_icd9)

# Basic cleaning
df.drop(columns=['weight', 'payer_code', 'medical_specialty'], inplace=True, errors='ignore')
df['high_risk'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
df.drop(columns=['readmitted'], inplace=True)
df_encoded = pd.get_dummies(df, drop_first=True)

# --- 2. MODEL TRAINING ---
print("Step 2: Training Clinical Risk Model...")
X = df_encoded.drop('high_risk', axis=1)
y = df_encoded['high_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using 'balanced' weights to prioritize catching high-risk patients
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# --- STEP 3: LIME EXPLANATION & STABILITY LOCK ---
print("Step 3: Generating Stable LIME Explanation...")

# 1. Initialize the Explainer and LOCK the randomness HERE
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['Low Risk', 'High Risk'],
    mode='classification',
    random_state=42  # <--- THE LOCK MOVES HERE
)

# 2. data_row=X_test.values[101] - Changing the '101' to another number with change the patient 
exp = explainer.explain_instance(
    data_row=X_test.values[101], 
    predict_fn=model.predict_proba, 
    num_features=10
)

# --- 4. EXPORT TO HTML ---
# Save as a standalone interactive webpage
exp.save_to_file('clinical_risk_report.html')
print("\nSuccess! 'clinical_risk_report.html' has been saved to your project folder.")

# Show the plot in VS Code
exp.as_pyplot_figure()
plt.title("Human-Readable Clinical Risk Factors (Locked Sampling)")
plt.tight_layout()
plt.show()