# Explainable AI (XAI) for Hospital Readmission Risk

### Executive Summary

Developed a machine learning pipeline to predict 30-day hospital readmission risk for diabetic patients using the UCI Diabetes dataset. The project transitions from a "black-box" Random Forest model to a transparent, ***clinically interpretable system*** using LIME (Local Interpretable Model-agnostic Explanations).

### Key Technical Contributions

* **Clinical Data Engineering:** Implemented an automated pipeline to map over 700 raw ICD-9 diagnosis codes into 9 high-level clinical categories (e.g., Circulatory, Respiratory, Diabetes), significantly improving model interpretability for medical staff.

* **Balanced Risk Modeling:** Utilized a ***Random Forest Classifier*** with cost-sensitive learning (`class_weight='balanced'`) to address class imbalance, ensuring the model prioritizes high-risk patient detection (Recall).

* **Explainable AI (XAI) Implementation:** ***LIME*** developed a local explanation module to provide "point-of-care" transparency, showing the specific medication changes and history that drive an individual patient's risk score.
    * **Stability Governance:** Integrated random seed locking to ensure reproducibility of explanations, a core requirement for AI deployment in regulated healthcare environments.

* **Interactive Clinical Reporting:** Built an automated export system that generates interactive HTML "Risk Dashboards," allowing non-technical stakeholders to explore model logic through visual gauges and feature-impact charts.

---

## Tools & Technologies

* **Languages:** Python (VS Code environment)
* **Libraries:** Scikit-Learn, Pandas, LIME, Matplotlib, NumPy
* **Domain Focus:** Clinical Informatics, AI Governance, Healthcare Risk Stratification

---

### Global vs. Local Explinations

| Type | Difference |
| :--- | :--- |
| Global (SHAP) | Makes sure the model isn't biased against certain age groups across the whole hospital |
| Local (LIME) | Tells a doctor exactly *why* this patient is being flagged for readmission today |

---

## Graph Interpretation

### Labels

| Technical Label | Clinical Translation |
| :--- | :--- |
| `feature <= 0.00` | The patient does NOT have this condition/medication |
| `feature > 0.00` | The patient DOES have this condition/medication |
| `_Steady, _Up, _Down` | Refers to whether their medication dose was unchanged, increased, or decreased |
| `diag_1_Circulatory` | The primary reason they were admitted was a heart/blood issue |

---

### Chart

| Bar | Clinical Translation |
| :--- | :--- |
| Red Bars (Left of 0) | These are **Protective Factors**. They are pulling the risk score down, making the model think the patient is less likely to come back. (e.g., No emergency visits) |
| Green Bars (Right of 0) | These are **Risk Factors**. They are pushing the score up, making the model more worried. (e.g., Being in an older age bracket) |