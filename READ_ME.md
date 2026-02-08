# 🏥 Explainable AI for Clinical Risk Stratification

### Executive Summary

An end-to-end machine learning pipeline to predict 30-day hospital readmission risk for diabetic patients using the UCI Diabetes dataset. The project transitions from a "black-box" Random Forest model to a transparent, ***clinically interpretable system*** using LIME (Local Interpretable Model-agnostic Explanations).

---

## 📑 Table of Contents
1. [Live Interactive Dashboard](#-live-interactive-dashboard)
2. [Project Overview](#-project-overview)
3. [The "Aris" Clinical Strategy](#-the-aris-clinical-strategy)
4. [Explainable AI (LIME) Logic](#-explainable-ai-lime-logic)
5. [Installation & Setup](#-installation--setup)
6. [Graph Interpretation](#-graph-interpretation)
7. [Lessons Learned & Project Insights](#-lessons-learned--project-insights)

---

## 🌐 Live Interactive Dashboard
👉 **[View the Live Clinical Risk Report](https://your-username.github.io/Readmission-XAI-Project/clinical_risk_report.html)**
*(Note: Replace with your actual GitHub Pages URL)*

---

## 📝 Project Overview
Using the UCI Diabetes dataset, this model identifies high-risk patients likely to be readmitted within 30 days. 
* **Model:** Random Forest Classifier.
* **Focus:** Recall-optimized to ensure medical staff do not miss "at-risk" individuals.
* **Regulation:** Built with 2026 AI Governance standards in mind (reproducibility and interpretability).

### 🛠️ Tools & Technologies

* **Languages:** Python (VS Code environment)
* **Libraries:** Scikit-Learn, Pandas, LIME, Matplotlib, NumPy
* **Domain Focus:** Clinical Informatics, AI Governance, Healthcare Risk Stratification

### 🚀 Key Technical Contributions

* **Clinical Data Engineering:** Implemented an automated pipeline to map over 700 raw ICD-9 diagnosis codes into 9 high-level clinical categories (e.g., Circulatory, Respiratory, Diabetes), significantly improving model interpretability for medical staff.

* **Balanced Risk Modeling:** Utilized a ***Random Forest Classifier*** with cost-sensitive learning (`class_weight='balanced'`) to address class imbalance, ensuring the model prioritizes high-risk patient detection (Recall).

* **Explainable AI (XAI) Implementation:** ***LIME*** developed a local explanation module to provide "point-of-care" transparency, showing the specific medication changes and history that drive an individual patient's risk score.
    * **Stability Governance:** Integrated random seed locking to ensure reproducibility of explanations, a core requirement for AI deployment in regulated healthcare environments.

* **Interactive Clinical Reporting:** Built an automated export system that generates interactive HTML "Risk Dashboards," allowing non-technical stakeholders to explore model logic through visual gauges and feature-impact charts.

---

## 🩺 The "Aris" Clinical Strategy
Raw medical data is often unreadable (e.g., ICD-9 code `428.0`). I implemented a **Clinical Mapper** to translate these into human-readable categories:
* **Circulatory:** Heart and blood vessel issues.
* **Respiratory:** Lung and breathing complications.
* **Diabetes:** Metabolic-specific markers.

---

## 🔍 Explainable AI (LIME) Logic
Instead of just giving a "High Risk" label, this system provides a **Local Explanation** for every patient.
* **Locked Sampling:** Uses a fixed `random_state` to ensure the explanation remains stable across sessions.
* **Risk Drivers:** Identifies which medication changes (e.g., Nateglinide dosage) or history markers are pushing the risk score up or down.

---

## ⚙️ Installation & Setup
1. Clone the repo: `git clone https://github.com/your-username/Readmission-XAI-Project.git`
2. Install requirements: `pip install -r requirements.txt`
3. Run the audit: `python readmission_model.py`

---

## 📊 Graph Interpretation

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
| 🔴 Red Bars (Negative Weight) | These are **Protective Factors**. Factors that ***decrease*** the risk of readmission (e.g., No emergency visits) |
| 🟢 Green Bars (Positive Weight) | These are **Risk Factors**. Factors that ***increase*** the risk of readmission (e.g., being in an older age bracket, or unstable medication dosage) |

---

### Global vs. Local Explinations

| Type | Difference |
| :--- | :--- |
| Global (SHAP) | Makes sure the model isn't biased against certain age groups across the whole hospital |
| Local (LIME) | Tells a doctor exactly *why* this patient is being flagged for readmission today |

## 💡 Lessons Learned & Project Insights
* **The Myth of the "Black Box":** I learned that complex models like Random Forests don't have to be mysterious. By using LIME, we can extract a "clinical narrative" for every prediction, which is essential for earning physician trust.

* **Feature Engineering is Medicine:** Simply feeding raw ICD-9 codes into a model provides poor results. Mapping them to Clinical Categories (like Circulatory or Diabetes) creates a much stronger signal and a more readable graph.

* **The Importance of Determinism:** In healthcare AI, "randomness" is a liability. By locking the random state of our explainer, we ensure that a patient's risk explanation doesn't change every time a user refreshes the page, satisfying core reproducibility requirements.

* **Precision vs. Recall Trade-offs:** In a hospital setting, missing a high-risk patient (a False Negative) is often more dangerous than a False Alarm. I learned to use class_weight='balanced' to bias the model toward Patient Safety.