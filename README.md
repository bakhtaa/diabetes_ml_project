# 🩺 MediAI - Intelligent Validation and Medical Analysis Diagnosis System

MediAI is a healthcare Machine Learning project that combines two predictive models:

- **Pre-Analytical Quality Validation Model** → *(Valid or Erroneous)*
- **Diabetes Diagnosis Model** → *(Diabetic or Non-Diabetic)*

The Streamlit web application applies a secure business workflow:

✅ If the analysis quality is **Erroneous**, diagnosis is blocked.  
✅ If the analysis quality is **Valid**, diabetes diagnosis is launched.

This project is based on:

- `app.py`
- `projet_ml_v2.ipynb`

---

# 📌 Objectives

The project aims to:

- Improve medical decision reliability by separating laboratory quality validation from clinical interpretation
- Prevent diagnosis based on non-compliant laboratory analyses
- Compare multiple Machine Learning algorithms and select the best-performing models
- Provide an interactive interface for user input and result interpretation

---

# 🚀 Main Features

## Complete 2-Step Diagnosis Process

### Step 1 — Laboratory Quality Verification

Checks laboratory pre-analytical indicators such as:

- Temperature
- Delay before processing
- Centrifugation
- Hemolysis
- Sample conditions

### Step 2 — Diabetes Diagnosis

Uses medical parameters including:

- Glucose level
- BMI
- Age
- Blood pressure
- Additional patient indicators

---

## Additional Functionalities

- Diabetes dataset EDA visualization
- Laboratory quality dataset EDA visualization
- Machine Learning model comparison
- Performance metrics visualization
- Architecture page explaining system logic

---

# 📁 Project Structure

```bash
MediAI/
│
├── app.py
├── projet_ml_v2.ipynb
├── diabetes.csv
├── qualite_labo.csv
├── notes.txt
│
├── model_artifacts/
│   ├── model_diagnostic.pkl
│   ├── scaler_diagnostic.pkl
│   ├── features_diagnostic.pkl
│   ├── needs_scaling_diag.pkl
│   ├── best_diag_name.pkl
│   ├── comp_diag.pkl
│   │
│   ├── model_qualite.pkl
│   ├── scaler_qualite.pkl
│   ├── features_qualite.pkl
│   ├── le_qualite.pkl
│   ├── le_type.pkl
│   ├── needs_scaling_qual.pkl
│   ├── best_qual_name.pkl
│   ├── comp_qual.pkl
│   │
│   ├── diabetes_clean.csv
│   └── qualite_labo.csv
```

---

# ⚙️ Requirements

- Python 3.9+
- pip

Required Python libraries:

```txt
streamlit
scikit-learn
pandas
numpy
matplotlib
seaborn
joblib
```

---

# 🔧 Installation

Open a terminal inside the project folder and install dependencies:

```bash
pip install streamlit scikit-learn pandas numpy matplotlib seaborn joblib
```

---

# ▶️ Running the Project

### Optional (recommended)

Run the notebook to regenerate model artifacts:

```bash
projet_ml_v2.ipynb
```

### Launch the Streamlit application

```bash
streamlit run app.py
```

---

# 📦 Expected Artifacts

The application automatically loads the following files from `model_artifacts/`:

```txt
model_diagnostic.pkl
scaler_diagnostic.pkl
features_diagnostic.pkl
needs_scaling_diag.pkl
best_diag_name.pkl
comp_diag.pkl

model_qualite.pkl
scaler_qualite.pkl
features_qualite.pkl
le_qualite.pkl
le_type.pkl
needs_scaling_qual.pkl
best_qual_name.pkl
comp_qual.pkl

diabetes_clean.csv
qualite_labo.csv
```

---

# 📊 Datasets Used

## Diabetes Dataset

Pima Indians Diabetes Dataset

Referenced from Kaggle/UCI inside the notebook.

---

## Pre-Analytical Quality Dataset

Semi-synthetic dataset based on biological quality rules inspired by laboratory standards.

---

# 🧭 Application Navigation

Available pages inside `app.py`:

- Complete Diagnosis
- EDA – Diabetes
- EDA – Laboratory Quality
- Model Comparison
- Architecture

---

# 🏗️ Decision Workflow

```text
Patient Input
       ↓
Quality Validation Model
       ↓
 ┌───────────────┐
 │ Erroneous ?   │
 └──────┬────────┘
        │ YES
        ↓

Block diagnosis
Recommend new sample collection

        OR

        ↓ NO

Launch Diabetes Diagnostic Model
```

---

# ⚠️ Limitations and Disclaimer

This project is intended for:

**Educational and academic purposes only**

Important notes:

- Predictions do not replace medical advice
- Diagnostic quality depends on dataset quality
- Results should never substitute professional healthcare evaluation

---

# 🔮 Possible Improvements

Future enhancements may include:

- Add a `requirements.txt` file
- Add unit testing for artifact loading
- Docker containerization
- Continuous Integration (CI)
- MLflow experiment tracking
- Deployment to cloud platforms

---

# 👨‍💻 Author

**MediAI – Healthcare Machine Learning Project**

Business logic and interface implementation:

```python
app.py
```

Machine Learning pipeline and training:

```python
projet_ml_v2.ipynb
```

---
Made with ❤️ by Bakhta, Mayssa et Nour
 using Streamlit + Scikit-Learn