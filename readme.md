# Diabetes Prediction – Logistic Regression From Scratch + Model Comparison

This project is my practical implementation of machine learning fundamentals.  
I went beyond theory and actually **built logistic regression from scratch** using NumPy – including:
- Sigmoid function
- Binary cross-entropy loss
- Gradient descent
- L2 regularization
- Custom evaluation metrics (accuracy, precision, recall, F1)

The goal was to understand **how ML really works under the hood**, not just use sklearn blindly.

---

## Dataset
Pima Indians Diabetes Dataset  
768 samples • 8 clinical features  
Target: 1 = diabetes, 0 = no diabetes

During EDA, I discovered:
- Several features contain **invalid zero values** (Glucose, BMI, Insulin…)
- Dataset is mildly **imbalanced**
- Diabetes is **not linearly separable**, meaning perfect prediction is unrealistic

---

## Preprocessing
I wrote my own preprocessing pipeline:
- Train-test split (stratified)
- Replaced biologically impossible zeros → NaN
- Median imputation (train-only to avoid leakage)
- Standard scaling (train stats only)

This preprocessing is fully modular (in `preprocessing.py`) and reused across models.

---

## Models Implemented

| Model | Built From Scratch | Accuracy | Recall | F1 |
|-------|-------------------|----------|--------|-----|
| Logistic Regression | NumPy | 0.7446 | 0.5308 | 0.5931 |
| Decision Tree | sklearn | 0.7229 | 0.4815 | 0.5493 |
| Random Forest | sklearn | 0.7489 | 0.5432 | 0.6027 |
| **XGBoost (tuned)** | xgboost | **0.7662** | **0.7777** | **0.70** |

**Best performer:** Tuned XGBoost – and most importantly, it improved **recall**, which matters in medical prediction (catching more diabetic patients).

---

## Key Learning
The biggest lesson wasn't accuracy – it was understanding **why** results behave the way they do.

- Logistic regression is a strong baseline when data is small + noisy
- Random Forest gives slight lift by reducing variance
- XGBoost only worked after **tuning + class balancing**
- Most limitations came from **data**, not algorithms

> “Good ML engineering is knowing when to improve the model – and when the dataset itself is the bottleneck.”
