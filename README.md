# Diabetes Prediction using Stacking Ensemble

This project builds a machine learning model to predict diabetes using health indicators such as HbA1c level, blood glucose level, age, BMI, and more. A Stacking Ensemble is used to combine the strengths of multiple classifiers, and special care is taken to handle class imbalance, especially minimizing Type II errors (false negatives).

## 📊 Dataset Source
This dataset is publicly available on [Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset).  
It contains anonymized patient health indicators used for diabetes classification.


## 🚀 Model Architecture
We use a **StackingClassifier** with the following base learners:
- Logistic Regression
- Random Forest (with class weights)
- Gradient Boosting
- XGBoost (with scale_pos_weight)

The meta-learner is another Logistic Regression model.

## ✅ Evaluation
- Optimized for high **Recall** to reduce missed diabetic cases
- Custom threshold selection based on precision-recall curve
- ROC-AUC: 0.98
- SHAP and XGBoost feature importance used for interpretation

## 📦 Contents
- `notebooks/`: Jupyter notebook with full pipeline
- `models/`: Saved `.joblib` models
- `plots/`: ROC, SHAP, and feature importance visualizations
- `requirements.txt`: Required Python packages

## 🛠️ Setup

```bash
pip install -r requirements.txt
```

## 📈 Results
- Best recall: 97.9% with a threshold of 0.112
- Most important features: HbA1c level, blood glucose level, age

## 📂 Deployment
Model is saved and ready for loading via `joblib`.

## 🧠 Author
Created by [Oghuz Hasanli], 2025.
