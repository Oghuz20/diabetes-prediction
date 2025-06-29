import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path('../models/stacking_model.joblib')
THRESHOLD = 0.112
INPUT_PATH = Path('../data/sample_input.csv')

model = joblib.load(MODEL_PATH)
X_new = pd.read_csv(INPUT_PATH)
y_prob = model.predict_proba(X_new)[:, 1]
y_pred = (y_prob >= THRESHOLD).astype(int)

for i, pred in enumerate(y_pred):
    print(f"Sample {i+1} → Diabetes Prediction: {'YES' if pred == 1 else 'NO'}  (Prob: {y_prob[i]:.3f})")