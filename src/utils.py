from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(y_true, y_pred, model_name="Model"):
    print(f"\nClassification Report for {model_name}:\n")
    print(classification_report(y_true, y_pred))
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.show()

def calculate_best_threshold(y_true, y_probs, min_precision=0.3, min_recall=0.9):
    from sklearn.metrics import precision_recall_curve
    prec, rec, thresh = precision_recall_curve(y_true, y_probs)

    for i in range(len(thresh)):
        if rec[i] >= min_recall and prec[i] >= min_precision:
            return thresh[i], rec[i], prec[i]

    return 0.5, 0.0, 0.0