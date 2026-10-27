"""
Model Evaluation and Metrics Module
Deep Learning Based Fake News Detection System (BCA Final Year Project)

Calculates genuine performance metrics on unseen test data:
- Accuracy, Precision, Recall, F1-Score
- Full Scikit-Learn Classification Report
- Confusion Matrix (True Positives, False Positives, True Negatives, False Negatives)
- Saves confusion matrix plot as high-resolution PNG
"""

import os
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(model, X_test_pad, y_test, output_dir: str = "models", threshold: float = 0.5):
    """
    Evaluates the trained model on test data and produces authentic metrics.
    
    Classes:
    - 0: Fake News
    - 1: Real News
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print("        GENUINE MODEL EVALUATION ON UNSEEN TEST DATA        ")
    print("=" * 60)

    # Generate prediction probabilities
    y_pred_probs = model.predict(X_test_pad, verbose=1)
    # Apply classification threshold
    y_pred = (y_pred_probs >= threshold).astype(int).flatten()
    y_true = np.array(y_test, dtype=int).flatten()

    # Calculate standard metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    clf_report = classification_report(
        y_true,
        y_pred,
        target_names=["Real News (0)", "Fake News (1)"],
        digits=4,
        output_dict=True
    )
    clf_report_text = classification_report(
        y_true,
        y_pred,
        target_names=["Real News (0)", "Fake News (1)"],
        digits=4
    )

    metrics_dict = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": {
            "true_reals_correct": int(tn),
            "false_fakes_real_as_fake": int(fp),
            "false_reals_fake_as_real": int(fn),
            "true_fakes_correct": int(tp)
        },
        "test_samples_count": len(y_true),
        "classification_report": clf_report
    }

    # Print clean results for terminal & viva presentation
    print("\n--- Key Performance Metrics ---")
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1-Score  : {f1 * 100:.2f}%\n")

    print("--- Detailed Classification Report ---")
    print(clf_report_text)

    print("--- Confusion Matrix Breakdown ---")
    print(f"True Reals  (Real identified as Real) : {tn}")
    print(f"False Fakes (Real identified as Fake) : {fp}")
    print(f"False Reals (Fake identified as Real) : {fn}")
    print(f"True Fakes  (Fake identified as Fake) : {tp}")

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"\n[OK] Saved evaluation metrics JSON to: {metrics_path}")

    # Plot and save Confusion Matrix Heatmap
    cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Real (0)", "Predicted Fake (1)"],
        yticklabels=["Actual Real (0)", "Actual Fake (1)"]
    )
    plt.title("BiLSTM Fake News Detection - Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("Actual Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"[OK] Saved confusion matrix heatmap plot to: {cm_plot_path}")

    return metrics_dict
