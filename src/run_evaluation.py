"""
Stand-Alone Genuine Evaluation Script
Deep Learning Based Fake News Detection System (BCA Final Year Project)

This script:
1. Re-creates the exact unseen test partition from the WELFake dataset (random_state=42).
2. Loads the trained BiLSTM model and fitted Tokenizer.
3. Computes authentic predictions on unseen test samples.
4. Generates and saves:
   - models/evaluation_metrics.json (Accuracy, Precision, Recall, F1, Confusion Matrix)
   - models/confusion_matrix.png (Heatmap with correct Real [0] vs Fake [1] axes)
"""

import os
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
from tensorflow.keras.models import load_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import load_and_clean_dataset, split_data, load_tokenizer, tokenize_and_pad
from src.evaluate import evaluate_model


def run_standalone_evaluation(
    dataset_path=os.path.join(PROJECT_ROOT, "data", "raw", "WELFake_Dataset.csv"),
    model_path=os.path.join(PROJECT_ROOT, "models", "bilstm_fake_news.keras"),
    tokenizer_path=os.path.join(PROJECT_ROOT, "models", "tokenizer.pickle"),
    max_samples=10000,
    maxlen=200
):
    print("==================================================================")
    print("      RUNNING GENUINE EVALUATION ON UNSEEN TEST SET               ")
    print("==================================================================")
    print(f"Loading dataset: {dataset_path}")
    df = load_and_clean_dataset(dataset_path, max_samples=max_samples)

    print("Splitting dataset (70% train, 15% val, 15% test)...")
    _, _, (X_test, y_test) = split_data(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42)

    print(f"\nLoading trained BiLSTM model from: {model_path}")
    model = load_model(model_path)

    print(f"Loading fitted Tokenizer from: {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path)

    print(f"Tokenizing and padding {len(X_test)} unseen test samples...")
    X_test_pad = tokenize_and_pad(tokenizer, X_test, maxlen=maxlen)

    print("\nRunning inference and generating genuine evaluation results...")
    metrics = evaluate_model(
        model=model,
        X_test_pad=X_test_pad,
        y_test=y_test,
        output_dir=os.path.join(PROJECT_ROOT, "models"),
        threshold=0.5
    )

    print("\n[OK] Standalone genuine evaluation completed successfully!")
    return metrics


if __name__ == "__main__":
    run_standalone_evaluation()
