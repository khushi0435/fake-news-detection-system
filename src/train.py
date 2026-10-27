"""
Model Training Pipeline
Deep Learning Based Fake News Detection System (BCA Final Year Project)

Steps in this pipeline:
1. Loads dataset from data/raw/WELFake_Dataset.csv (or creates starter set).
2. Cleans text, removes duplicates, handles missing values.
3. Splits into Train (70%), Validation (15%), and Test (15%) sets.
4. Fits Keras Tokenizer ONLY on training set to prevent data leakage.
5. Builds Bidirectional LSTM neural network.
6. Trains with EarlyStopping and learning rate reduction callbacks.
7. Saves trained model (models/bilstm_fake_news.keras) and tokenizer.
8. Evaluates on unseen test set and outputs genuine performance metrics.
"""

import os
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import (
    load_and_clean_dataset,
    split_data,
    fit_and_save_tokenizer,
    tokenize_and_pad
)
from src.model import build_bilstm_model
from src.evaluate import evaluate_model
from data.download_dataset import create_starter_sample


def plot_training_history(history, save_path="models/training_history.png"):
    """Plots and saves Loss and Accuracy progression over epochs."""
    plt.figure(figsize=(12, 5))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2, linestyle='--')
    plt.title('BiLSTM Model Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2, linestyle='--')
    plt.title('BiLSTM Model Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] Saved training history plot to: {save_path}")


def run_training(
    dataset_path: str = None,
    epochs: int = 6,
    batch_size: int = 64,
    vocab_size: int = 15000,
    embedding_dim: int = 100,
    maxlen: int = 200,
    max_samples: int = None
):
    """
    Executes the full training and evaluation workflow.
    """
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    if dataset_path is None:
        dataset_path = os.path.join(PROJECT_ROOT, "data", "raw", "WELFake_Dataset.csv")

    # If dataset file does not exist, create starter dataset for instant development
    if not os.path.exists(dataset_path) or os.path.getsize(dataset_path) < 1000:
        print("[!] Dataset not found. Generating sample starter dataset for instant setup...")
        dataset_path = create_starter_sample(sample_size=1600)

    # 1. Load and Clean Dataset
    df = load_and_clean_dataset(dataset_path, max_samples=max_samples)

    # 2. Strict Train / Val / Test Split (Prevents data leakage)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(
        df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    # 3. Fit Tokenizer on Training Data ONLY
    tokenizer_path = os.path.join(models_dir, "tokenizer.pickle")
    tokenizer = fit_and_save_tokenizer(X_train, save_path=tokenizer_path, num_words=vocab_size)

    # 4. Tokenize and Pad Sequences
    print("Tokenizing and padding text sequences...")
    X_train_pad = tokenize_and_pad(tokenizer, X_train, maxlen=maxlen)
    X_val_pad = tokenize_and_pad(tokenizer, X_val, maxlen=maxlen)
    X_test_pad = tokenize_and_pad(tokenizer, X_test, maxlen=maxlen)

    print(f"Padded shape: Train={X_train_pad.shape}, Val={X_val_pad.shape}, Test={X_test_pad.shape}")

    # 5. Build BiLSTM Model
    print("\nInitializing Bidirectional LSTM Neural Network...")
    model = build_bilstm_model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        maxlen=maxlen,
        lstm_units=64,
        dense_units=32,
        dropout_rate=0.3,
        learning_rate=0.001
    )
    model.summary()

    # 6. Define Callbacks
    model_save_path = os.path.join(models_dir, "bilstm_fake_news.keras")
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    # 7. Train the Model
    print(f"\nStarting training for {epochs} epochs with batch size {batch_size}...")
    history = model.fit(
        X_train_pad,
        y_train,
        validation_data=(X_val_pad, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model explicitly
    model.save(model_save_path)
    print(f"\n[OK] Model successfully saved at: {model_save_path}")

    # Plot training curves
    history_plot_path = os.path.join(models_dir, "training_history.png")
    plot_training_history(history, save_path=history_plot_path)

    # 8. Genuine Evaluation on Unseen Test Set
    metrics = evaluate_model(model, X_test_pad, y_test, output_dir=models_dir)

    print("\n" + "=" * 60)
    print("           TRAINING & EVALUATION PIPELINE COMPLETED           ")
    print("=" * 60)
    return model, tokenizer, metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train BiLSTM Fake News Detector")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_samples", type=int, default=15000, help="Max dataset rows to use (for memory/time efficiency)")
    args = parser.parse_args()

    run_training(epochs=args.epochs, batch_size=args.batch_size, max_samples=args.max_samples)
