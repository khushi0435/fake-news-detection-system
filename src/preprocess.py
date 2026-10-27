"""
Data Cleaning, Preprocessing and Tokenization Module
Deep Learning Based Fake News Detection System (BCA Final Year Project)

This module handles:
1. Loading raw dataset (WELFake or starter dataset).
2. Cleaning missing values and removing duplicate news entries.
3. Text normalization (lowercase, regex cleaning of URLs, HTML, punctuation).
4. Data splitting (Train/Val/Test) BEFORE fitting tokenizer (prevents data leakage).
5. Tokenization and sequence padding for deep learning ingestion.
"""

import os
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Standard English stopwords list without needing heavy external downloads
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'can', 'will', 'just', 'don', 'should', "should've", 'now', 'd', 'll', 'm',
    'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn',
    'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn',
    'weren', 'won', 'wouldn'
}


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Cleans raw news text for deep learning input:
    - Converts to lowercase
    - Removes HTML tags and URLs
    - Strips special characters and punctuation
    - Filters common stopwords (optional)
    - Normalizes multi-spaces
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags (e.g. <br>, <p>, &amp;)
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove URLs (http:// or https://)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)

    # Remove non-alphabetic characters (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Split into words
    words = text.split()

    if remove_stopwords:
        words = [w for w in words if w not in STOPWORDS and len(w) > 2]

    return " ".join(words)


def load_and_clean_dataset(csv_path: str, max_samples: int = None) -> pd.DataFrame:
    """
    Loads dataset, drops missing/duplicate records, and creates unified 'content' column.
    
    WELFake columns:
    - title: Headline of the news
    - text: Main body
    - label: 0 for Fake, 1 for Real
    """
    print(f"Loading raw data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Check required columns
    if 'label' not in df.columns:
        raise ValueError("Dataset must contain a 'label' column (0 for Fake, 1 for Real).")

    # Combine title and body text to maximize contextual information
    df['title'] = df['title'].fillna("") if 'title' in df.columns else ""
    df['text'] = df['text'].fillna("") if 'text' in df.columns else ""

    # Create raw combined content
    df['combined_text'] = (df['title'] + " " + df['text']).str.strip()

    # Filter out empty or whitespace-only articles
    initial_count = len(df)
    df = df[df['combined_text'].str.len() > 20].copy()

    # Drop duplicate articles
    df.drop_duplicates(subset=['combined_text'], inplace=True)

    # Filter valid binary labels (0 or 1)
    df = df[df['label'].isin([0, 1])].copy()
    df['label'] = df['label'].astype(int)

    # If dataset is very large (e.g. 72,000 rows) and max_samples is set, sample proportionally
    if max_samples and len(df) > max_samples:
        print(f"Sampling {max_samples} balanced rows from {len(df)} total rows for training efficiency...")
        sample_per_class = max_samples // 2
        df_real = df[df['label'] == 1].sample(n=min((df['label'] == 1).sum(), sample_per_class), random_state=42)
        df_fake = df[df['label'] == 0].sample(n=min((df['label'] == 0).sum(), sample_per_class), random_state=42)
        df = pd.concat([df_real, df_fake], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Dataset cleaned: {len(df)} articles remaining (from initial {initial_count}).")
    real_count = (df['label'] == 1).sum()
    fake_count = (df['label'] == 0).sum()
    print(f"Distribution: Real={real_count} ({real_count/len(df)*100:.1f}%), Fake={fake_count} ({fake_count/len(df)*100:.1f}%)")

    # Clean text content
    print("Preprocessing and cleaning text content...")
    df['clean_text'] = df['combined_text'].apply(clean_text)

    # Drop any that became blank after cleaning
    df = df[df['clean_text'].str.strip().str.len() > 10].copy()
    return df[['clean_text', 'label']]


def split_data(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15, random_state: int = 42):
    """
    Performs stratified split into Train, Validation, and Test sets.
    CRITICAL: Split happens BEFORE Tokenizer is fitted to prevent data leakage!
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    X = np.array(df['clean_text'].tolist(), dtype=object)
    y = np.array(df['label'].tolist(), dtype=int)

    # Step 1: Split into train and temp (val + test)
    val_test_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=val_test_ratio,
        stratify=y,
        random_state=random_state
    )

    # Step 2: Split temp into val and test
    test_portion_of_temp = test_ratio / val_test_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=test_portion_of_temp,
        stratify=y_temp,
        random_state=random_state
    )

    print(f"Data Split Summary:")
    print(f" - Train set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
    print(f" - Val set:   {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
    print(f" - Test set:  {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def fit_and_save_tokenizer(train_texts, save_path: str, num_words: int = 15000):
    """
    Fits Keras Tokenizer EXCLUSIVELY on training data to prevent data leakage.
    Saves tokenizer to disk using pickle.
    """
    # Import locally so module can be inspected without immediate tf import if needed
    from tensorflow.keras.preprocessing.text import Tokenizer

    print(f"Fitting Tokenizer on {len(train_texts)} training samples (vocabulary size={num_words})...")
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[OK] Tokenizer saved successfully at: {save_path}")
    return tokenizer


def load_tokenizer(load_path: str):
    """Loads a previously saved Tokenizer."""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Tokenizer file not found at: {load_path}")
    with open(load_path, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer


def tokenize_and_pad(tokenizer, texts, maxlen: int = 200):
    """
    Converts list of texts to integer sequences and applies post-padding.
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
    return padded
