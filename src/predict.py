"""
Inference and Prediction Module
Deep Learning Based Fake News Detection System (BCA Final Year Project)

Provides high-level prediction functions for:
- Streamlit web interface
- Command-line interactive testing
- Batch evaluation
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import clean_text, load_tokenizer, tokenize_and_pad

# Global cache for loaded model & tokenizer
_CACHED_MODEL = None
_CACHED_TOKENIZER = None


def load_artifacts(
    model_path: str = os.path.join(PROJECT_ROOT, "models", "bilstm_fake_news.keras"),
    tokenizer_path: str = os.path.join(PROJECT_ROOT, "models", "tokenizer.pickle")
):
    """
    Loads and caches the trained BiLSTM model and fitted Tokenizer.
    Avoids re-loading artifacts on every prediction call.
    """
    global _CACHED_MODEL, _CACHED_TOKENIZER

    if _CACHED_MODEL is None or _CACHED_TOKENIZER is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. Please run 'python src/train.py' first!"
            )
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer file not found at '{tokenizer_path}'. Please run 'python src/train.py' first!"
            )

        from tensorflow.keras.models import load_model

        print(f"Loading trained model from {model_path}...")
        _CACHED_MODEL = load_model(model_path)
        print(f"Loading tokenizer from {tokenizer_path}...")
        _CACHED_TOKENIZER = load_tokenizer(tokenizer_path)

    return _CACHED_MODEL, _CACHED_TOKENIZER


def predict_news(
    title: str = "",
    text: str = "",
    model=None,
    tokenizer=None,
    maxlen: int = 200,
    threshold: float = 0.5
) -> dict:
    """
    Predicts whether a given news article (title + text) is Real or Fake.
    
    Returns:
        dict containing:
        - 'is_real': bool
        - 'prediction': 'Real News' or 'Fake News'
        - 'label': 1 (Real) or 0 (Fake)
        - 'confidence': percentage float (0.0 to 100.0)
        - 'raw_probability': float (0.0 to 1.0)
        - 'cleaned_text': preprocessed string
        - 'word_count': count of cleaned words
    """
    # Load default artifacts if not provided
    if model is None or tokenizer is None:
        model, tokenizer = load_artifacts()

    # Combine title and text
    raw_combined = (title.strip() + " " + text.strip()).strip()

    if not raw_combined:
        return {
            "error": "Input text is completely empty. Please enter title or news content."
        }

    # Preprocess text
    cleaned = clean_text(raw_combined, remove_stopwords=True)
    words = cleaned.split()

    if len(words) < 3:
        return {
            "error": "Text is too short after preprocessing to reliably detect fake news. Please enter at least a few words."
        }

    # Tokenize and pad
    padded_seq = tokenize_and_pad(tokenizer, [cleaned], maxlen=maxlen)

    # Model inference (WELFake convention: 0 = Real News, 1 = Fake News)
    raw_prob = float(model.predict(padded_seq, verbose=0)[0][0])

    is_fake = raw_prob >= threshold
    is_real = not is_fake
    label = 1 if is_fake else 0
    prediction = "Fake News" if is_fake else "Real News"

    # Confidence calculation:
    # If prob is 0.98 -> 98% confident Fake News
    # If prob is 0.02 -> 98% confident Real News (1 - 0.02)
    confidence = (raw_prob if is_fake else (1.0 - raw_prob)) * 100.0

    return {
        "is_real": is_real,
        "is_fake": is_fake,
        "prediction": prediction,
        "label": label,
        "confidence": round(confidence, 2),
        "raw_probability": round(raw_prob, 4),
        "cleaned_text": cleaned,
        "word_count": len(words)
    }


if __name__ == "__main__":
    # Test 1: Real News (Formal journalistic report)
    real_title = "Federal Reserve Holds Interest Rates Steady Amid Cooling Inflation"
    real_body = "The Federal Reserve announced on Wednesday that benchmark interest rates will remain unchanged. Chairman Powell noted that economic indicators show steady consumer spending and easing labor market pressures across industrial sectors."
    
    # Test 2: Fake News (Conspiracy clickbait)
    fake_title = "SHOCKING BOMBSHELL: Proof Exposed That Government Built Secret Underground Tunnels Under All Major Cities!"
    fake_body = "Whistleblowers have leaked explosive documents proving that secret global elites constructed subterranean tunnels to control the population. Mainstream media refuses to report this shocking conspiracy! Share this immediately before the authorities delete the video!"

    try:
        print("\n--- Testing Real News Inference ---")
        res_real = predict_news(real_title, real_body)
        print(f"Prediction: {res_real['prediction']} | Confidence: {res_real['confidence']}% | Probability: {res_real['raw_probability']}")

        print("\n--- Testing Fake News Inference ---")
        res_fake = predict_news(fake_title, fake_body)
        print(f"Prediction: {res_fake['prediction']} | Confidence: {res_fake['confidence']}% | Probability: {res_fake['raw_probability']}")
    except Exception as e:
        print(f"Prediction test note: {e}")
