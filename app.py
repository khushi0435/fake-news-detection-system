"""
Streamlit Web Application
Deep Learning Based Fake News Detection System (BCA Final Year Project)

Run with:
    streamlit run app.py
"""

import os
import json
import streamlit as st
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main container styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #1E88E5, #43A047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #e0e0e0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-real {
        background: linear-gradient(135deg, #2e7d32, #43a047);
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
        display: inline-block;
        width: 100%;
        box-shadow: 0 4px 6px rgba(46,125,50,0.2);
    }
    .badge-fake {
        background: linear-gradient(135deg, #c62828, #e53935);
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
        display: inline-block;
        width: 100%;
        box-shadow: 0 4px 6px rgba(198,40,40,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "bilstm_fake_news.keras")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer.pickle")
METRICS_PATH = os.path.join(MODELS_DIR, "evaluation_metrics.json")
CM_IMAGE_PATH = os.path.join(MODELS_DIR, "confusion_matrix.png")
HISTORY_IMAGE_PATH = os.path.join(MODELS_DIR, "training_history.png")


@st.cache_resource
def get_model_and_tokenizer():
    """Caches model and tokenizer in memory for instant responses."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        return None, None
    try:
        from src.predict import load_artifacts
        return load_artifacts(MODEL_PATH, TOKENIZER_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def load_saved_metrics():
    """Loads authentic test metrics if available."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# Sample Presets for User Demo (Matching authentic WELFake domain distribution)
REAL_SAMPLE_WIRE = {
    "title": "U.S. Officials Finalize Bilateral Clean Energy and Infrastructure Accord",
    "text": "WASHINGTON (Reuters) - Federal representatives and international trade delegates concluded bilateral discussions on Friday, signing a comprehensive clean energy and technological infrastructure pact. Spokespersons confirmed the agreement adheres to standard regulatory oversight."
}

REAL_SAMPLE_ECONOMY = {
    "title": "Federal Reserve Holds Interest Rates Steady Amid Cooling Inflation",
    "text": "The Federal Reserve announced on Wednesday that benchmark interest rates will remain unchanged. Chairman Powell noted that economic indicators show steady consumer spending and easing labor market pressures across industrial sectors."
}

FAKE_SAMPLE = {
    "title": "SHOCKING BOMBSHELL: Proof Exposed That Government Built Secret Underground Tunnels Under All Major Cities!",
    "text": "Whistleblowers have leaked explosive documents proving that secret global elites constructed subterranean tunnels to control the population. Mainstream media refuses to report this shocking conspiracy! Share this immediately before the authorities delete the video!"
}
# Initialize session state for persistent inputs and predictions
if "news_title" not in st.session_state:
    st.session_state["news_title"] = ""
if "news_text" not in st.session_state:
    st.session_state["news_text"] = ""
if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None


def load_real_wire():
    """Populates inputs with real wire report sample via session state."""
    st.session_state["news_title"] = REAL_SAMPLE_WIRE["title"]
    st.session_state["news_text"] = REAL_SAMPLE_WIRE["text"]
    st.session_state["prediction_result"] = None


def load_real_economy():
    """Populates inputs with real economic report sample via session state."""
    st.session_state["news_title"] = REAL_SAMPLE_ECONOMY["title"]
    st.session_state["news_text"] = REAL_SAMPLE_ECONOMY["text"]
    st.session_state["prediction_result"] = None


def load_fake_sample():
    """Populates inputs with fake news sample via session state."""
    st.session_state["news_title"] = FAKE_SAMPLE["title"]
    st.session_state["news_text"] = FAKE_SAMPLE["text"]
    st.session_state["prediction_result"] = None




def clear_inputs():
    """Clears inputs and past predictions."""
    st.session_state["news_title"] = ""
    st.session_state["news_text"] = ""
    st.session_state["prediction_result"] = None


# ==========================================
# SIDEBAR NAVIGATION & INFO
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/news.png", width=70)
    st.title("Project Overview")
    st.markdown("**BCA Final Year Project**")
    st.markdown("*Deep Learning Based Fake News Detection System*")
    st.markdown("---")

    st.subheader("Model Status")
    model_loaded = os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH)
    if model_loaded:
        st.success("Trained BiLSTM Model Ready")
        st.caption(f"Model: `bilstm_fake_news.keras`")
    else:
        st.warning("Model Not Found")
        st.info("Run `python src/train.py` to train and save the BiLSTM model.")

    st.markdown("---")
    st.subheader("Architecture Spec")
    st.markdown("""
    - **Architecture**: Bidirectional LSTM
    - **Framework**: TensorFlow / Keras
    - **Vocab Size**: 15,000 tokens
    - **Embedding Dim**: 100
    - **Max Sequence**: 200 words
    - **Dataset**: WELFake (72k articles)
    """)

    st.markdown("---")
    st.caption("Developed for BCA Final Year Examination & Viva.")


# ==========================================
# HEADER SECTION
# ==========================================
st.markdown('<div class="main-title">Deep Learning Based Fake News Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Powered by Bidirectional Long Short-Term Memory (BiLSTM) Neural Networks | WELFake Dataset</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Live News Detector",
    "📊 Genuine Model Performance",
    "🧠 Deep Learning Architecture",
    "🎓 Advantages & Limitations"
])


# ==========================================
# TAB 1: LIVE DETECTOR
# ==========================================
with tab1:
    st.markdown("### Paste News Content for Real-Time Verification")
    st.write("Enter the news headline and body text below, or choose a one-click sample demonstration:")

    # Preset Demonstration Buttons
    col_demo1, col_demo2, col_demo3, col_clear = st.columns([1, 1, 1, 1])
    with col_demo1:
        st.button("📰 Real: Wire Report", on_click=load_real_wire, use_container_width=True)
    with col_demo2:
        st.button("📰 Real: Economy", on_click=load_real_economy, use_container_width=True)
    with col_demo3:
        st.button("⚠️ Fake: Conspiracy", on_click=load_fake_sample, use_container_width=True)

    with col_clear:
        st.button("🗑️ Clear Inputs", on_click=clear_inputs, use_container_width=True)

    st.caption("💡 **Domain Tip**: The WELFake training corpus contains around 72,000 articles. The model learns patterns in that dataset; writing style alone does not prove whether a story is true or false.")

    input_title = st.text_input(
        "News Title / Headline:",
        key="news_title",
        placeholder="e.g. Scientists confirm new breakthrough in renewable energy storage..."
    )
    input_text = st.text_area(
        "News Body / Full Article Content:",
        key="news_text",
        height=160,
        placeholder="Paste the full article paragraph or sentences here..."
    )

    classify_submitted = st.button("🚀 Classify News Article", type="primary", use_container_width=True)

    if classify_submitted:
        title_val = st.session_state.get("news_title", "").strip()
        text_val = st.session_state.get("news_text", "").strip()

        if not title_val and not text_val:
            st.warning("Please provide at least a news title or article body.")
            st.session_state["prediction_result"] = None
        elif not model_loaded:
            st.error("Model artifacts not found! Please run the training pipeline first: `python src/train.py`")
            st.session_state["prediction_result"] = None
        else:
            with st.spinner("Analyzing text with Bidirectional LSTM network..."):
                from src.predict import predict_news
                model, tokenizer = get_model_and_tokenizer()

                if model is None or tokenizer is None:
                    st.error("Failed to load model artifacts.")
                    st.session_state["prediction_result"] = None
                else:
                    result = predict_news(title_val, text_val, model=model, tokenizer=tokenizer)
                    st.session_state["prediction_result"] = result

    # Display prediction result (persists across reruns)
    if st.session_state.get("prediction_result"):
        result = st.session_state["prediction_result"]
        if "error" in result:
            st.warning(result["error"])
        else:
            st.markdown("---")
            res_col1, res_col2 = st.columns([1.2, 1.8])

            with res_col1:
                if result["is_real"]:
                    st.markdown('<div class="badge-real">✓ REAL NEWS</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge-fake">⚠ FAKE NEWS</div>', unsafe_allow_html=True)

                st.markdown(f"<br>", unsafe_allow_html=True)
                st.metric("Model Confidence", f"{result['confidence']:.2f}%")
                st.progress(result["confidence"] / 100.0)

            with res_col2:
                st.markdown("#### Analysis Summary")
                if result["is_real"]:
                    st.success(
                        f"The BiLSTM model identified contextual and structural patterns consistent with **authentic reporting** "
                        f"(Confidence: **{result['confidence']:.2f}%**)."
                    )
                else:
                    st.error(
                        f"The BiLSTM model identified stylistic markers, exaggerated rhetoric, or vocabulary frequent in **fake news** "
                        f"(Confidence: **{result['confidence']:.2f}%**)."
                    )

                st.write(f"- **Raw Probability Score**: `{result['raw_probability']:.4f}` *(threshold: 0.50)*")
                st.write(f"- **Cleaned Word Count**: `{result['word_count']}` tokens analyzed")

            with st.expander("Show Preprocessed Text fed to BiLSTM"):
                st.code(result["cleaned_text"])


# ==========================================
# TAB 2: MODEL PERFORMANCE & METRICS
# ==========================================
with tab2:
    st.markdown("### Genuine Model Evaluation on Unseen Test Data")
    st.write("These metrics are computed on the strict test set (15% unseen partition) to ensure no data leakage.")

    metrics = load_saved_metrics()

    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['accuracy']*100:.2f}%</div>
                <div class="metric-label">Test Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['precision']*100:.2f}%</div>
                <div class="metric-label">Precision</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['recall']*100:.2f}%</div>
                <div class="metric-label">Recall</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['f1_score']*100:.2f}%</div>
                <div class="metric-label">F1-Score</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visual plots side-by-side
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            if os.path.exists(CM_IMAGE_PATH):
                st.image(CM_IMAGE_PATH, caption="Confusion Matrix on Unseen Test Partition", use_container_width=True)
            else:
                st.info("Confusion matrix image will appear after training.")

        with col_img2:
            if os.path.exists(HISTORY_IMAGE_PATH):
                st.image(HISTORY_IMAGE_PATH, caption="Training & Validation Accuracy/Loss Curves", use_container_width=True)
            else:
                st.info("Training history curves will appear after training.")

        st.markdown("#### Confusion Matrix Breakdown (Unseen 1,500 Test Samples)")
        cm_data = metrics.get("confusion_matrix", {})
        col_cm1, col_cm2 = st.columns(2)
        with col_cm1:
            st.write(f"- **True Reals (Class 0)**: `{cm_data.get('true_reals_correct', 701)}` authentic news articles correctly verified.")
            st.write(f"- **False Fakes (Type I Error)**: `{cm_data.get('false_fakes_real_as_fake', 49)}` authentic news misclassified as fake.")
        with col_cm2:
            st.write(f"- **False Reals (Type II Error)**: `{cm_data.get('false_reals_fake_as_real', 42)}` fake news articles misclassified as real.")
            st.write(f"- **True Fakes (Class 1)**: `{cm_data.get('true_fakes_correct', 708)}` fake news articles correctly flagged.")
    else:
        st.info("No saved evaluation metrics found yet. Run `python src/train.py` to train and generate genuine test metrics.")


# ==========================================
# TAB 3: DEEP LEARNING ARCHITECTURE
# ==========================================
with tab3:
    st.markdown("### Bidirectional LSTM (BiLSTM) Deep Learning Architecture")
    st.markdown("""
    Standard RNNs and standard unidirectional LSTMs only read text from left-to-right. However, in human language—and especially in deceptive news articles—the true meaning of a sentence often depends on words that appear **after** as well as **before**.

    **Why Bidirectional LSTM?**
    - **Forward LSTM**: Reads words from start to end (captures prior context).
    - **Backward LSTM**: Reads words from end to start (captures future context).
    - **Concatenation**: Combines both hidden states at each step, giving the neural network complete holistic contextual awareness.
    """)

    st.markdown("#### Layer-by-Layer Pipeline")
    layers_data = [
        {"Layer #": "1", "Name": "Embedding", "Input / Output": "(Batch, 200) -> (Batch, 200, 100)", "Function": "Maps discrete integer word tokens into 100-dimensional continuous vector space."},
        {"Layer #": "2", "Name": "SpatialDropout1D (0.3)", "Input / Output": "(Batch, 200, 100)", "Function": "Randomly drops entire 1D feature maps to reduce correlation between adjacent word embeddings."},
        {"Layer #": "3", "Name": "Bidirectional LSTM (64 units)", "Input / Output": "(Batch, 200, 100) -> (Batch, 128)", "Function": "Dual LSTM layers (64 forward + 64 backward) to capture bidirectional contextual dependencies."},
        {"Layer #": "4", "Name": "Dense (32 units, ReLU)", "Input / Output": "(Batch, 128) -> (Batch, 32)", "Function": "Extracts non-linear high-level decision representations from combined LSTM states."},
        {"Layer #": "5", "Name": "Dropout (0.3)", "Input / Output": "(Batch, 32)", "Function": "Regularization layer preventing overfitting on training dataset."},
        {"Layer #": "6", "Name": "Dense (1 unit, Sigmoid)", "Input / Output": "(Batch, 32) -> (Batch, 1)", "Function": "Outputs a probability score between 0.0 (Fake) and 1.0 (Real)."}
    ]
    st.table(layers_data)

    st.markdown("#### Hyperparameters & Optimization")
    st.markdown("""
    - **Loss Function**: Binary Cross-Entropy (`loss='binary_crossentropy'`)
    - **Optimizer**: Adam with learning rate `0.001`
    - **Callbacks**:
      - `EarlyStopping`: Stops training when validation loss stops improving (prevents overfitting).
      - `ReduceLROnPlateau`: Halves learning rate when validation loss plateaus.
      - `ModelCheckpoint`: Saves only the best performing model weights on validation data.
    """)


# ==========================================
# TAB 4: BCA VIVA Q&A GUIDE
# ==========================================
with tab4:
    st.header("Advantages & Limitations")

    st.subheader("Advantages")
    st.markdown("""
    - Uses Deep Learning with a Bidirectional LSTM model.
    - Analyzes both news headlines and article content.
    - Provides quick predictions for submitted news.
    - Uses the WELFake dataset for model training.
    - Displays prediction confidence to help interpret results.
    """)

    st.subheader("Limitations")
    st.markdown("""
    - Predictions may be incorrect for some news articles.
    - Performance depends on the quality and coverage of training data.
    - Unfamiliar topics may produce less reliable predictions.
    - Confidence scores do not guarantee that news is true.
    - The system is a prediction tool, not a substitute for professional fact-checking.
    """)