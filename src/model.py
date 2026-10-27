"""
Bidirectional LSTM (BiLSTM) Model Architecture
Deep Learning Based Fake News Detection System (BCA Final Year Project)

Architecture Details:
1. Embedding Layer: Maps word indices into dense continuous vector space.
2. SpatialDropout1D: Regularization to reduce inter-word feature correlation.
3. Bidirectional LSTM:
   - Forward LSTM reads sequence from left-to-right (past context).
   - Backward LSTM reads sequence from right-to-left (future context).
   - Both hidden states concatenated to capture rich contextual dependencies.
4. Dense Layer with ReLU: Non-linear feature combination.
5. Dropout Layer: Drops random neurons to prevent overfitting.
6. Sigmoid Output Layer: Outputs binary classification probability:
   - Probability >= 0.5 -> Label 1 (Real News)
   - Probability < 0.5  -> Label 0 (Fake News)
"""

import io
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_bilstm_model(
    vocab_size: int = 15000,
    embedding_dim: int = 100,
    maxlen: int = 200,
    lstm_units: int = 64,
    dense_units: int = 32,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001
) -> Sequential:
    """
    Constructs and compiles the Bidirectional LSTM model for Fake News classification.
    """
    model = Sequential([
        # 1. Word Embedding Layer
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name="word_embedding"
        ),

        # 2. Spatial Dropout (regularizes embeddings across word dimensions)
        SpatialDropout1D(dropout_rate, name="spatial_dropout"),

        # 3. Bidirectional LSTM Layer
        Bidirectional(
            LSTM(
                units=lstm_units,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate,
                return_sequences=False
            ),
            name="bidirectional_lstm"
        ),

        # 4. Fully Connected (Dense) Layer
        Dense(dense_units, activation="relu", name="dense_features"),

        # 5. Dropout Layer to avoid overfitting
        Dropout(dropout_rate, name="dense_dropout"),

        # 6. Binary Output Layer (Sigmoid gives probability between 0 and 1)
        Dense(1, activation="sigmoid", name="classification_output")
    ], name="FakeNews_BiLSTM_Classifier")

    # Compile with Binary Cross-Entropy loss and Adam optimizer
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def get_model_summary(model: Sequential) -> str:
    """Returns the model summary as a readable text string."""
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    return stream.getvalue()


if __name__ == "__main__":
    test_model = build_bilstm_model()
    test_model.summary()
