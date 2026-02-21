import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from preprocess import load_data, add_rul_column, scale_data, create_sequences
from model import build_lstm_model


# Reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)


def train_model():

    train, test, rul = load_data(
        "data/train_FD001.txt",
        "data/test_FD001.txt",
        "data/RUL_FD001.txt"
    )

    train = add_rul_column(train)
    X_scaled, y, scaler = scale_data(train)

    X_seq, y_seq = create_sequences(X_scaled, y, sequence_length=30)

    # Train-validation split
    split = int(0.8 * len(X_seq))
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_seq[:split], y_seq[split:]

    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=64,
        callbacks=[early_stop]
    )

    model.save("model_lstm.keras")

    return history


if __name__ == "__main__":
    train_model()