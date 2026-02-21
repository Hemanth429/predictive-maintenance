import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from preprocess import load_data, add_rul_column, scale_data, create_sequences


def evaluate_model():

    train, test, rul = load_data(
        "data/train_FD001.txt",
        "data/test_FD001.txt",
        "data/RUL_FD001.txt"
    )

    train = add_rul_column(train)
    X_scaled, y, scaler = scale_data(train)
    X_seq, y_seq = create_sequences(X_scaled, y, sequence_length=30)

    model = tf.keras.models.load_model("model_lstm.keras")

    predictions = model.predict(X_seq)

    rmse = np.sqrt(mean_squared_error(y_seq, predictions))
    print("Test RMSE:", rmse)

    plt.figure()
    plt.scatter(y_seq[:200], predictions[:200])
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("True vs Predicted RUL (Sample)")
    plt.show()


if __name__ == "__main__":
    evaluate_model()