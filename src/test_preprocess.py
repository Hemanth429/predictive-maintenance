from preprocess import load_data, add_rul_column, scale_data, create_sequences

train, test, rul = load_data(
    "data/train_FD001.txt",
    "data/test_FD001.txt",
    "data/RUL_FD001.txt"
)

train = add_rul_column(train)
X_scaled, y, scaler = scale_data(train)
X_seq, y_seq = create_sequences(X_scaled, y, sequence_length=30)

print("X_seq shape:", X_seq.shape)
print("y_seq shape:", y_seq.shape)