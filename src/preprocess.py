import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_data(train_path, test_path, rul_path):
    train = pd.read_csv(train_path, sep="\s+", header=None)
    test = pd.read_csv(test_path, sep="\s+", header=None)
    rul = pd.read_csv(rul_path, header=None)

    # Drop last two empty columns
    train = train.dropna(axis=1, how='all')
    test = test.dropna(axis=1, how='all')

    return train, test, rul


def add_rul_column(train):
    train.columns = ['unit', 'cycle'] + [f'op{i}' for i in range(1,4)] + \
                    [f'sensor{i}' for i in range(1,22)]

    max_cycle = train.groupby('unit')['cycle'].max()
    train['RUL'] = train.apply(lambda row: max_cycle[row['unit']] - row['cycle'], axis=1)
    train['RUL'] = train['RUL'].clip(upper=125)

    return train


def scale_data(train):
    features = train.drop(columns=['unit', 'cycle', 'RUL'])
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    return scaled_features, train['RUL'].values, scaler


def create_sequences(X, y, sequence_length=30):
    X_seq = []
    y_seq = []

    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])

    return np.array(X_seq), np.array(y_seq)