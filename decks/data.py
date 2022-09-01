import numpy as np
import pandas as pd
import tensorflow as tf


def _calculate_weights_carer(labels):
    samples = np.zeros(labels.max() + 1)

    for i, count in labels.value_counts().items():
        samples[i] = count

    weights = np.max(samples) / samples
    # weights = np.interp(weights, (weights.min(), weights.max()), (1, 10))
    return {i: w for i, w in enumerate(weights)}


def load_carer(batch_size):
    df = pd.read_csv("carer.csv", index_col=0)
    datasets = []

    for split in range(3):
        split_df = df[df["split"] == split]
        features = split_df.iloc[:, 0]
        labels = split_df.iloc[:, 1]

        ds = tf.data.Dataset.from_tensor_slices((features.values, labels.values))
        ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        datasets.append(ds)

    return datasets


def _calculate_weights_goemotions(labels):
    samples = np.zeros(len(labels.columns))

    for i, col in enumerate(labels.columns):
        samples[i] = labels[col].value_counts()[1]

    weights = np.max(samples) / samples
    weights = np.interp(weights, (weights.min(), weights.max()), (1, 10))
    return {i: w for i, w in enumerate(weights)}


def load_goemotions(batch_size):
    df = pd.read_csv("goemotions.csv", index_col=0)
    train_df = df[df["split"] == 0]
    test_df = df[df["split"] == 1]
    val_df = df[df["split"] == 2]

    datasets = []

    for i, df in enumerate((train_df, test_df, val_df)):
        features = df.iloc[:, 0]
        labels = df.iloc[:, 2:]

        if i == 0:
            weights = _calculate_weights_goemotions(labels)

        ds = tf.data.Dataset.from_tensor_slices((features.values, labels.values))
        ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        datasets.append(ds)

    return datasets, weights


if __name__ == "__main__":
    train_ds, test_ds, val_ds = load_carer(512)
    print(len(train_ds), len(test_ds), len(val_ds))
    print(len(train_ds) * 512)
