import pandas as pd
import tensorflow as tf


def load_carer(batch_size):
    df = pd.read_csv("carer.csv", index_col=0)
    train_df = df[df["split"] == 0]
    test_df = df[df["split"] == 1]
    val_df = df[df["split"] == 2]

    datasets = []

    for df in train_df, test_df, val_df:
        features = df.iloc[:, 0]
        labels = df.iloc[:, 1]
        ds = tf.data.Dataset.from_tensor_slices((features.values, labels.values))
        ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        datasets.append(ds)

    return datasets


if __name__ == "__main__":
    train_ds, test_ds, val_ds = load_carer(64)
    print(len(train_ds), len(test_ds), len(val_ds))
