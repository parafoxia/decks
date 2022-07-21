import pandas as pd
import tensorflow as tf


def load_carer(batch_size):
    df = pd.read_csv("carer.csv", index_col=0)
    rows = len(df)

    features = df.iloc[:, 0]
    labels = df.iloc[:, 1]
    ds = tf.data.Dataset.from_tensor_slices((features.values, labels.values))
    ds = ds.shuffle(rows)

    train_ds = ds.take(rows - 50_000)
    test_ds = ds.skip(rows - 50_000)
    val_ds = test_ds.skip(25_000)
    test_ds = test_ds.take(25_000)

    return (
        train_ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE),
        test_ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE),
        val_ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE),
    )



if __name__ == "__main__":
    train_ds, test_ds, val_ds = load_carer(64)
    print(len(train_ds), len(test_ds), len(val_ds))
