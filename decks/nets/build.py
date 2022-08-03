import tensorflow as tf
from decks.nets.utils import text_encoder


def build_net(ds):
    enc = text_encoder(ds, 2500)
    model = tf.keras.Sequential(
        [
            enc,
            tf.keras.layers.Embedding(
                input_dim=len(enc.get_vocabulary()),
                output_dim=64,
                mask_zero=True,
            ),
            tf.keras.layers.Dropout(0.8),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64)),
            tf.keras.layers.Dropout(0.7),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.6),
            tf.keras.layers.Dense(6, activation="softmax"),
        ]
    )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def build_dist_net(ds, strategy):
    with strategy.scope():
        enc = text_encoder(ds, 2500)
        model = tf.keras.Sequential(
            [
                enc,
                tf.keras.layers.Embedding(
                    input_dim=len(enc.get_vocabulary()),
                    output_dim=64,
                    mask_zero=True,
                ),
                tf.keras.layers.Dropout(0.8),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64)),
                tf.keras.layers.Dropout(0.7),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.6),
                tf.keras.layers.Dense(6, activation="softmax"),
            ]
        )
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )

    return model
