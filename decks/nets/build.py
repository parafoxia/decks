import tensorflow as tf
import tensorflow_addons as tfa

from decks.nets import metrics
from decks.nets.utils import text_encoder


def calculate_hidden_nodes(N, m):
    p1 = ((m + 2) * N) ** 0.5
    p2 = 2 * ((N / (m + 2)) ** 0.5)
    p3 = m * ((N / (m + 2)) ** 0.5)
    return round((p1 + p2) * 0.75), round(p3 * 0.75)


def build_net(ds):
    outputs = 6
    h1, h2 = calculate_hidden_nodes(len(ds), outputs)
    print(f"Using {h1}:{h2} for hidden layers")

    enc = text_encoder(ds, None)
    model = tf.keras.Sequential(
        [
            enc,
            tf.keras.layers.Embedding(
                input_dim=len(enc.get_vocabulary()),
                output_dim=h1,
                mask_zero=True,
            ),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(h1, dropout=0.5)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(h2),
            tfa.layers.Maxout(5),
            tf.keras.layers.Dense(outputs, activation="softmax"),
        ]
    )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=[
            "accuracy",
            *[metrics.PrecisionForClass(x) for x in range(outputs)],
            *[metrics.RecallForClass(x) for x in range(outputs)],
            *[metrics.MccForClass(x) for x in range(outputs)],
            *[metrics.F1ForClass(x) for x in range(outputs)],
        ],
    )
    return model


def build_dist_net(ds, strategy):
    with strategy.scope():
        model = build_net(ds)
    return model
