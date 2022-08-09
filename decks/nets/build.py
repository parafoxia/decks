import tensorflow as tf

from decks.nets.utils import text_encoder


# def build_net(ds):
#     enc = text_encoder(ds, 2_500)
#     model = tf.keras.Sequential(
#         [
#             enc,
#             tf.keras.layers.Embedding(
#                 input_dim=len(enc.get_vocabulary()),
#                 output_dim=64,
#                 mask_zero=True,
#             ),
#             tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64)),
#             tf.keras.layers.Dense(64, activation="relu"),
#             tf.keras.layers.Dense(6, activation="softmax"),
#         ]
#     )
#     model.compile(
#         loss="sparse_categorical_crossentropy",
#         optimizer=tf.keras.optimizers.Adam(1e-4),
#         metrics=["accuracy"],
#     )
#     return model


def calculate_nodes_for(ds, N):
    for _, l in ds.take(1):
        l = l.numpy()
        m = len(l[0]) if l.shape[1] > 1 else len(l)

    p1 = ((m + 2) * N) ** 0.5
    p2 = 2 * ((N / (m + 2)) ** 0.5)
    p3 = m * ((N / (m + 2)) ** 0.5)
    return round((p1 + p2) * .9), round(p3 * .9), m


def build_net(ds, n_samples):
    hl1, hl2, outputs = calculate_nodes_for(ds, n_samples)
    enc = text_encoder(ds, None)
    model = tf.keras.Sequential(
        [
            enc,
            tf.keras.layers.Embedding(
                input_dim=len(enc.get_vocabulary()),
                output_dim=hl1,
                mask_zero=True,
            ),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(hl1)),
            tf.keras.layers.Dense(hl2, activation="relu"),
            tf.keras.layers.Dense(outputs, activation="sigmoid"),
        ]
    )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=["accuracy"],
    )
    return model


def build_dist_net(ds, n_samples, strategy):
    with strategy.scope():
        model = build_net(ds, n_samples)
    return model
