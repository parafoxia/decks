import tensorflow as tf

from decks.nets.utils import text_encoder


def calculate_nodes_for(ds, m):
    N = len(ds)
    p1 = ((m + 2) * N) ** 0.5
    p2 = 2 * ((N / (m + 2)) ** 0.5)
    p3 = m * ((N / (m + 2)) ** 0.5)
    return round((p1 + p2) * .75), round(p3 * .75)


def build_net(ds):
    hl1, hl2 = calculate_nodes_for(ds, 6)
    print(f"Using \33[1m{hl1}:{hl2}\33[0m for hidden node counts")

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
            tf.keras.layers.Dense(6, activation="softmax"),
        ]
    )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(2e-5),
        metrics=["accuracy"],
    )
    return model


# def build_net(ds):
#     hl1, hl2, outputs = calculate_nodes_for(ds)
#     enc = text_encoder(ds, None)
#     model = tf.keras.Sequential(
#         [
#             enc,
#             tf.keras.layers.Embedding(
#                 input_dim=len(enc.get_vocabulary()),
#                 output_dim=hl1,
#                 mask_zero=True,
#             ),
#             tf.keras.layers.Bidirectional(tf.keras.layers.GRU(hl1)),
#             tf.keras.layers.Dense(hl2, activation="relu"),
#             tf.keras.layers.Dense(outputs, activation="sigmoid"),
#         ]
#     )
#     model.compile(
#         loss="binary_crossentropy",
#         # optimizer=tf.keras.optimizers.Adam(1e-4),
#         optimizer="adam",
#         metrics=["accuracy"],
#     )
#     return model


def build_dist_net(ds, strategy):
    with strategy.scope():
        model = build_net(ds)
    return model
