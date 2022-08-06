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


class Model(tf.keras.Model):
    def __init__(self, ds):
        super().__init__()

        self.encoder = text_encoder(ds, 2_500)
        self.embedding = tf.keras.layers.Embedding(input_dim=len(self.encoder.get_vocabulary()), output_dim=64, mask_zero=True)

        self.b1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64), return_state=True, return_sequences=True)
        self.b2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64), return_state=True, return_sequences=True)
        self.b3 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64))

        self.d1 = tf.keras.layers.Dense(64, activation="relu")
        self.d2 = tf.keras.layers.Dense(64, activation="relu")
        self.d3 = tf.keras.layers.Dense(64, activation="relu")

        self.out = tf.keras.layers.Dense(28, activation="sigmoid")

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.embedding(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.out(x)


def build_net(ds):
    model = Model(ds)
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=["accuracy"],
    )
    return model


def build_dist_net(ds, strategy):
    with strategy.scope():
        model = build_net(ds)
    return model
