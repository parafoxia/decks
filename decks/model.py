import tensorflow as tf

import decks
from decks import data


def _load_net(id):
    train_ds, _, _ = data.load_carer(int(id.split("-")[-1]))
    net = decks.build_net(train_ds)
    weights = tf.train.latest_checkpoint(decks.DATA_DIR / f"checkpoints/{id}")
    net.load_weights(weights)
    return net


class DecksNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.emo_net = _load_net("0002-0010-0512")
        self.deca_layer = decks.Contextualiser()

    def call(self, inputs):
        x = self.emo_net(inputs)
        x = self.deca_layer(x)
        return x
