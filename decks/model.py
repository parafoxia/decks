import tensorflow as tf

import decks
from decks import data


def _load_net():
    # Loads C6-10 classifier.
    ds, _, _ = data.load_carer(512)
    net = decks.build_net(ds)
    weights = decks.DATA_DIR / "checkpoints/2006-0100-0512/cp-0010.ckpt"
    net.load_weights(weights).expect_partial()
    # custom_objects = {"_standardise": _standardise}
    # with tf.keras.utils.custom_object_scope(custom_objects):
    #     net = tf.keras.models.load_model("DecksC6")
    return net


class DecksNet(tf.keras.Model):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.emotion = _load_net()
        self.context = decks.Contextualiser(alpha=alpha)

    def call(self, inputs):
        x = self.emotion(inputs)
        print(x)
        x = self.context(x)
        return x
