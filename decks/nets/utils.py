import re

import tensorflow as tf
from stopwordsiso import stopwords


def _standardise(inputs):
    x = tf.strings.lower(inputs)
    x = tf.strings.regex_replace(x, "[^a-z0-9 ]", "")
    x = tf.strings.regex_replace(x, " +", " ")
    sw = [re.sub("[^a-z0-9]", "", w) for w in stopwords("en")]
    x = tf.strings.regex_replace(x, r"\b(" + r"|".join(sw) + r")\b\s*", " ")
    return x


def text_encoder(ds, max_tokens):
    enc = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=max_tokens,
        standardize=_standardise,
    )
    enc.adapt(ds.map(lambda text, _: text))
    return enc


def cp_callback(path, save_freq):
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=path,
        save_weights_only=True,
        save_freq=save_freq,
    )
