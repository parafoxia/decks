import tensorflow as tf


def text_encoder(ds, max_tokens):
    enc = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=max_tokens
    )
    enc.adapt(ds.map(lambda text, _: text))
    return enc


def cp_callback(path, save_freq):
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=path,
        save_weights_only=True,
        save_freq=save_freq,
    )
