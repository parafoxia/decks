import tensorflow as tf

import decks
from decks.nets.utils import cp_callback


def train(epochs, nid, n_gpus, *, train_ds, val_ds):
    cp_path = decks.DATA_DIR / f"checkpoints/{nid}" / "cp-{epoch:04d}.ckpt"

    if n_gpus >= 2:
        strategy = tf.distribute.MirroredStrategy()
        net = decks.build_dist_net(train_ds, strategy)
    else:
        net = decks.build_net(train_ds)

    history = net.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[cp_callback(cp_path, 10 * len(train_ds))],
        verbose=2,
    )

    return net, history.history
