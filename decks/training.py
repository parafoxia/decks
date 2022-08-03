import json
import tensorflow as tf

import decks
from decks.nets.utils import cp_callback


def train(epochs, initial_epoch, nid, n_gpus, *, train_ds, val_ds):
    cp_path = decks.DATA_DIR / f"checkpoints/{nid}" / "cp-{epoch:04d}.ckpt"

    if n_gpus >= 2:
        print("Distributed training will be enabled")
        strategy = tf.distribute.MirroredStrategy()
        net = decks.build_dist_net(train_ds, strategy)
    else:
        net = decks.build_net(train_ds)

    if initial_epoch > 0:
        parts = nid.split("-")
        parts[1] = f"{initial_epoch:>04}"
        nid = "-".join(parts)

        print(f"Loading weights for {nid}...", end="")
        weights = decks.DATA_DIR / f"checkpoints/{nid}/cp-{initial_epoch:>04}.ckpt"
        net.load_weights(weights)
        print(" done")

    history = net.fit(
        train_ds,
        epochs=epochs,
        initial_epoch=initial_epoch,
        validation_data=val_ds,
        callbacks=[cp_callback(cp_path, 10 * len(train_ds))],
        verbose=2,
    )

    net_history = history.history

    if initial_epoch > 0:
        with open(decks.DATA_DIR / f"history/{nid}.json") as f:
            prior_history = json.load(f)

        for k in net_history.keys():
            net_history[k] = prior_history[k] + net_history[k]

    return net, net_history
