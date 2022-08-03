import argparse
import json
import os
import datetime as dt
import tensorflow as tf

import decks

def cli():
    if not decks.DATA_DIR.is_dir():
        os.makedirs(decks.DATA_DIR / "checkpoints")
        os.makedirs(decks.DATA_DIR / "history")

    parser = argparse.ArgumentParser()
    parser.add_argument("generation", help="The model's generation.")
    parser.add_argument("-e", "--epochs", help="The number of epochs to train for.", type=int)
    parser.add_argument("-b", "--batch-size", help="The batch size to use.", type=int, default=64)
    ns = parser.parse_args()

    if ns.epochs % 10 != 0:
        raise ValueError("the number of epochs must be a multiple of 10")

    n_gpus = len(tf.config.list_physical_devices("GPU"))
    if n_gpus >= 2:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = None

    train_ds, test_ds, val_ds = decks.load_carer(ns.batch_size)

    net_id = f"{ns.generation:>04}-{ns.epochs:04}-{ns.batch_size:04}"
    print(f"\nNow training {net_id}... ({n_gpus} GPUs available)")
    if strategy:
        print("Using distributed strategy")

    start = dt.datetime.utcnow()
    net, history = decks.train(ns.epochs, net_id, train_ds=train_ds, val_ds=val_ds, strategy=strategy)

    with open(decks.DATA_DIR / f"history/{net_id}.json", "w") as f:
        json.dump(history, f)

    print("\nEvaluation:")
    net.evaluate(test_ds)
    print(f"\nTraining for {net_id} FINISHED (time: {dt.datetime.utcnow() - start}).")
