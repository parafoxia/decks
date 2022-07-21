import argparse
import json
import os
import datetime as dt

import amber

def cli():
    if not amber.DATA_DIR.is_dir():
        os.makedirs(amber.DATA_DIR / "checkpoints")
        os.makedirs(amber.DATA_DIR / "history")

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", help="The number of epochs to train for.", type=int)
    parser.add_argument("-b", "--batch-size", help="The batch size to use.", type=int, default=64)
    ns = parser.parse_args()

    if ns.epochs % 10 != 0:
        raise ValueError("the number of epochs must be a multiple of 10")

    n_files = len(os.listdir(amber.DATA_DIR / "history"))
    net_id = f"{n_files + 1:04}-{ns.epochs:04}-{ns.batch_size:04}"
    train_ds, test_ds, val_ds = amber.load_carer(ns.batch_size)

    start = dt.datetime.utcnow()
    net, history = amber.train(ns.epochs, net_id, train_ds=train_ds, val_ds=val_ds)

    with open(amber.DATA_DIR / f"history/{net_id}.json", "w") as f:
        json.dump(history, f)

    print("\nEvaluation:")
    net.evaluate(test_ds)
    print(f"\nTraining for {net_id} FINISHED (time: {dt.datetime.utcnow() - start}).")
