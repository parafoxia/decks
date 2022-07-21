import amber
from amber.nets.utils import cp_callback

def train(epochs, nid, *, train_ds, val_ds):
    cp_path = amber.DATA_DIR / f"checkpoints/{nid}" / "cp-{epoch:04d}.ckpt"

    net = amber.build_net(train_ds)
    history = net.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[cp_callback(cp_path, 10 * len(train_ds))],
        verbose=2,
    )

    return net, history.history