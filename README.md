# DECKS

The Dimensional Emotion-Contextualising Kinetic System.

## Prerequisits

- docker

## Training DECKS

Clone this repo, then build the Docker image:

```sh
docker build -t decks:<gen> .
```

To train models:

```sh
docker run --gpus <devices> --ipc=host -it decks:<gen> python -m decks <gen> <-e epochs> [-b batch_size] [-i epoch]
```

Arguments:

- `devices` - A Docker flag for GPU devices. See below for more information.
- `gen` - The model generation.
- `-e epochs` - The number of epochs to train for.
- `-b batch_size` (optional) - The size of data batches.
- `-i epoch` (optional) - Continue training from the given epoch. The generation, epochs, and batch size must have been used for a previous model, otherwise an error will be raised.

**Note:** `ipc=host` can be omitted if only one GPU device is made available.

### Specifying devices

| Use case         | Command                 |
| ---------------- | ----------------------- |
| Only use GPU 0   | `--gpus '"device=0"'`   |
| Use GPUs 1 and 3 | `--gpus '"device=1,3"'` |
| Use all GPUs     | `--gpus all`            |

**Note:** If more than one GPU device is made available, distributed training is automatically enabled.

## Evaluating DECKS

Several notebooks have been provided in the `notebooks` directory.
These will need to moved to the root directory to run.
