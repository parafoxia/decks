# DECKS

The Dimensional Emotion-Contextualising Kinetic System.

## Prerequisits

- docker

## Training DECKS

Clone this repo, then build the Docker image:

```sh
docker build -t decks .
```

To train models:

```sh
docker run --gpus <devices> -it decks python -m decks <gen> -e <epochs> -b [batch_size]
```

Arguments:

- `devices` - A Docker flag for GPU devices. See below for more information.
- `gen` - The model generation.
- `epochs` - The number of epochs to train for.
- `batch_size` (optional) - The size of data batches.

### Specifying devices

| Use case         | Command                 |
| ---------------- | ----------------------- |
| Only use GPU 0   | `--gpus '"device=0"'`   |
| Use GPUs 1 and 3 | `--gpus '"device=1,3"'` |
| Use all GPUs     | `--gpus all`            |

**Note:** If more than one GPU device is made available, distributed training is automatically enabled.
