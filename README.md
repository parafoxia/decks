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
docker run --gpus all -it decks python -m decks <gen> -e <epochs> -b [batch size]
```

Run `python -m decks --help` for more information about command arguments.

## Distributed training

To use distributed training:

```sh
docker run --gpus all --ipc=host -it decks python -m decks <gen> -e <epochs> -b [batch size] -d
```

Note that if `--ipc=host` is not passed, Tensorflow will error.
Make sure this Docker flag is specified when passing the `-d` or `--distributed` flags.
